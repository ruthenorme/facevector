from django.conf import settings
from django.utils import timezone

from detection.models import UserRetrainingModel

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import datetime
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

from detection.models import UserDetectionModel
from detection.models import UserRecordModel


def train_data():
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.join(settings.BASE_DIR, "face_detection_model/deploy.prototxt")
    modelPath = os.path.join(settings.BASE_DIR, "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(os.path.join(settings.BASE_DIR, "openface_nn4.small2.v1.t7"))

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(os.path.join(settings.BASE_DIR, "media")))

    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    knownEmbeddings = []
    knownNames = []

    # initialize the total number of faces processed
    total = 0

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # add the name of the person + corresponding face
                # embedding to their respective lists
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(os.path.join(settings.BASE_DIR, "output/embeddings.pickle"), "wb")
    f.write(pickle.dumps(data))
    f.close()

    # load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(os.path.join(settings.BASE_DIR, "output/embeddings.pickle"), "rb").read())

    # encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")
    X_train, X_test, y_train, y_test = train_test_split(data["embeddings"], labels)
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open(os.path.join(settings.BASE_DIR, "output/recognizer.pickle"), "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(os.path.join(settings.BASE_DIR, "output/le.pickle"), "wb")
    f.write(pickle.dumps(le))
    f.close()

    # Get prediction
    predicted = recognizer.predict(X_test)
    prediction_percent = accuracy_score(y_test, predicted)
    print("Accuracy: " + str(prediction_percent))

    print("Training successful!")
    training = UserRetrainingModel(accuracy=float(prediction_percent))
    training.save()


def detect_faces(input_image):
    persons = []

    # load our serialized face detector from disk
    protoPath = os.path.join(settings.BASE_DIR, "face_detection_model/deploy.prototxt")
    modelPath = os.path.join(settings.BASE_DIR, "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load our serialized face embedding model from disk
    embedder = cv2.dnn.readNetFromTorch(os.path.join(settings.BASE_DIR, "openface_nn4.small2.v1.t7"))

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(os.path.join(settings.BASE_DIR, "output/recognizer.pickle"), "rb").read())
    le = pickle.loads(open(os.path.join(settings.BASE_DIR, "output/le.pickle"), "rb").read())

    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    image = input_image
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated
            # probability
            text = "{}".format(name)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            persons.append(text)

    for person in persons:
        print(f"Person: {person}")
        person_names = person.split(' ')

        #
        #   For unknown
        #
        if person_names[0].lower() == 'unknown':
            if not UserDetectionModel.objects.filter(user=None,
                                                     datetime__gt=timezone.now() - timezone.timedelta(
                                                         seconds=5)).exists():
                print("Adding new unknown user")
                UserDetectionModel(user=None).save()

        #
        #   For users
        #
        elif person_names[0].lower != 'unknown' and not UserDetectionModel.objects.filter(
                user__first_name__icontains=person_names[0],
                user__last_name__icontains=person_names[-1],
                datetime__gt=timezone.now() - datetime.timedelta(hours=1)).exists():
            print("Adding new recorded user")
            user = UserDetectionModel(
                user=UserRecordModel.objects.get(first_name__icontains=person_names[0],
                                                 last_name__icontains=person_names[-1]))
            user.save()

        else:
            pass

    return image
