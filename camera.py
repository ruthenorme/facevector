from django.http import StreamingHttpResponse
from django.http import HttpResponseServerError
from django.views.decorators import gzip
from django.conf import settings

import cv2
import vlc
import os

from .detect import detect_faces
from .models import CameraModel


class VideoCamera(object):
    def __init__(self, id):
        camera = CameraModel.objects.get(id=int(id))
        self.id = id
        self.url = str(camera.url)
        i = vlc.Instance(
            "--verbose=1 --network-caching=250 --vout=dummy --no-snapshot-preview --no-osd --transform-type=hflip")
        self.video = i.media_player_new()
        self.video.set_mrl(self.url)
        self.video.audio_set_mute(True)
        self.video.play()


def get_frame(self):
    if self.video.video_take_snapshot(0, os.path.join(settings.BASE_DIR, self.id + '.png'), 0, 0) == 0:
        image = cv2.imread(self.id + '.png', 1)
    else:
        image = cv2.imread(os.path.join(settings.BASE_DIR, 'black.png'))

    image = detect_faces(image)

    ret, jpeg = cv2.imencode('.jpg', image)
    return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


video_cameras = {}
cameras = CameraModel.objects.all()
for camera in cameras:
    video_cameras[str(camera.id)] = VideoCamera(str(camera.id))


@gzip.gzip_page
def camera_stream(request):
    global video_cameras

    try:
        return StreamingHttpResponse(gen(video_cameras[request.GET.get('id')]),
                                     content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("aborted: {}".format(e))
