import os

from django.conf import settings
from django.db import models


class TimestampAbstractModel(models.Model):
    created_datetime = models.DateTimeField(auto_now_add=True, verbose_name='Created datetime')
    updated_datetime = models.DateTimeField(auto_now=True, verbose_name='Updated datetime')

    class Meta:
        abstract = True


class UserRecordModel(TimestampAbstractModel):
    first_name = models.CharField(max_length=64, verbose_name='First name')
    middle_name = models.CharField(max_length=64, null=True, blank=True, verbose_name='Middle name')
    last_name = models.CharField(max_length=64, verbose_name='Last name')

    class Meta:
        db_table = 'user_record'
        verbose_name = 'User Record'
        verbose_name_plural = 'User Records'

    def __str__(self):
        return f"{self.first_name}{' ' + self.middle_name if self.middle_name else ''} {self.last_name}"


class UserDatasetModel(TimestampAbstractModel):
    def upload_path(self, filename):
        return str(self.user) + '/' + filename

    user = models.ForeignKey(UserRecordModel, on_delete=models.CASCADE, verbose_name='User')
    image = models.ImageField(upload_to=upload_path, verbose_name='Image')

    class Meta:
        db_table = 'users_dataset'
        verbose_name = 'Dataset'
        verbose_name_plural = 'Datasets'

    def __str__(self):
        return f"{self.user} - {self.image}"


class UserNegativeModel(TimestampAbstractModel):
    def upload_path(self, filename):
        return 'Unknown' + '/' + filename

    image = models.ImageField(upload_to=upload_path, verbose_name='Image')

    class Meta:
        db_table = 'users_negative'
        verbose_name = 'Negative Dataset'
        verbose_name_plural = 'Negative Datasets'

    def __str__(self):
        return f"{self.image}"


class UserDetectionModel(models.Model):
    user = models.ForeignKey(UserRecordModel, null=True, blank=True, on_delete=models.SET_NULL, verbose_name='User')
    datetime = models.DateTimeField(auto_now_add=True, verbose_name='Datetime')

    class Meta:
        db_table = 'users_detection'
        verbose_name = 'Detection'
        verbose_name_plural = 'Detections'

    def __str__(self):
        return f"{self.user} - {self.datetime}"


class UserRetrainingModel(models.Model):
    accuracy = models.FloatField(default=0.0, verbose_name='Accuracy')
    datetime = models.DateTimeField(auto_now_add=True, verbose_name='Datetime')

    class Meta:
        db_table = 'user_retraining'
        verbose_name = 'Retraining'
        verbose_name_plural = 'Retrainings'

    def __str__(self):
        return f"{self.datetime}"


class CameraModel(TimestampAbstractModel):
    url = models.CharField(max_length=500, verbose_name='URL')

    class Meta:
        db_table = 'camera'
        verbose_name = 'Camera'
        verbose_name_plural = 'Cameras'

    def __str__(self):
        return f"Camera {self.url}"
