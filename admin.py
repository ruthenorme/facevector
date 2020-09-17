import threading

from django.contrib import admin
from django.contrib.auth.models import Group
from django.http import HttpResponseRedirect
from django.urls import path

from detection.detect import train_data

from .forms import UserDatasetForm
from .forms import NegativeDatasetForm
from .models import UserRecordModel
from .models import UserDatasetModel
from .models import UserDetectionModel
from .models import UserNegativeModel
from .models import UserRetrainingModel
from .models import CameraModel


@admin.register(CameraModel)
class CameraModelAdmin(admin.ModelAdmin):
    fields = ('url',)
    list_display = ('url',)


@admin.register(UserRecordModel)
class UserRecordModelAdmin(admin.ModelAdmin):
    fields = ('first_name', 'middle_name', 'last_name',)
    list_display = ('first_name', 'middle_name', 'last_name', 'created_datetime', 'updated_datetime',)


@admin.register(UserNegativeModel)
class UserNegativeDatasetModelAdmin(admin.ModelAdmin):
    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('train-data/', self.train_data),
        ]
        return my_urls + urls

    def train_data(self, request):
        threading.Thread(target=train_data).start()
        self.message_user(request, "Training data in progress")
        return HttpResponseRedirect("../")

    change_list_template = 'detection/detection_changelist.html'
    fields = ('image',)
    list_display = ('image', 'created_datetime')
    form = NegativeDatasetForm


@admin.register(UserDatasetModel)
class UserDatasetModelAdmin(admin.ModelAdmin):

    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('train-data/', self.train_data),
        ]
        return my_urls + urls

    def train_data(self, request):
        threading.Thread(target=train_data).start()
        self.message_user(request, "Training data in progress")
        return HttpResponseRedirect("../")

    change_list_template = 'detection/detection_changelist.html'
    fields = ('user', 'image',)
    list_display = ('user', 'image', 'created_datetime')
    form = UserDatasetForm


@admin.register(UserDetectionModel)
class UserDetectionModelAdmin(admin.ModelAdmin):
    fields = ('user',)
    list_display = ('user', 'datetime',)


@admin.register(UserRetrainingModel)
class UserRetrainingModelAdmin(admin.ModelAdmin):
    fields = ('datetime', 'accuracy',)
    list_display = ('datetime', 'accuracy',)


admin.site.site_header = 'Face Vector'
admin.site.site_title = 'Face Vector'

admin.site.unregister(Group)
