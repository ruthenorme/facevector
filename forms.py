from django import forms

from natsort import natsorted

from .models import UserDatasetModel
from .models import UserNegativeModel


class NegativeDatasetForm(forms.ModelForm):
    image = forms.ImageField(widget=forms.FileInput(attrs={'multiple': True}), required=True)

    class Meta:
        model = UserNegativeModel
        fields = ['image']

    def save(self, *args, **kwargs):
        file_list = natsorted(self.files.getlist('image'), key=lambda file: file.name)

        self.instance.image = file_list[0]
        for file in file_list[1:]:
            UserNegativeModel.objects.create(
                image=file,
            )

        return super().save(*args, **kwargs)


class UserDatasetForm(forms.ModelForm):
    image = forms.ImageField(widget=forms.FileInput(attrs={'multiple': True}), required=True)

    class Meta:
        model = UserDatasetModel
        fields = ['user', 'image']

    def save(self, *args, **kwargs):
        file_list = natsorted(self.files.getlist('image'), key=lambda file: file.name)

        self.instance.image = file_list[0]
        for file in file_list[1:]:
            UserDatasetModel.objects.create(
                user=self.cleaned_data['user'],
                image=file,
            )

        return super().save(*args, **kwargs)
