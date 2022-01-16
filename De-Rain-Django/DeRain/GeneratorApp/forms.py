from django import forms
from .models import ImageAdib


class ImageForm(forms.ModelForm):
    class Meta:
        model = ImageAdib
        fields = '__all__'
        labels = {"inp_img": ""}
