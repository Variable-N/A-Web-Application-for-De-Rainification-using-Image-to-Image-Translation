from distutils.command import upload
from django.db import models

# Create your models here.

class ImageAdib(models.Model):
    inp_img = models.ImageField(upload_to="my_image")

