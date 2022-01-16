from django.contrib import admin
from .models import ImageAdib
# Register your models here.


@admin.register(ImageAdib)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'inp_img']
