from django.shortcuts import render
from .forms import ImageForm
from .models import ImageAdib
from .website_image import *
from skimage import io
import os

# Create your views here.


def home(request):
    old_img = ImageAdib.objects.all()
    for img in old_img:
        img.inp_img.delete(save=True)

    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
        main()
        img = ImageAdib.objects.latest("id")
        return render(request, 'GeneratorApp/home.html', {"img":img, "form":form})
    form = ImageForm()
    return render(request, 'GeneratorApp/home.html', {"img":None, "form":form})
        
    
    


def result(request):
    return render(request, 'GeneratorApp/result.html')