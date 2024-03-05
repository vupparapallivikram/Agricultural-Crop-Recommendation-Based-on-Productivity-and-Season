from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class crop_details(models.Model):

    State_Name= models.CharField(max_length=300)
    District_Name= models.CharField(max_length=300)
    Crop_Year= models.CharField(max_length=300)
    Season= models.CharField(max_length=300)
    names= models.CharField(max_length=300)
    Area= models.CharField(max_length=300)
    Production= models.CharField(max_length=300)

class crop_prediction(models.Model):

    State_Name = models.CharField(max_length=300)
    District_Name = models.CharField(max_length=300)
    Crop_Year = models.CharField(max_length=300)
    Season = models.CharField(max_length=300)
    names = models.CharField(max_length=300)
    Area = models.CharField(max_length=300)
    Production = models.CharField(max_length=300)
    Yield_Prediction= models.CharField(max_length=300)
    Production_Prediction= models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)


