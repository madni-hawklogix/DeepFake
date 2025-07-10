# models.py

from django.db import models
from django.contrib.auth.models import User  # Assuming you're using Django's built-in User model
from django.utils import timezone
import json


class UserImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Link to the User model
    original_image = models.CharField(max_length=255)
    processed_image = models.CharField(max_length=255)
    result = models.CharField(max_length=10)
    fake_prediction = models.FloatField()
    real_prediction = models.FloatField()
    # created_at = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.username}'s image - {self.result}"


class UserVoice(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    original_voice = models.CharField(max_length=255)
    result = models.CharField(max_length=10)
    fake_prediction = models.FloatField()
    real_prediction = models.FloatField()
    # created_at = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.username}'s voice - {self.result}"


class VideoProcessingResult(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    original_video = models.CharField(max_length=255)
    models_location = models.CharField(max_length=255)
    output = models.CharField(max_length=10)
    confidence = models.FloatField()
    preprocessed_images = models.JSONField(default=list)
    faces_cropped_images = models.JSONField(default=list)
    heatmap_images = models.JSONField(default=list)
    # created_at = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.username}'s video - {self.output}"

    def set_preprocessed_images(self, images):
        self.preprocessed_images = images
        self.save()

    def set_faces_cropped_images(self, images):
        self.faces_cropped_images = images
        self.save()

    def set_heatmap_images(self, images):
        self.heatmap_images = images
        self.save()

    def get_preprocessed_images(self):
        return self.preprocessed_images

    def get_faces_cropped_images(self):
        return self.faces_cropped_images

    def get_heatmap_images(self):
        return self.heatmap_images
