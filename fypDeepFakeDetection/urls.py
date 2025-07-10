from django.contrib import admin
from django.urls import path
from .views import Home, Image, Video, predict_page, About, Login, Register, logoutUser, Profile, Voice

urlpatterns = [
    path('', Home, name="home"),
    path('images/', Image, name="image"),
    path('videos/', Video, name="video"),
    path('voice/', Voice, name="voice"),
    path('predict/', predict_page, name='predict'),
    path('about/', About, name='about'),
    path('login/', Login, name='login'),
    path('register/', Register, name='register'),
    path('logout-user/', logoutUser, name='logout'),
    path('profile/', Profile, name='profile'),
]


