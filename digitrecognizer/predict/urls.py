from django.urls import path
from .views import upload_image,index
urlpatterns = [
    path('',index),
    path('predict/',upload_image,name='predict'),
]
