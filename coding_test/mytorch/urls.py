from django.urls import path
from mytorch import views
urlpatterns = [
    path('',views.index),
    path('modeling/',views.modeling )

]