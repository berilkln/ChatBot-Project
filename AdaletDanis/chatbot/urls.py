from django.urls import path
from . import views

app_name = 'chatbot'

urlpatterns = [
    path('', views.index, name='index'),
    path('ask_question/', views.ask_question, name='ask_question'),
]
