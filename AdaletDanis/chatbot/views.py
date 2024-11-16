from django.shortcuts import render
from transformers import pipeline

# Hugging Face modelini yükle
nlp = pipeline("question-answering", model="deepset/roberta-base-squad2")

def ask_question(request):
    answer = ""
    if request.method == 'POST':
        question = request.POST.get('question')
        context = "Burada Türk Hukuku ile ilgili bilgi bulunacak."  # Türk Hukuku ile ilgili metin
        answer = nlp(question=question, context=context)['answer']
    return render(request, 'chatbot/index.html', {'answer': answer})

def index(request):
    return render(request, 'chatbot/index.html')
