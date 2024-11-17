from django.shortcuts import render
from transformers import pipeline
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import pandas as pd


# Model ve tokenizer'ın yükleneceği path
MODEL_PATH = "./chatbot/models/fine_tuned_model"

# Hugging Face pipeline oluşturuluyor
qa_pipeline = pipeline("question-answering", model=MODEL_PATH, tokenizer=MODEL_PATH)

# Veri setini yükle
DATASET_PATH = "./data/qac_medeni_kanun.json"
dataset = pd.read_json(DATASET_PATH)

# Kullanıcının sorduğu soruya veri setine dayalı cevap üreten view
def ask_question(request):
    answer = ""
    if request.method == 'POST':
        question = request.POST.get('question')  # Kullanıcıdan gelen soru
        if question:
            try:
                # Veri setinden en uygun bağlamı bul
                most_similar_context = find_best_context(question)
                if not most_similar_context:
                    answer = "Veri setinde bu soruya uygun bir bağlam bulunamadı."
                else:
                    # Pipeline ile cevap oluşturuluyor
                    result = qa_pipeline(question=question, context=most_similar_context)
                    answer = result['answer']
            except Exception as e:
                answer = f"Bir hata oluştu: {str(e)}"
    return render(request, 'chatbot/index.html', {'answer': answer})

def index(request):
    return render(request, 'chatbot/index.html')

# Veri setinden en uygun bağlamı bulan yardımcı fonksiyon
def find_best_context(user_question):
    """
    Kullanıcı sorusuna en uygun bağlamı veri setinden bulur.
    """
    # Basit bir benzerlik ölçütü (örneğin: 'question' sütununda kullanıcı sorusuna benzeyen soru)
    dataset['similarity'] = dataset['question'].apply(lambda q: calculate_similarity(user_question, q))
    best_match = dataset.sort_values(by='similarity', ascending=False).iloc[0]
    if best_match['similarity'] > 0.5:  # Belirli bir eşik değeri kontrolü
        return best_match['context']
    return None

def calculate_similarity(user_question, dataset_question):
    """
    Basit bir metin benzerliği hesaplama fonksiyonu (örn: ortak kelime oranı).
    Daha gelişmiş bir benzerlik ölçütü için cosine similarity veya pre-trained bir model kullanılabilir.
    """
    user_words = set(user_question.lower().split())
    dataset_words = set(dataset_question.lower().split())
    return len(user_words & dataset_words) / len(user_words | dataset_words)










def index(request):
    return render(request, 'chatbot/index.html')
