from transformers import BertForQuestionAnswering, BertTokenizer

# Hugging Face modelini indir ve belirli bir dizine kaydet
model_name = "dbmdz/bert-base-turkish-cased"
save_directory = "./chatbot/models/trained_model"  

# Model ve tokenizer'ı indirip kaydet
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Kaydet
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
