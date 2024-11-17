import torch
from transformers import BertForQuestionAnswering, BertTokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import default_data_collator
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === Ayarlar ===
MODEL_PATH = "./chatbot/models/trained_model"
DATASET_PATH = "./data/qac_medeni_kanun.json"
SAVE_PATH = "./chatbot/models/fine_tuned_model"
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Veri Seti ===
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        context = item["context"]
        answer = item["answer"]
        start_idx = context.find(answer)

        if start_idx == -1:
            raise ValueError(f"Answer '{answer}' not found in context.")

        end_idx = start_idx + len(answer)

        # Tokenize data
        inputs = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Calculate start and end token positions
        offset_mapping = inputs["offset_mapping"].squeeze()
        input_ids = inputs["input_ids"].squeeze()

        start_positions = []
        end_positions = []
        for idx, (start, end) in enumerate(offset_mapping):
            if start <= start_idx < end:
                start_positions.append(idx)
            if start < end_idx <= end:
                end_positions.append(idx)

        if not start_positions or not end_positions:
            raise ValueError("Could not find token positions for the answer.")

        inputs["start_positions"] = torch.tensor(start_positions[0], dtype=torch.long)
        inputs["end_positions"] = torch.tensor(end_positions[-1], dtype=torch.long)

        inputs.pop("offset_mapping")  # Offset mapping is no longer needed

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "start_positions": inputs["start_positions"],
            "end_positions": inputs["end_positions"]
        }


# === Model ve Tokenizer ===
print("Loading model and tokenizer...")
model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.to(DEVICE)

# === Veri Setini Yükleme ===
print("Loading dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

train_dataset = QADataset(train_data, tokenizer)
val_dataset = QADataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=default_data_collator)

# === Eğitim Ayarları ===
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# === Eğitim Fonksiyonu ===
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        start_positions = batch["start_positions"].to(DEVICE)
        end_positions = batch["end_positions"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# === Doğrulama Fonksiyonu ===
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            start_positions = batch["start_positions"].to(DEVICE)
            end_positions = batch["end_positions"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# === Eğitim Döngüsü ===
print("Starting training...")
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = evaluate(model, val_loader)
    print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

# === Eğitilmiş Modeli Kaydetme ===
print("Saving fine-tuned model...")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("Fine-tuning completed and model saved!")
