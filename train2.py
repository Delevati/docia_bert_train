from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import json
import re
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Carregar modelo e tokenizer do mBERT
model_name = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Carregar documentos rotulados
with open('/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/positive/organized_positive_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extraindo textos e rótulos do JSON
documents = [item['text'] for item in data]
labels = [item['label'] for item in data]

# Preparar dados de treinamento
def process_data(texts, labels):
    processed_data = []
    for doc, label in tqdm(zip(texts, labels), total=len(texts), desc="Processando documentos"):
        encoding = tokenizer.encode_plus(doc, return_tensors='pt', padding='max_length', truncation=True)
        processed_data.append({
            "input_ids": encoding['input_ids'].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "labels": torch.tensor(label).unsqueeze(0),
        })
    return processed_data

processed_data = process_data(documents, labels)

# Preparar dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

dataset = CustomDataset(processed_data)

# Definir função para calcular métricas
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
    }

# Definir parâmetros de treinamento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='steps',
    eval_steps=100,
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
)

# Treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Treinamento
trainer.train()
trainer.save_model('/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/results')

# Avaliação
results = trainer.evaluate(dataset)
print(results)
