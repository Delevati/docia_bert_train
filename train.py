from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import json
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Carregar modelo e tokenizer do mBERT
model_name = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Carregar documentos e labels
with open('/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/combinado.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extraindo textos e rótulos do JSON combinado
documents = [item['text'] for item in data]
labels = [item['label'] for item in data]

# Dividir o conjunto de dados em treinamento e avaliação
train_texts, eval_texts, train_labels, eval_labels = train_test_split(documents, labels, test_size=0.2, random_state=42, stratify=labels)

# Carregar o checklist
with open('/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/checklist.json', 'r', encoding='utf-8') as f:
    checklist_data = json.load(f)

# Extrair o texto do checklist
checklist_text = checklist_data['text']

# Preparar dados com o checklist para treinamento
def process_data(texts, labels):
    processed_data = []
    for doc, label in tqdm(zip(texts, labels), total=len(texts), desc="Processando documentos"):
        # Verificar se o item do checklist está presente no documento
        label_presence = 1 if re.search(re.escape(checklist_text), doc.lower()) else 0
        
        # Codificar o documento com o checklist
        encoding = tokenizer.encode_plus(doc, checklist_text, return_tensors='pt', padding='max_length', truncation=True)
        processed_data.append({
            "input_ids": encoding['input_ids'].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "labels": torch.tensor(label).unsqueeze(0),  # Manter o rótulo original
            "label_presence": label_presence  # Rótulo baseado na presença do item
        })
    return processed_data

processed_train_data = process_data(train_texts, train_labels)
processed_eval_data = process_data(eval_texts, eval_labels)

# Preparar datasets
class CheckListDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

train_dataset = CheckListDataset(processed_train_data)
eval_dataset = CheckListDataset(processed_eval_data)

# Definir função para calcular métricas adicionais
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
    evaluation_strategy='steps',  # Avaliar a cada N steps
    eval_steps=100,  # Avaliar a cada 100 steps
    save_steps=1000,  # Salvar a cada 1000 steps
    save_total_limit=2,  # Limitar o número total de modelos salvos (melhor e último)
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',  # Métrica usada para determinar o melhor modelo
)

# Treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Treinamento
trainer.train()

# Avaliação
results = trainer.evaluate(eval_dataset)
print(results)
