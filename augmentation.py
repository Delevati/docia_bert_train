import os
import random
import json
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

def synonym_replacement(text, n=1):
    words = text.split()
    for _ in range(n):
        idx = random.randint(0, len(words) - 1)
        synonyms = wordnet.synsets(words[idx])
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            words[idx] = synonym
    return ' '.join(words)

def word_insertion(text, n=1):
    words = text.split()
    for _ in range(n):
        idx = random.randint(0, len(words))
        words.insert(idx, 'random_word')
    return ' '.join(words)

def word_elimination(text, n=1):
    words = text.split()
    for _ in range(n):
        if words:
            idx = random.randint(0, len(words) - 1)
            del words[idx]
    return ' '.join(words)

def word_permutation(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def process_existing_json(input_json_path, output_json_path, num_negative_examples=10):
    data = load_json(input_json_path)
    
    processed_examples = []
    
    for text in tqdm(data, desc="Processing examples"):
        examples = []
        for _ in range(num_negative_examples):
            # Escolher aleatoriamente uma técnica de data augmentation
            augmentation_function = random.choice([synonym_replacement, word_insertion, word_elimination, word_permutation])
            augmented_text = augmentation_function(text)
            examples.append(augmented_text)  # Adiciona o texto diretamente sem listas aninhadas
        
        processed_examples.extend(examples)
    
    # Salvar os exemplos processados em um arquivo JSON com a label "negative"
    output_data = {"documents": processed_examples, "label": "negative"}
    save_to_json(output_data, output_json_path)
    print(f"Resultados salvos em {output_json_path}")

# Defina o caminho para o arquivo JSON de entrada e o arquivo JSON de saída
input_json_path = "/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/negative/output.json"
output_json_path = "/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/negative/processed_negative_examples.json"

# Chame a função para processar o arquivo JSON existente
process_existing_json(input_json_path, output_json_path)
