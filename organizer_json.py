import json
import os
import random

# # varios no mesmo json
# with open('/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/negative/output.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # Transformar os documentos no formato esperado, adicionando o rótulo
# formatted_data = []
# for document in data['documents']:
#     formatted_data.append({"text": document, "label": 0})

# # Salvar o novo JSON formatado
# with open('/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/negative/organized_output.json', 'w', encoding='utf-8') as f:
#     json.dump(formatted_data, f, indent=4, ensure_ascii=False)


# # diversos arquivos
# def format_json_files(input_folder, output_path):
#     # Lista para armazenar todos os documentos positivos
#     positive_documents = []

#     # Ler documentos de cada arquivo JSON no diretório de entrada
#     for file_name in os.listdir(input_folder):
#         if file_name.endswith('.json'):
#             file_path = os.path.join(input_folder, file_name)
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 # Adicionar documentos à lista de documentos positivos
#                 positive_documents.extend(data)

#     # Transformar os documentos no formato esperado, adicionando o rótulo
#     formatted_positive_data = [{"text": document, "label": 1} for document in positive_documents]

#     # Salvar o novo JSON formatado
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(formatted_positive_data, f, indent=4, ensure_ascii=False)

#     print(f'Documentos positivos organizados foram salvos em: {output_path}')

# # Caminho para a pasta contendo os arquivos JSON positivos
# input_folder = '/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/positive'

# # Caminho para salvar o novo JSON formatado
# output_path = '/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/positive/organized_positive_output.json'

# # Chamar a função para formatar os arquivos JSON
# format_json_files(input_folder, output_path)

# def merge_json_files(positive_file, negative_file, output_file):
#     # Abrir arquivos JSON de exemplos positivos e negativos
#     with open(positive_file, 'r', encoding='utf-8') as f:
#         positive_data = json.load(f)

#     with open(negative_file, 'r', encoding='utf-8') as f:
#         negative_data = json.load(f)

#     # Extrair os textos dos exemplos positivos e negativos
#     positive_texts = extract_texts(positive_data)
#     negative_texts = extract_texts(negative_data)

#     # Misturar exemplos positivos e negativos alternadamente
#     merged_data = []
#     min_len = min(len(positive_texts), len(negative_texts))
#     for i in range(min_len):
#         merged_data.append({"text": negative_texts[i], "label": 0})
#         merged_data.append({"text": positive_texts[i], "label": 1})

#     # Salvar o novo JSON formatado
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(merged_data, f, indent=4, ensure_ascii=False)

#     print(f'Dados combinados foram salvos em: {output_file}')

# def extract_texts(data):
#     texts = []
#     for item in data:
#         # Verificar se o texto está em um formato aninhado
#         if isinstance(item, dict) and 'text' in item:
#             text = item['text']
#             # Verificar se há mais um nível de aninhamento
#             if isinstance(text, dict) and 'text' in text:
#                 text = text['text']
#             texts.append(text)
#         else:
#             # Se não estiver em um formato aninhado, assumir que está no formato padrão
#             texts.append(item)
#     return texts

# # Caminho para o arquivo JSON contendo exemplos positivos
# positive_file = '/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/positive/organized_positive_output.json'

# # Caminho para o arquivo JSON contendo exemplos negativos
# negative_file = '/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/negative/organized_negative_output.json'

# # Caminho para salvar o novo JSON combinado
# output_file = '/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/combinado.json'

# # Chamar a função para mesclar os arquivos JSON
# merge_json_files(positive_file, negative_file, output_file)

def text_to_json(input_file, output_file):
    # Ler o conteúdo do arquivo de texto
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Formatar a string em um formato JSON aceito pelo BERT
    data = {"text": text}

    # Escrever os dados formatados em um arquivo JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Caminho para o arquivo de texto de entrada
input_file = '/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/Check List Georreferenciamento - 2023 [Comentado].txt'

# Caminho para o arquivo JSON de saída
output_file = 'checklist.json'

# Converter o arquivo de texto para JSON
text_to_json(input_file, output_file)
