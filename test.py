import os
import fitz
import json
import google.generativeai as genai
import dotenv
from tqdm import tqdm
import time

dotenv.load_dotenv()

genai.configure(api_key=os.getenv('API_KEY'))

model = genai.GenerativeModel('gemini-pro')

def extract_text_from_pdf(pdf_file_path):
    text = ""
    pdf_document = fitz.open(pdf_file_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
        
    text = text.replace('\n', ' ')
        
    return text

input_folder_path = "/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/_Dossiês IA"
output_folder_path = "/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/negative"

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

prompt = """
Revise o documento original e faça as seguintes modificações para gerar exemplos negativos:

1. Introduza erros de digitação em palavras-chave importantes, como "Matrícula", "Proprietários", "Área do Georreferenciamento", etc.
2. Altere algumas informações para torná-las inconsistentes com o restante do documento, como os nomes dos proprietários, números de CPF, datas, códigos INCRA, etc.
3. Introduza frases ou parágrafos que pareçam confusos ou ambíguos, especialmente em seções como "Objetivo do Requerimento", "Certificação INCRA", "ART/TRT", etc.
4. Mude o estilo de escrita de algumas seções para algo menos formal ou técnico, adicionando expressões coloquiais ou linguagem menos técnica.
5. Adicione informações irrelevantes ou falsas em determinadas partes do documento, como detalhes sobre proprietários ou características do imóvel que não estão presentes no documento original.
6. Altere a ordem ou a estrutura das seções para torná-las menos lógicas, como trocar a ordem das seções "Requerimento" e "Certificação INCRA".
7. Introduza contradições entre diferentes partes do documento, como inconsistências entre a área do georreferenciamento e a área da matrícula.
8. Faça alterações que tornem o texto menos claro ou mais difícil de entender, como introduzir frases mal formuladas, erros de concordância ou informações confusas.

Por favor, use essas diretrizes para gerar exemplos negativos com base no documento original. Certifique-se de manter o contexto geral do documento, mas faça as modificações necessárias para tornar os exemplos negativos distintos dos originais.
"""

output_json_path = os.path.join(output_folder_path, "output.json")
negative_texts = []

# Use tqdm to track progress
for i, filename in enumerate(tqdm(os.listdir(input_folder_path))):
    if filename.endswith(".pdf"):
        pdf_file_path = os.path.join(input_folder_path, filename)
        documento_texto = extract_text_from_pdf(pdf_file_path)
        
        response = model.generate_content(f'{prompt} {documento_texto}')
        
        # Append the response text as a separate list within the main list
        negative_texts.append([response.text])

        # Salvar os resultados a cada 3 arquivos processados
        if (i + 1) % 2 == 0:
            with open(output_json_path, 'w', encoding='utf-8') as output_json_file:
                json.dump({"documents": negative_texts}, output_json_file, ensure_ascii=False)

            print(f"Resultados parciais salvos em {output_json_path}")
            time.sleep(60)  # Pausar para evitar sobrecarregar a API

# Salvar os resultados finais
if negative_texts:
    with open(output_json_path, 'w', encoding='utf-8') as output_json_file:
        json.dump({"documents": negative_texts}, output_json_file, ensure_ascii=False)

    print(f"Resultados finais salvos em {output_json_path}")
