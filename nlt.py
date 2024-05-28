import fitz  # PyMuPDF
from spellchecker import SpellChecker

def extract_text_from_pdf(pdf_file_path):
    text = ""
    pdf_document = fitz.open(pdf_file_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
        
    # Remover quebras de linha
    text = text.replace('\n', ' ')
    
    # Converter para UTF-8
    text = text.encode('utf-8').decode('utf-8')

    return text

def identify_spelling_errors(text):
    spell = SpellChecker(language='pt')  # Definindo o idioma como português

    # Tokenizando o texto em palavras
    words = text.split()

    # Identificando palavras escritas incorretamente
    misspelled = spell.unknown(words)

    return misspelled

def main():
    pdf_file_path = "/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/_Dossiês IA/Dossiê José Otávio .pdf"

    print("Extraindo texto do PDF...")
    documento_texto = extract_text_from_pdf(pdf_file_path)

    print("Identificando erros de ortografia...")
    spelling_errors = identify_spelling_errors(documento_texto)

    print("Erros de ortografia identificados:")
    print(spelling_errors)

if __name__ == "__main__":
    main()
