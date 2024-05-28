import torch
from transformers import BertTokenizer, BertForSequenceClassification
import fitz  # PyMuPDF

class TextCorrectionModel:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def identify_errors(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        chunks = [token_ids[i:i + 512] for i in range(0, len(token_ids), 512)]

        errors = []
        for chunk in chunks:
            inputs = self.tokenizer.prepare_for_model(chunk, return_tensors="pt", padding='max_length', truncation=True, max_length=512, add_special_tokens=False)
            inputs = {key: value.unsqueeze(0).to(self.device) for key, value in inputs.items()}  # Add batch dimension

            # Debugging: Print input shapes
            print("Input shapes:")
            for key, value in inputs.items():
                print(f"{key}: {value.shape}")

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            chunk_errors = self._extract_errors_from_predictions(inputs, predictions)
            errors.extend(chunk_errors)
        
        return errors

    def _extract_errors_from_predictions(self, inputs, predictions):
        errors = []
        input_ids = inputs['input_ids'].cpu().numpy()[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Debugging: Print predictions
        print("Predictions:")
        print(predictions)
        
        if predictions.ndim == 1:  # Handle the case where predictions is a 1D array
            predictions = [predictions]
        
        for i, prediction in enumerate(predictions[0]):  # Iterate over the first (and only) batch
            if prediction == 1:  # Assuming label 1 is for errors
                errors.append(tokens[i])
        
        return errors

def extract_text_from_pdf(pdf_file_path):
    text = ""
    pdf_document = fitz.open(pdf_file_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
        
    text = text.replace('\n', ' ')
    
    return text

def main():
    model_path = "/Users/luryan/Documents/persona_project/Doc_IA/API/model_bert"
    pdf_file_path = "/Users/luryan/Documents/persona_project/Doc_IA/BERT_train/input/_Dossiês IA/Dossiê Aparicio.pdf"

    print("Inicializando o modelo de correção de texto...")
    text_correction_model = TextCorrectionModel(model_path)

    print("Extraindo texto do PDF...")
    documento_texto = extract_text_from_pdf(pdf_file_path)

    print("Identificando erros de português...")
    portuguese_errors = text_correction_model.identify_errors(documento_texto)
    print(documento_texto)
    
    print("Erros de Português Identificados:")
    print(portuguese_errors)

if __name__ == "__main__":
    main()