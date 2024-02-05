import argparse
import requests
import io
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
import torch
import faiss
import numpy as np
import logging

# Убираем ворнинги из вывода терминала
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# Скачивание PDF файла по url
def download_and_extract_text(url):
    response = requests.get(url)
    f = io.BytesIO(response.content)
    text = extract_text(f)
    return text

# Создание эмбеддингов
def create_embeddings(text, model_name='bert-base-uncased', device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Сохранение индексов в FAISS
def save_embeddings_to_faiss(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# Поиск похожего контекста
def find_similar_contexts(embedding, index, k=1):

    # Поиск KNN
    D, I = index.search(np.array([embedding]), k)
    return I[0]

# Загрузка модели
def load_qa_model(model_path, tokenizer_path):
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

# Поиск ответов по ближайшему контексту к вопросу
def find_answers(model, tokenizer, text, question, device='cpu'):
    inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

def main(pdf_url, question, embed_mode):
    # Выбор девайса GPU (CUDA) / CPU
    device = 'cuda' if embed_mode == 'gpu' and torch.cuda.is_available() else 'cpu'
    
    text = download_and_extract_text(pdf_url)
    
    # Разделение текста по параграфам
    paragraphs = text.split('\n\n')
    all_embeddings = []
    for paragraph in paragraphs:
        if paragraph.strip(): 
            embeddings = create_embeddings(paragraph, device=device)
            all_embeddings.append(embeddings[0]) 

    # Добавление эмбеддингов в FAISS
    combined_embeddings = np.vstack(all_embeddings)
    index = save_embeddings_to_faiss(combined_embeddings)
    
    # Создание эмбэддинга по вопросу
    question_embedding = create_embeddings(question, device=device)[0]
    
    # Поиск похожего контекста
    similar_context_ids = find_similar_contexts(question_embedding, index, k=1)
    if len(similar_context_ids) == 0:
        print("No context found to answer the question.")
        return
    
    similar_context = paragraphs[similar_context_ids[0]]
    
    # Загрузка модели (в данном случае был выбран BERT)
    model, tokenizer = load_qa_model(model_path='bert-large-uncased-whole-word-masking-finetuned-squad', tokenizer_path='bert-large-uncased-whole-word-masking-finetuned-squad')
    model.to(device)

    # Поиск ответа
    answer = find_answers(model, tokenizer, similar_context, question, device=device)
    
    print("Answer to the question:", answer)

if __name__ == '__main__':
    # Парсинг аргументов
    parser = argparse.ArgumentParser(description='Answer questions based on a PDF document.')
    parser.add_argument('--pdf_url', required=True, help='URL of the PDF file')
    parser.add_argument('--question', required=True, help='Question to be answered')
    parser.add_argument('--embed_mode', choices=['cpu', 'gpu'], default='cpu', help='Embedding computation mode')
    
    args = parser.parse_args()
    
    main(args.pdf_url, args.question, args.embed_mode)
