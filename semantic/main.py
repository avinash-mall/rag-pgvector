# main.py

import os
from dotenv import load_dotenv
from document_processing import read_documents_from_directory
from chunking import process_document, extract_keywords
from node_embedding import construct_nodes_from_chunks, generate_embeddings_for_nodes, NvidiaEmbedModel
from transformers import AutoTokenizer, AutoModel
from langdetect import detect

# Load environment variables from .env file
load_dotenv()

# Load variables from .env file and check if they are correctly loaded
DATA_DIR = os.getenv('DATA_DIRECTORY')
NLTK_DATA_DIR = os.getenv('NLTK_DATA_DIRECTORY')
LOCAL_MODEL_DIR = os.getenv('LOCAL_MODEL_DIR')
LOCAL_QUERY_MODEL_DIR = os.getenv('LOCAL_QUERY_MODEL_DIR')

if not DATA_DIR or not NLTK_DATA_DIR or not LOCAL_MODEL_DIR or not LOCAL_QUERY_MODEL_DIR:
    raise ValueError("Please ensure all required paths are set in the .env file")

import nltk
nltk.data.path.append(NLTK_DATA_DIR)

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
context_encoder = AutoModel.from_pretrained(LOCAL_MODEL_DIR)
embed_model = NvidiaEmbedModel(context_encoder, tokenizer)

def main(data_directory):
    try:
        documents = read_documents_from_directory(data_directory)
    except Exception as e:
        print(f"Error reading documents from directory: {e}")
        return

    chunked_documents = []
    doc_idxs = []
    for doc_idx, document in enumerate(documents):
        try:
            detected_language = detect(document["text"])
            document["metadata"]["language"] = detected_language
            document_keywords = extract_keywords(document["text"], detected_language)
            document["metadata"]["document_keywords"] = document_keywords
            chunks = process_document(document, detected_language, tokenizer, context_encoder)
            if chunks:
                chunked_documents.extend(chunks)
                doc_idxs.extend([doc_idx] * len(chunks))
        except Exception as e:
            print(f"Error processing document: {e}")

    nodes = construct_nodes_from_chunks(chunked_documents, documents, doc_idxs)
    nodes = generate_embeddings_for_nodes(nodes, embed_model)

    for node in nodes:
        print(f"INSERT INTO table_name (language, document_keywords, chunk_keywords, embedding, text, file_name, creation_date, last_modified_date, processed_date) VALUES ('{node.metadata['language']}', '{node.metadata['document_keywords']}', '{node.metadata['chunk_keywords']}', '{node.embedding}', '{node.text}', '{node.metadata['file_name']}', '{node.metadata['creation_date']}', '{node.metadata['last_modified_date']}', '{node.metadata['processed_date']}');")

if __name__ == "__main__":
    main(DATA_DIR)
