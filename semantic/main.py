# main.py

import os
from dotenv import load_dotenv
from document_processing import read_documents_from_directory
from chunking import process_document
from node_embedding import construct_nodes_from_chunks, generate_embeddings_for_nodes, NvidiaEmbedModel
from transformers import AutoTokenizer, AutoModel
from langdetect import detect
import nltk
# Load environment variables from .env file
load_dotenv()

# Load variables from .env file and check if they are correctly loaded
DATA_DIR = os.getenv('DATA_DIRECTORY')
NLTK_DATA_DIR = os.getenv('NLTK_DATA_DIRECTORY')
LOCAL_MODEL_DIR = os.getenv('LOCAL_MODEL_DIR')
LOCAL_QUERY_MODEL_DIR = os.getenv('LOCAL_QUERY_MODEL_DIR')

if not DATA_DIR or not NLTK_DATA_DIR or not LOCAL_MODEL_DIR or not LOCAL_QUERY_MODEL_DIR:
    raise ValueError("Please ensure all required paths are set in the .env file")

# Set the NLTK data path to your local directory


nltk.data.path.append(NLTK_DATA_DIR)

# Load NVIDIA context encoder models from local storage
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
query_encoder = AutoModel.from_pretrained(LOCAL_QUERY_MODEL_DIR)
context_encoder = AutoModel.from_pretrained(LOCAL_MODEL_DIR)

# Initialize the embedding model
embed_model = NvidiaEmbedModel(context_encoder, tokenizer)


# Main function to read data from directory, perform chunking, and generate embeddings
def main(data_directory):
    # Read documents from directory
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
            print(f"Detected language: {detected_language}")
            chunks = process_document(document, detected_language, tokenizer, context_encoder)
            if chunks:
                chunked_documents.extend(chunks)
                doc_idxs.extend([doc_idx] * len(chunks))
        except Exception as e:
            print(f"Error processing document: {e}")

    # Construct nodes from text chunks
    nodes = construct_nodes_from_chunks(chunked_documents, documents, doc_idxs)

    # Generate embeddings for each node
    generate_embeddings_for_nodes(nodes, embed_model)

    # Print node embeddings for verification
    for node in nodes:
        print(f"Node text: {node.text}")
        print(f"Node metadata: {node.metadata}")
        print(f"Node embedding: {node.embedding}")


# Example usage:
if __name__ == "__main__":
    main(DATA_DIR)
