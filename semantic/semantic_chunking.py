import os
import re
import torch
from transformers import AutoTokenizer, AutoModel
from langdetect import detect, DetectorFactory
from llama_index.core import SimpleDirectoryReader
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv
from sklearn.cluster import KMeans

# Load environment variables from .env file
load_dotenv()

# Load variables from .env file and check if they are correctly loaded
DATA_DIR = os.getenv('DATA_DIRECTORY')
if not DATA_DIR:
    raise ValueError("Please ensure DATA_DIRECTORY is set in the .env file")

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Load NVIDIA context encoder models from local storage
local_dir = './local_nvidia_models/dragon-multiturn-context-encoder'
local_query_dir = './local_nvidia_models/dragon-multiturn-context-encoder_query_encoder'
tokenizer = AutoTokenizer.from_pretrained(local_dir)
query_encoder = AutoModel.from_pretrained(local_query_dir)
context_encoder = AutoModel.from_pretrained(local_dir)

# Load English and Arabic stop words
stopwords_english = set(stopwords.words('english'))
stopwords_arabic = set(stopwords.words('arabic'))

# Function to clean text
def clean_text(text, language='en'):
    if language == 'en':
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = ' '.join(word for word in text.split() if word not in stopwords_english)
    elif language == 'ar':
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Retain Arabic characters and spaces
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = text.strip()
        text = ' '.join(word for word in text.split() if word not in stopwords_arabic)
    return text

# Function to read documents from a directory
def read_documents_from_directory(directory):
    documents = []
    reader = SimpleDirectoryReader(
        input_dir=directory,
        recursive=True,
        errors='ignore',
        required_exts=[".pdf", ".docx", ".txt"],
    )
    for docs in reader.iter_data():
        for doc in docs:
            documents.append(doc.text)  # Assuming doc has a 'text' attribute
    return documents

# Function to process text and get sentence vectors using NVIDIA context encoder
def process_text_with_nvidia_encoder(text, is_query=False):
    if is_query:
        formatted_text = '\n'.join([turn['role'] + ": " + turn['content'] for turn in text]).strip()
        input_tensors = tokenizer(formatted_text, return_tensors='pt')
        embeddings = query_encoder(**input_tensors).last_hidden_state[:, 0, :]
    else:
        input_tensors = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        embeddings = context_encoder(**input_tensors).last_hidden_state[:, 0, :]
    return embeddings

# Function to process text (both English and Arabic) and get sentence vectors
def process_text(text, language):
    sentences = sent_tokenize(text)
    cleaned_sentences = [clean_text(sentence, language) for sentence in sentences if len(clean_text(sentence, language).split()) >= 3]
    embeddings = process_text_with_nvidia_encoder(cleaned_sentences)
    return cleaned_sentences, embeddings

# Function to cluster sentences using K-Means
def cluster_sentences_kmeans(sentences, embeddings, min_clusters=5, max_clusters=20):
    embeddings_np = embeddings.detach().numpy()
    n_clusters = min(max(min_clusters, len(sentences) // 5), max_clusters)
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeddings_np)
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)
    return clusters

# Function to merge small clusters
def merge_small_clusters(clusters, sentences, min_cluster_size=30):
    new_clusters = []
    current_cluster = []

    for cluster in clusters:
        cluster_text = ' '.join([sentences[i] for i in cluster if i < len(sentences)])
        if len(cluster_text) < min_cluster_size:
            current_cluster.extend(cluster)
        else:
            if current_cluster:
                new_clusters.append(current_cluster)
                current_cluster = []
            new_clusters.append(cluster)

    if current_cluster:
        new_clusters.append(current_cluster)

    return new_clusters

# Function to process document and perform semantic chunking
def process_document(text, language):
    print(f"Processing document in language: {language}")
    sentences, embeddings = process_text(text, language)

    if embeddings.size(0) == 0:
        print("No embeddings found, skipping document.")
        return []

    # Calculate dynamic number of clusters
    n_clusters = min(max(5, int(len(sentences) / 5)), 20)
    clusters = cluster_sentences_kmeans(sentences, embeddings, n_clusters)
    clusters = merge_small_clusters(clusters, sentences)

    chunked_texts = []

    for cluster in clusters:
        cluster_text = ' '.join([sentences[i] for i in cluster if i < len(sentences)])
        chunked_texts.append(cluster_text)

    return chunked_texts

# Main function to read data from directory and perform chunking
def main(data_directory):
    # Read documents from directory
    try:
        documents = read_documents_from_directory(data_directory)
        print(f"Documents read: {len(documents)}")
    except Exception as e:
        print(f"Error reading documents from directory: {e}")
        return

    chunked_documents = []
    for document in documents:
        try:
            detected_language = detect(document)
            print(f"Detected language: {detected_language}")
            chunks = process_document(document, detected_language)
            if chunks:
                chunked_documents.append(chunks)
        except Exception as e:
            print(f"Error processing document: {e}")

    # Print chunked documents
    for i, doc_chunks in enumerate(chunked_documents):
        print(f"Document {i + 1}:")
        for j, chunk in enumerate(doc_chunks):
            print(f" Chunk {j + 1}: {chunk}\n")

# Example usage:
if __name__ == "__main__":
    main(DATA_DIR)
