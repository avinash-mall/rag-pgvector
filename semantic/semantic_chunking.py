import os
import re
import numpy as np
import spacy
from sklearn.cluster import KMeans
from langdetect import detect, DetectorFactory
from llama_index.core import SimpleDirectoryReader
import gensim
import arabic_reshaper
from bidi.algorithm import get_display
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv
import string

# Load environment variables from .env file
load_dotenv()

# Load variables from .env file and check if they are correctly loaded
DATA_DIR = os.getenv('DATA_DIRECTORY')
ARAVEC_MODEL_DIR = os.getenv('ARAVEC_MODEL_PATH')

if not DATA_DIR or not ARAVEC_MODEL_DIR:
    raise ValueError("Please ensure DATA_DIRECTORY and ARAVEC_MODEL_PATH are set in the .env file")

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Load the SpaCy model for English
nlp_english = spacy.load("en_core_web_sm")

# Load AraVec pre-trained embeddings
def load_aravec_embeddings(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified AraVec model path does not exist: {model_path}")
    model = gensim.models.Word2Vec.load(model_path)
    return model

aravec_model = load_aravec_embeddings(ARAVEC_MODEL_DIR)

# Load English and Arabic stop words
stopwords_english = set(stopwords.words('english'))
stopwords_arabic = set(stopwords.words('arabic'))

# Function to clean text
def clean_text(text, language='en'):
    if language == 'en':
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        text = text.lower()
        text = ' '.join(word for word in text.split() if word not in stopwords_english)
    elif language == 'ar':
        arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
        text = re.sub(arabic_diacritics, '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
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

# Function to preprocess Arabic text
def preprocess_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return word_tokenize(bidi_text)

# Function to vectorize Arabic tokens using AraVec
def vectorize_arabic_tokens(tokens, model):
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
        else:
            vectors.append(np.zeros(model.vector_size, dtype=np.float64))
    return np.array(vectors, dtype=np.float64)

# Function to process English text and get sentence vectors
def process_english_text(text):
    document = nlp_english(text)
    sentences = list(document.sents)
    vectors = np.stack([sentence.vector / np.linalg.norm(sentence.vector) for sentence in sentences if np.linalg.norm(sentence.vector) != 0])
    return sentences, vectors

# Function to process Arabic text and get sentence vectors
def process_arabic_text(text, model):
    sentences = sent_tokenize(text)
    processed_sentences = []
    vectors = []
    for sentence in sentences:
        cleaned_sentence = clean_text(sentence, language='ar')
        tokens = preprocess_arabic_text(cleaned_sentence)
        token_vectors = vectorize_arabic_tokens(tokens, model)
        if token_vectors.size == 0:
            continue
        sentence_vector = np.mean(token_vectors, axis=0)
        norm_vector = np.linalg.norm(sentence_vector)
        if norm_vector != 0:
            processed_sentences.append(sentence)
            vectors.append(sentence_vector / norm_vector)
    if len(vectors) > 0:
        vectors = np.array(vectors)
    return processed_sentences, vectors

# Function to cluster sentences based on vector similarity
def cluster_sentences(sentences, vectors, similarity_threshold):
    clusters = [[0]]
    for i in range(1, len(sentences)):
        if np.dot(vectors[i], vectors[i - 1]) < similarity_threshold:
            clusters.append([])
        clusters[-1].append(i)
    return clusters

# Function to process document and perform semantic chunking
def process_document(text, language, similarity_threshold=0.3):
    if language == 'en':
        sentences, vectors = process_english_text(text)
    elif language == 'ar':
        sentences, vectors = process_arabic_text(text, aravec_model)
    else:
        return []

    if len(vectors) == 0:
        return []

    clusters = cluster_sentences(sentences, vectors, similarity_threshold)
    clusters_lengths = []
    chunked_texts = []

    for cluster in clusters:
        cluster_text = clean_text(' '.join([sentences[i].text if language == 'en' else sentences[i] for i in cluster]), language=language)
        cluster_length = len(cluster_text)

        if cluster_length < 60:
            continue

        elif cluster_length > 3000:
            if language == 'en':
                sub_sentences, sub_vectors = process_english_text(cluster_text)
            elif language == 'ar':
                sub_sentences, sub_vectors = process_arabic_text(cluster_text, aravec_model)
            sub_clusters = cluster_sentences(sub_sentences, sub_vectors, similarity_threshold=0.6)

            for sub_cluster in sub_clusters:
                sub_cluster_text = clean_text(
                    ' '.join([sub_sentences[i].text if language == 'en' else sub_sentences[i] for i in sub_cluster]), language=language)
                sub_cluster_length = len(sub_cluster_text)

                if sub_cluster_length < 60 or sub_cluster_length > 3000:
                    continue

                clusters_lengths.append(sub_cluster_length)
                chunked_texts.append(sub_cluster_text)
        else:
            clusters_lengths.append(cluster_length)
            chunked_texts.append(cluster_text)

    return chunked_texts

# Main function to read data from directory and perform chunking
def main(data_directory):
    # Read documents from directory
    documents = read_documents_from_directory(data_directory)

    chunked_documents = []
    for document in documents:
        detected_language = detect(document)
        chunks = process_document(document, detected_language)
        if chunks:
            chunked_documents.append(chunks)

    # Print chunked documents for verification
    for i, chunks in enumerate(chunked_documents):
        print(f"Document {i + 1}:")
        for j, chunk in enumerate(chunks):
            print(f" Chunk {j + 1}: {chunk}\n")

# Example usage:
if __name__ == "__main__":
    main(DATA_DIR)