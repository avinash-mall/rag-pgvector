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
DATA_DIRECTORY = os.getenv('DATA_DIRECTORY')
ARAVEC_MODEL_PATH = os.getenv('ARAVEC_MODEL_PATH')

if not DATA_DIRECTORY or not ARAVEC_MODEL_PATH:
    raise ValueError("Please ensure DATA_DIRECTORY and ARAVEC_MODEL_PATH are set in the .env file")

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Load the SpaCy model for English
nlp_en = spacy.load("en_core_web_sm")


# Load AraVec pre-trained embeddings
def load_aravec_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified AraVec model path does not exist: {model_path}")
    model = gensim.models.Word2Vec.load(model_path)
    return model


aravec_model = load_aravec_model(ARAVEC_MODEL_PATH)

# Load English and Arabic stop words
stop_words_en = set(stopwords.words('english'))
stop_words_ar = set(stopwords.words('arabic'))


# Function to clean text
def clean_text(text, lang='en'):
    if lang == 'en':
        # Remove unwanted characters and punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        # Convert text to lowercase
        text = text.lower()

        # Remove stop words
        text = ' '.join(word for word in text.split() if word not in stop_words_en)

    elif lang == 'ar':
        # Remove diacritics
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

        # Normalize Arabic text
        text = re.sub(r'[^\w\s]', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())

        # Remove Arabic stop words
        text = ' '.join(word for word in text.split() if word not in stop_words_ar)

    return text


# Function to read documents from a directory
def read_directory(directory):
    all_docs = []
    reader = SimpleDirectoryReader(
        input_dir=directory,
        recursive=True,
        errors='ignore',
        required_exts=[".pdf", ".docx", ".txt"],
    )
    for docs in reader.iter_data():
        for doc in docs:
            all_docs.append(doc.text)  # Assuming doc has a 'text' attribute
    return all_docs


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
    doc = nlp_en(text)
    sents = list(doc.sents)
    vecs = np.stack([sent.vector / np.linalg.norm(sent.vector) for sent in sents if np.linalg.norm(sent.vector) != 0])
    return sents, vecs


# Function to process Arabic text and get sentence vectors
def process_arabic_text(text, aravec_model):
    sentences = sent_tokenize(text)
    sents = []
    vecs = []
    for sent in sentences:
        cleaned_sent = clean_text(sent, lang='ar')
        tokens = preprocess_arabic_text(cleaned_sent)
        token_vectors = vectorize_arabic_tokens(tokens, aravec_model)
        if token_vectors.size == 0:
            continue
        vec = np.mean(token_vectors, axis=0)
        norm_vec = np.linalg.norm(vec)
        if norm_vec != 0:
            sents.append(sent)
            vecs.append(vec / norm_vec)
    if len(vecs) > 0:
        vecs = np.array(vecs)
    return sents, vecs


# Function to cluster sentences based on vector similarity
def cluster_text(sents, vecs, threshold):
    clusters = [[0]]
    for i in range(1, len(sents)):
        if np.dot(vecs[i], vecs[i - 1]) < threshold:
            clusters.append([])
        clusters[-1].append(i)
    return clusters


# Function to process document and perform semantic chunking
def process_document(text, lang, threshold=0.3):
    if lang == 'en':
        sents, vecs = process_english_text(text)
    elif lang == 'ar':
        sents, vecs = process_arabic_text(text, aravec_model)
    else:
        return []

    if len(vecs) == 0:
        return []

    clusters = cluster_text(sents, vecs, threshold)
    clusters_lens = []
    final_texts = []

    for cluster in clusters:
        cluster_txt = clean_text(' '.join([sents[i].text if lang == 'en' else sents[i] for i in cluster]), lang=lang)
        cluster_len = len(cluster_txt)

        if cluster_len < 60:
            continue

        elif cluster_len > 3000:
            if lang == 'en':
                sents_div, vecs_div = process_english_text(cluster_txt)
            elif lang == 'ar':
                sents_div, vecs_div = process_arabic_text(cluster_txt, aravec_model)
            reclusters = cluster_text(sents_div, vecs_div, threshold=0.6)

            for subcluster in reclusters:
                div_txt = clean_text(
                    ' '.join([sents_div[i].text if lang == 'en' else sents_div[i] for i in subcluster]), lang=lang)
                div_len = len(div_txt)

                if div_len < 60 or div_len > 3000:
                    continue

                clusters_lens.append(div_len)
                final_texts.append(div_txt)
        else:
            clusters_lens.append(cluster_len)
            final_texts.append(cluster_txt)

    return final_texts


# Main function to read data from directory and perform chunking
def main(data_directory):
    # Read documents from directory
    documents = read_directory(data_directory)

    chunked_documents = []
    for doc in documents:
        detected_lang = detect(doc)
        chunks = process_document(doc, detected_lang)
        if chunks:
            chunked_documents.append(chunks)

    # Print chunked documents for verification
    for i, chunks in enumerate(chunked_documents):
        print(f"Document {i + 1}:")
        for j, chunk in enumerate(chunks):
            print(f" Chunk {j + 1}: {chunk}\n")


# Example usage:
if __name__ == "__main__":
    main(DATA_DIRECTORY)
