# Semantic Text Chunking

This repository contains a script for semantic chunking of English and Arabic text documents. It processes documents to create meaningful chunks of text based on sentence similarity.

## Features

- **Environment Variable Configuration:** Loads paths for data and models from a `.env` file.
- **Multi-language Support:** Handles both English and Arabic text.
- **Text Cleaning:** Removes unwanted characters and stopwords.
- **Sentence Vectorization:** Uses SpaCy for English and AraVec for Arabic.
- **Sentence Clustering:** Clusters sentences based on vector similarity.
- **Semantic Chunking:** Creates chunks of text that are semantically coherent and within length constraints.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/avinash-mall/rag-pgvector.git
    cd semantic-text-chunking
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file with the following variables:
    ```env
    DATA_DIRECTORY=path/to/your/data
    ARAVEC_MODEL_PATH=path/to/your/aravec/model
    ```

## Usage

1. Place your text documents in the directory specified by `DATA_DIRECTORY` in your `.env` file.
2. Run the script:
    ```sh
    python chunking_script.py
    ```

### Explanation
This script is designed to process and semantically chunk English and Arabic text documents. It uses a variety of natural language processing (NLP) techniques to clean, tokenize, vectorize, and cluster sentences based on vector similarity. The main steps are:

1. Load Environment Variables: Reads configuration from a .env file, ensuring required paths for data and models are set.
2. Load Models and Resources: Loads the SpaCy model for English, AraVec embeddings for Arabic, and stopwords for both languages.
3. Text Cleaning: Provides functions to clean text by removing unwanted characters and stopwords, tailored for English and Arabic.
4. Document Reading: Reads documents from a specified directory.
5. Text Preprocessing: Tokenizes Arabic text and processes it using AraVec embeddings.
6. Sentence Vectorization: Vectorizes sentences using SpaCy for English and AraVec for Arabic.
7. Clustering Sentences: Clusters sentences based on their vector similarity to chunk the text into meaningful sections.
8. Semantic Chunking: Processes documents to create chunks of text that are semantically coherent and within specified length constraints.
9. Main Function: Reads documents from a directory, processes them to detect language and create chunks, and prints the results for verification.


## Code Overview

### Imports and Setup


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

**Explanation:**

-   Imports necessary libraries for file handling (`os`), regular expressions (`re`), numerical operations (`numpy`), natural language processing (`spacy`, `nltk`, `gensim`), language detection (`langdetect`), Arabic text reshaping (`arabic_reshaper`), bidirectional text support (`bidi`), environment variable loading (`dotenv`), and directory reading (`llama_index`).

### Load Environment Variables

    # Load environment variables from .env file
    load_dotenv()

    # Load variables from .env file and check if they are correctly loaded

    DATA_DIR = os.getenv('DATA_DIRECTORY')
    ARAVEC_MODEL_DIR = os.getenv('ARAVEC_MODEL_PATH')
    
    if not DATA_DIR or not ARAVEC_MODEL_DIR:
        raise ValueError("Please ensure DATA_DIRECTORY and ARAVEC_MODEL_PATH are set in the .env file")` 

**Explanation:**

-   Loads environment variables from a `.env` file to get the data directory and the AraVec model path. If these are not set, the script raises an error.

### Ensure Consistent Results from Language Detection

    # Ensure consistent results from langdetect
    DetectorFactory.seed = 0 

**Explanation:**

-   Seeds the `langdetect` library to ensure consistent language detection results.

### Load SpaCy Model for English

    # Load the SpaCy model for English
    
        nlp_english = spacy.load("en_core_web_sm")

**Explanation:**

-   Loads the SpaCy model for English NLP tasks.

### Load AraVec Pre-trained Embeddings

    # Load AraVec pre-trained embeddings
    
        def load_aravec_embeddings(model_path):
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"The specified AraVec model path does not exist: {model_path}")
            model = gensim.models.Word2Vec.load(model_path)
            return model
        
        aravec_model = load_aravec_embeddings(ARAVEC_MODEL_DIR)

**Explanation:**

-   Defines a function to load AraVec pre-trained embeddings for Arabic. Checks if the model path exists and loads the model using Gensim. If the path does not exist, it raises a `FileNotFoundError`.

### Load Stop Words


    # Load English and Arabic stop words
    stopwords_english = set(stopwords.words('english'))
    stopwords_arabic = set(stopwords.words('arabic')) 

**Explanation:**

-   Loads stop words for English and Arabic using NLTK.

### Clean Text Function


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

**Explanation:**

-   Defines a function to clean text for both English and Arabic. Removes unwanted characters, punctuation, diacritics (for Arabic), and stop words.

### Read Documents from Directory


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

**Explanation:**

-   Defines a function to read documents from a directory using `SimpleDirectoryReader`. Reads files with extensions `.pdf`, `.docx`, and `.txt`.

### Preprocess Arabic Text


    # Function to preprocess Arabic text
    def preprocess_arabic_text(text):
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        return word_tokenize(bidi_text) 

**Explanation:**

-   Defines a function to preprocess Arabic text by reshaping it for proper display and tokenizing it.

### Vectorize Arabic Tokens Using AraVec

    # Function to vectorize Arabic tokens using AraVec
    def vectorize_arabic_tokens(tokens, model):
        vectors = []
        for token in tokens:
            if token in model.wv:
                vectors.append(model.wv[token])
            else:
                vectors.append(np.zeros(model.vector_size, dtype=np.float64))
        return np.array(vectors, dtype=np.float64) 

**Explanation:**

-   Defines a function to vectorize Arabic tokens using the AraVec model. If a token is not in the model, it assigns a zero vector.

### Process English Text and Get Sentence Vectors

    # Function to process English text and get sentence vectors
    def process_english_text(text):
        document = nlp_english(text)
        sentences = list(document.sents)
        vectors = np.stack([sentence.vector / np.linalg.norm(sentence.vector) for sentence in sentences if np.linalg.norm(sentence.vector) != 0])
        return sentences, vectors 

**Explanation:**

-   Defines a function to process English text using SpaCy, obtaining sentence vectors normalized to unit length.

### Process Arabic Text and Get Sentence Vectors

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

**Explanation:**

-   Defines a function to process Arabic text. Tokenizes, vectorizes, and normalizes sentences, excluding empty vectors.

### Cluster Sentences Based on Vector Similarity

    # Function to cluster sentences based on vector similarity
    def cluster_sentences(sentences, vectors, similarity_threshold):
        clusters = [[0]]
        for i in range(1, len(sentences)):
            if np.dot(vectors[i], vectors[i - 1]) < similarity_threshold:
                clusters.append([])
            clusters[-1].append(i)
        return clusters 

**Explanation:**

-   Defines a function to cluster sentences based on vector similarity. Creates a new cluster if the similarity between consecutive sentence vectors falls below a threshold.

### Process Document and Perform Semantic Chunking

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

**Explanation:**

-   Defines a function to process a document, cluster sentences, and create chunks of text based on vector similarity. Ensures chunks are within specified length constraints and handles very long chunks by further subdividing them.

### Main Function to Read Data and Perform Chunking

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

### Example usage:

    if __name__ == "__main__":
        main(DATA_DIR)` 

**Explanation:**

-   Defines the main function to read documents from a specified directory, detect their language, process them to create chunks, and print the chunks for verification. It calls the `main` function if the script is run as the main module.
