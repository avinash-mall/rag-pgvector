# chunking.py

import torch
from transformers import AutoTokenizer, AutoModel
from langdetect import DetectorFactory
from sklearn.cluster import KMeans
from collections import Counter
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tree import Tree
from document_processing import clean_text
from nltk.corpus import stopwords  # Import stopwords

# Ensure consistent results from langdetect
DetectorFactory.seed = 0


# Function to process text and get sentence vectors using NVIDIA context encoder
def process_text_with_nvidia_encoder(text, tokenizer, model, is_query=False):
    if is_query:
        formatted_text = '\n'.join([turn['role'] + ": " + turn['content'] for turn in text]).strip()
        input_tensors = tokenizer(formatted_text, return_tensors='pt')
        embeddings = model(**input_tensors).last_hidden_state[:, 0, :]
    else:
        input_tensors = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        embeddings = model(**input_tensors).last_hidden_state[:, 0, :]
    return embeddings


# Function to process text (both English and Arabic) and get sentence vectors
def process_text(text, language, tokenizer, model):
    sentences = sent_tokenize(text)
    cleaned_sentences = [clean_text(sentence, language) for sentence in sentences if
                         len(clean_text(sentence, language).split()) >= 3]
    embeddings = process_text_with_nvidia_encoder(cleaned_sentences, tokenizer, model)
    return cleaned_sentences, embeddings


# Function to cluster sentences using K-Means
def cluster_sentences_kmeans(sentences, embeddings, num_keywords, min_clusters=2, max_clusters=10):
    embeddings_np = embeddings.detach().numpy()
    # Calculate the number of clusters dynamically based on the number of keywords
    n_clusters = min(max(min_clusters, num_keywords // 2), max_clusters)
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeddings_np)
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)
    return clusters


# Function to merge small clusters
def merge_small_clusters(clusters, sentences, min_cluster_size=50):
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


# Function to extract keywords from text
def extract_keywords(text, language='en', num_keywords=10):
    words = word_tokenize(text)
    if language == 'en':
        words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]
    elif language == 'ar':
        words = [word for word in words if word.isalpha() and word not in stopwords.words('arabic')]
    word_freq = Counter(words)
    common_words = word_freq.most_common(num_keywords)
    keywords = [word for word, _ in common_words]
    return keywords


# Function to extract named entities from text
def extract_named_entities(text):
    chunks = ne_chunk(pos_tag(word_tokenize(text)))
    entities = []
    for chunk in chunks:
        if isinstance(chunk, Tree):
            entity = ' '.join(c[0] for c in chunk.leaves())
            entities.append(entity)
    return entities


# Function to add dynamic metadata to text
def add_dynamic_metadata(text, language, chunk_keywords, document_keywords, document_metadata):
    entities = extract_named_entities(text)
    metadata = (
        f"Language: {language}\n"
        f"Document Keywords: {', '.join(document_keywords)}\n"
        f"Chunk Keywords: {', '.join(chunk_keywords)}\n"
        f"Named Entities: {', '.join(entities)}\n"
        f"Creation Date: {document_metadata['creation_date']}\n"
        f"Last Modified Date: {document_metadata['last_modified_date']}\n"
        f"Processed Date: {document_metadata['processed_date']}\n"
        f"File Name: {document_metadata['file_name']}\n\n"
    )
    return metadata + text


# Function to process document and perform semantic chunking
def process_document(document, language, tokenizer, model):
    text = document["text"]
    document_metadata = document["metadata"]
    print(f"Processing document in language: {language}")
    sentences, embeddings = process_text(text, language, tokenizer, model)

    if embeddings.size(0) == 0:
        print("No embeddings found, skipping document.")
        return []

    document_keywords = extract_keywords(text, language)
    # Calculate dynamic number of clusters based on the number of keywords
    clusters = cluster_sentences_kmeans(sentences, embeddings, len(document_keywords))
    clusters = merge_small_clusters(clusters, sentences)

    chunked_texts = []

    for cluster in clusters:
        cluster_text = ' '.join([sentences[i] for i in cluster if i < len(sentences)])
        chunk_keywords = extract_keywords(cluster_text, language)
        chunked_texts.append(
            add_dynamic_metadata(cluster_text, language, chunk_keywords, document_keywords, document_metadata))

    return chunked_texts
