# chunking.py

from collections import Counter
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tree import Tree
from sklearn.cluster import KMeans
from langdetect import DetectorFactory
from document_processing import clean_text

DetectorFactory.seed = 0

def process_text_with_nvidia_encoder(text, tokenizer, model, is_query=False):
    if is_query:
        formatted_text = '\n'.join([turn['role'] + ": " + turn['content'] for turn in text]).strip()
        input_tensors = tokenizer(formatted_text, return_tensors='pt')
        embeddings = model(**input_tensors).last_hidden_state[:, 0, :]
    else:
        input_tensors = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        embeddings = model(**input_tensors).last_hidden_state[:, 0, :]
    return embeddings

def process_text(text, language, tokenizer, model):
    sentences = sent_tokenize(text)
    cleaned_sentences = [clean_text(sentence, language) for sentence in sentences if
                         len(clean_text(sentence, language).split()) >= 3]
    embeddings = process_text_with_nvidia_encoder(cleaned_sentences, tokenizer, model)
    return cleaned_sentences, embeddings

def cluster_sentences_kmeans(sentences, embeddings, num_keywords, min_clusters=2, max_clusters=10):
    embeddings_np = embeddings.detach().numpy()
    n_clusters = min(max(min_clusters, num_keywords // 2), max_clusters)
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(embeddings_np)
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)
    return clusters

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

def extract_named_entities(text):
    chunks = ne_chunk(pos_tag(word_tokenize(text)))
    entities = []
    for chunk in chunks:
        if isinstance(chunk, Tree):
            entity = ' '.join(c[0] for c in chunk.leaves())
            entities.append(entity)
    return entities

def process_document(document, language, tokenizer, model):
    text = document["text"]
    document_metadata = document["metadata"]
    sentences, embeddings = process_text(text, language, tokenizer, model)

    if embeddings.size(0) == 0:
        return []

    document_keywords = extract_keywords(text, language)
    document_metadata['document_keywords'] = document_keywords
    clusters = cluster_sentences_kmeans(sentences, embeddings, len(document_keywords))
    clusters = merge_small_clusters(clusters, sentences)

    chunked_texts = []

    for cluster in clusters:
        cluster_text = ' '.join([sentences[i] for i in cluster if i < len(sentences)])
        chunk_keywords = extract_keywords(cluster_text, language)
        document_metadata['chunk_keywords'] = chunk_keywords  # Add chunk keywords to metadata
        chunked_texts.append(cluster_text)

    return chunked_texts
