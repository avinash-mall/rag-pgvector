# Document Processing and Semantic Chunking

This repository contains a Python script for processing documents from a directory, performing text cleaning, extracting embeddings using NVIDIA context encoder models, and clustering sentences for semantic chunking.

These scripts represent a pipeline for processing documents, performing semantic chunking, and generating embeddings using a pretrained NVIDIA model. Here's a high-level overview and some key details for each script:

### document_processing.py

This script reads documents from a directory, cleans the text, and retrieves metadata.

- **Functions:**
  - `clean_text(text, language='en')`: Cleans the text based on the specified language (either English or Arabic).
  - `get_meta(file_path)`: Retrieves metadata for a file such as creation date, last modified date, and file name.
  - `read_documents_from_directory(directory)`: Reads documents from the specified directory using `SimpleDirectoryReader`, extracts text, and combines text from all pages of each file.

### chunking.py

This script processes text, performs chunking, and extracts keywords and named entities.

- **Functions:**
  - `process_text_with_nvidia_encoder(text, tokenizer, model, is_query=False)`: Encodes text using NVIDIA's context encoder.
  - `process_text(text, language, tokenizer, model)`: Cleans and tokenizes the text, then encodes it.
  - `cluster_sentences_kmeans(sentences, embeddings, num_keywords, min_clusters=2, max_clusters=10)`: Clusters sentences using K-Means based on the embeddings.
  - `merge_small_clusters(clusters, sentences, min_cluster_size=50)`: Merges small clusters into larger ones.
  - `extract_keywords(text, language='en', num_keywords=10)`: Extracts keywords from the text.
  - `extract_named_entities(text)`: Extracts named entities from the text.
  - `add_dynamic_metadata(text, language, chunk_keywords, document_keywords, document_metadata)`: Adds metadata to the text.
  - `process_document(document, language, tokenizer, model)`: Processes a document to extract chunks and metadata, and performs clustering.

### main.py

This is the main script that integrates the document processing and chunking functions, and generates embeddings for the processed chunks.

- **Main function:**
  - `main(data_directory)`: Reads documents from the specified directory, detects the language, processes the documents, constructs nodes from text chunks, and generates embeddings for each node.

### node_embedding.py

This script defines the construction of nodes from text chunks and the generation of embeddings.

- **Classes and Functions:**
  - `construct_nodes_from_chunks(text_chunks, documents, doc_idxs)`: Constructs `TextNode` objects from text chunks.
  - `generate_embeddings_for_nodes(nodes, embed_model)`: Generates embeddings for each node using the NVIDIA encoder.
  - `NvidiaEmbedModel`: A class that handles text embeddings using a pretrained NVIDIA model.

### Usage:

To run the entire pipeline, ensure that the `.env` file is properly set with the paths for data directory, NLTK data directory, and local model directories. Then execute the `main.py` script. Here's how you can execute it:

```bash
python main.py
```

### Potential Enhancements and Improvements:

1. **Error Handling**: Enhance error handling and logging throughout the scripts to provide more detailed information on failures.
2. **Unit Tests**: Implement unit tests to validate the functionality of each component.
3. **Configuration Management**: Use a more robust configuration management system instead of relying solely on `.env` files.
4. **Performance Optimization**: Optimize the text processing and clustering steps for large datasets.
5. **Documentation**: Add more detailed docstrings and comments to improve code readability and maintainability.

## Acknowledgments

-   The `transformers` library by Hugging Face for providing the pre-trained models and tokenizers.
-   The `nltk` library for text processing utilities.
-   The `scikit-learn` library for clustering algorithms.
-   NVIDIA for the context encoder models.
