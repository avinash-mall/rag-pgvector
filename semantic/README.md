# Document Processing and Semantic Chunking

This repository contains a Python script for processing documents from a directory, performing text cleaning, extracting embeddings using NVIDIA context encoder models, and clustering sentences for semantic chunking.

## Features

- Load environment variables from a `.env` file.
- Read and process documents in various formats (PDF, DOCX, TXT) from a specified directory.
- Detect language of the document using `langdetect`.
- Clean and preprocess text based on the detected language (supports English and Arabic).
- Extract sentence embeddings using NVIDIA context encoder models.
- Cluster sentences using K-Means clustering.
- Merge small clusters to ensure meaningful chunks.
- Print chunked text for each document.

## Requirements

- Python 3.6 or higher
- Required Python packages (install using `pip`):
  - `os`
  - `re`
  - `torch`
  - `transformers`
  - `langdetect`
  - `llama_index`
  - `nltk`
  - `python-dotenv`
  - `scikit-learn`

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/avinash-mall/rag-pgvector.git
   cd agentic
```
2.  Create a virtual environment and activate it:
  
```	
	python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate 
```
    
3.  Install the required packages:
    
    ```
    pip install -r requirements.txt
    ``` 
    
4.  Create a `.env` file in the project root directory and add the following line:

    `DATA_DIRECTORY=/path/to/your/data_directory` 
    

## Usage

1.  Ensure the NVIDIA context encoder models are available in the specified local directories (`./local_nvidia_models/dragon-multiturn-context-encoder` and `./local_nvidia_models/dragon-multiturn-context-encoder_query_encoder`).
    
2.  Run the script:
        ```python script_name.py``` 
    
3.  The script will read documents from the specified directory, process each document, and print the chunked text for each document.
    

## Directory Structure
```document-processing-chunking/
├── .env
├── requirements.txt
├── semantic_chunking.py
└── local_nvidia_models/
    ├── dragon-multiturn-context-encoder/
    └── dragon-multiturn-context-encoder_query_encoder/
```

## Functions

-   `clean_text(text, language='en')`: Cleans and processes text based on the specified language.
-   `read_documents_from_directory(directory)`: Reads documents from a specified directory.
-   `process_text_with_nvidia_encoder(text, is_query=False)`: Processes text to get sentence vectors using NVIDIA context encoder.
-   `process_text(text, language)`: Tokenizes text into sentences and cleans each sentence.
-   `cluster_sentences_kmeans(sentences, embeddings, min_clusters=5, max_clusters=20)`: Clusters sentences using K-Means based on their embeddings.
-   `merge_small_clusters(clusters, sentences, min_cluster_size=30)`: Merges clusters that are smaller than a specified minimum size.
-   `process_document(text, language)`: Processes a document, detects its language, and clusters sentences into chunks.
-   `main(data_directory)`: Reads documents from the specified directory, processes each document, and prints the results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   The `transformers` library by Hugging Face for providing the pre-trained models and tokenizers.
-   The `nltk` library for text processing utilities.
-   The `scikit-learn` library for clustering algorithms.
-   NVIDIA for the context encoder models.
