# RAG Implementation *(Open Source Only)* 
**(Work in Progress)**
RAG implementation with Arabic and English language support 

## Step 1: *(Completed)* Chunking Implementation for Arabic and English
### Semantic Chunking:
Adjacent Sequence Clustering approach for chunking.
This script processes English and Arabic texts, clustering sentences based on their semantic similarity using vector representations. It begins by loading environment variables and ensuring the required paths are set. The script employs various libraries to handle language detection, tokenization, stopword removal, text normalization, and vectorization using pre-trained models like SpaCy for English and AraVec for Arabic. It reads documents from a specified directory, preprocesses and cleans the text, and then vectorizes sentences. The vectorized sentences are clustered based on cosine similarity, and large clusters are recursively divided. Finally, it processes the documents and prints out the chunked text for verification.
### Agentic Chunking *(Preferred but Computationally Expensive)*:
The script processes and categorizes text documents by leveraging an external API (ollama-phi3) for summarization and chunking. It reads configuration values and prompts from environment variables and text files, respectively. Documents are loaded from a specified folder and split into individual propositions, which are then added to relevant chunks or used to create new ones. The script interfaces with the API to generate and update chunk summaries and titles dynamically. Detailed and summary views of the chunks are printed, providing a clear overview of the categorized content. This systematic approach ensures accurate and up-to-date chunk metadata, facilitating effective text processing and categorization.

## Step 2: *(Not Implemented)* Create Nodes and Embedding from nodes and store vectors to pgvector for Arabic and English

## Step 3: *(Not Implemented)* Build Retrieval Pipeline from local LLM for Arabic and English
