# rag-pgvector
(Work in Progress)
RAG implementation with arabic and english language support 

### Step 1: (Completed) Chunking Implimentation for Arabic and English
Adjacent Sequence Clustering approach for chunking.
This method generates a more varied distribution, indicative of its context-sensitive approach. By clustering based on semantic similarity, it ensures that the content within each chunk is coherent while allowing for flexibility in chunk size. This method may be advantageous when it is important to preserve the semantic continuity of text data.
The fundamental premise of this approach is that two sentences that appear consecutively in a text are more likely to be semantically related than two sentences that are farther apart.
### Step 2: (Not Implimented) Create Nodes and Embedding from nodes and store vectors to pgvector for Arabic and English

### Step 3: (Not Implimented) Build Retrieval Pipeline from local LLM for Arabic and English
