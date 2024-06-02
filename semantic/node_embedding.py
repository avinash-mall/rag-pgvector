# node_embedding.py

from llama_index.core.schema import TextNode
from transformers import AutoTokenizer, AutoModel
import torch


# Function to construct nodes from text chunks
def construct_nodes_from_chunks(text_chunks, documents, doc_idxs):
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc["metadata"]
        nodes.append(node)
    return nodes


# Function to generate embeddings for each node using NVIDIA encoder
def generate_embeddings_for_nodes(nodes, embed_model):
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding.tolist()  # Convert to list


# Function to get text embeddings using NVIDIA encoder
class NvidiaEmbedModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_text_embedding(self, text):
        input_tensors = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            embeddings = self.model(**input_tensors).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return embeddings


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    # Load environment variables from .env file
    load_dotenv()

    # Load variables from .env file and check if they are correctly loaded
    LOCAL_MODEL_DIR = os.getenv('LOCAL_MODEL_DIR')
    LOCAL_QUERY_MODEL_DIR = os.getenv('LOCAL_QUERY_MODEL_DIR')

    if not LOCAL_MODEL_DIR or not LOCAL_QUERY_MODEL_DIR:
        raise ValueError("Please ensure all required paths are set in the .env file")

    # Load NVIDIA context encoder models from local storage
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    context_encoder = AutoModel.from_pretrained(LOCAL_MODEL_DIR)

    # Initialize the embedding model
    embed_model = NvidiaEmbedModel(context_encoder, tokenizer)

    # Placeholder data for text_chunks, documents, and doc_idxs
    text_chunks = ["This is a text chunk.", "Another text chunk."]
    documents = [{"metadata": {"title": "Doc 1"}}, {"metadata": {"title": "Doc 2"}}]
    doc_idxs = [0, 1]

    # Construct nodes from text chunks
    nodes = construct_nodes_from_chunks(text_chunks, documents, doc_idxs)

    # Generate embeddings for each node
    generate_embeddings_for_nodes(nodes, embed_model)

    # Print node embeddings for verification
    for node in nodes:
        print(f"Node text: {node.text}")
        print(f"Node metadata: {node.metadata}")
        print(f"Node embedding: {node.embedding}")
