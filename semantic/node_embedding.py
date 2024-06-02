# node_embedding.py

from transformers import AutoTokenizer, AutoModel
import torch
from chunking import process_document
from langdetect import detect
from llama_index.core.schema import TextNode

class NvidiaEmbedModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_text_embedding(self, text):
        input_tensors = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            embeddings = self.model(**input_tensors).last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return embeddings

def construct_nodes_from_chunks(text_chunks, documents, doc_idxs):
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc["metadata"]
        # Ensure metadata contains necessary fields
        node.metadata['language'] = src_doc["metadata"].get('language', 'unknown')
        node.metadata['document_keywords'] = src_doc["metadata"].get('document_keywords', [])
        node.metadata['chunk_keywords'] = src_doc["metadata"].get('chunk_keywords', [])
        nodes.append(node)
    return nodes

def generate_embeddings_for_nodes(nodes, embed_model):
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(node.text)
        node.embedding = node_embedding.tolist()
    return nodes
