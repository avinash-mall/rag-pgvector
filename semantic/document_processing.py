# document_processing.py

import os
from datetime import datetime
from llama_index.core import SimpleDirectoryReader
import re
from nltk.corpus import stopwords


# Function to clean text
def clean_text(text, language='en'):
    if language == 'en':
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    elif language == 'ar':
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Retain Arabic characters and spaces
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = text.strip()
        text = ' '.join(word for word in text.split() if word not in stopwords.words('arabic'))
    return text


def get_meta(file_path):
    # Retrieve file creation date, last modified date, and time
    creation_time = datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
    last_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
    # Get file name
    file_name = os.path.basename(file_path)
    # Processed date and time
    processed_time = datetime.now().isoformat()
    return {
        "creation_date": creation_time,
        "last_modified_date": last_modified_time,
        "processed_date": processed_time,
        "file_name": file_name
    }


def read_documents_from_directory(directory):
    documents = []
    combined_documents = {}

    # Initialize the reader with the specified parameters and metadata function
    reader = SimpleDirectoryReader(
        input_dir=directory,
        recursive=True,
        errors='ignore',
        required_exts=[".pdf", ".docx", ".txt"],
        file_metadata=get_meta
    )

    # Iterate over the data returned by the reader
    for docs in reader.iter_data():
        for doc in docs:
            file_path = doc.metadata['file_name']
            if file_path not in combined_documents:
                combined_documents[file_path] = {"text": doc.text, "metadata": doc.metadata}
            else:
                combined_documents[file_path]["text"] += "\n" + doc.text

    # Combine text from all pages of each file into a single document entry
    for combined in combined_documents.values():
        documents.append(combined)

    return documents
