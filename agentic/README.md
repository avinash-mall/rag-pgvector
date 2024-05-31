# Agentic Chunking
## Imports and Environment Loading


```
import uuid
import requests
import os
import glob
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
```
### Imports: 
- The script imports necessary libraries such as uuid for generating unique IDs, requests for making HTTP requests, os and glob for file handling, pydantic for data validation, and dotenv for loading environment variables from a .env file.
### Environment Loading: 
- The load_dotenv() function loads environment variables from a .env file into the script's environment.

## AgenticChunker Class
### Initialization
```
class AgenticChunker:
    def __init__(self):
        self.chunks = {}
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = True
        self.print_logging = True
        self.ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")
        self.prompt_path = os.getenv("PROMPT_PATH")
```
### Instance Variables:
- self.chunks: Dictionary to store chunks of propositions.
- self.id_truncate_limit: Limit for truncating UUIDs.
- self.generate_new_metadata_ind: Boolean flag to indicate if metadata should be updated.
- self.print_logging: Boolean flag for printing logs.
- self.ollama_endpoint: URL for the Ollama API, loaded from the .env file.
- self.prompt_path: Path to the directory containing prompt text files, loaded from the .env file.
### Helper Methods
```
def _call_ollama_api(self, prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "phi3",
        "prompt": prompt,
        "temperature": 0,
        "stream": False
    }
    response = requests.post(self.ollama_endpoint, headers=headers, json=data)
    if response.status_code != 200:
        raise ValueError(f"Error from Ollama API: {response.status_code} {response.text}")

    try:
        response_json = response.json()
    except requests.exceptions.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON response from Ollama API: {e.msg}\nResponse text: {response.text}")

    return response_json['response'].strip()
```
- _call_ollama_api: This method sends a POST request to the Ollama API with the provided prompt and returns the response text. It handles errors for unsuccessful responses and JSON decoding issues.

```
def _load_prompt(self, filename):
    filepath = os.path.join(self.prompt_path, filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()
```
- _load_prompt: This method reads a prompt text file from the specified filename and returns its content.

### Proposition Methods
```
def add_propositions(self, propositions):
    for proposition in propositions:
        self.add_proposition(proposition)
```
- add_propositions: This method takes a list of propositions and adds each one to the chunker using the add_proposition method.
```
def add_proposition(self, proposition):
    if self.print_logging:
        print(f"\nAdding: '{proposition}'")

    if not self.chunks:
        if self.print_logging:
            print("No chunks, creating a new one")
        self._create_new_chunk(proposition)
        return

    chunk_id = self._find_relevant_chunk(proposition)

    if chunk_id in self.chunks:
        if self.print_logging:
            print(f"Chunk Found ({chunk_id}), adding to: {self.chunks[chunk_id]['title']}")
        self.add_proposition_to_chunk(chunk_id, proposition)
    else:
        if self.print_logging:
            print("No chunks found")
        self._create_new_chunk(proposition)
```

- add_proposition: This method adds a single proposition to the most relevant chunk or creates a new chunk if no relevant chunk is found. It logs the process if logging is enabled.

```
def add_proposition_to_chunk(self, chunk_id, proposition):
    self.chunks[chunk_id]['propositions'].append(proposition)
    if self.generate_new_metadata_ind:
        self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
        self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])
```

- add_proposition_to_chunk: This method adds a proposition to an existing chunk and updates the chunk's summary and title if generate_new_metadata_ind is set to True.

### Chunk Metadata Methods

```
def _update_chunk_summary(self, chunk):
    prompt = self._load_prompt("update_chunk_summary_prompt.txt").format(
        chunk_propositions='\n'.join(chunk['propositions']),
        current_summary=chunk['summary']
    )
    return self._call_ollama_api(prompt)
```

- _update_chunk_summary: This method updates the summary of a chunk by formatting the appropriate prompt and sending it to the Ollama API.

```
def _update_chunk_title(self, chunk):
    prompt = self._load_prompt("update_chunk_title_prompt.txt").format(
        chunk_propositions='\n'.join(chunk['propositions']),
        current_summary=chunk['summary'],
        current_title=chunk['title']
    )
    return self._call_ollama_api(prompt)
```
- _update_chunk_title: This method updates the title of a chunk by formatting the appropriate prompt and sending it to the Ollama API.
```
def _get_new_chunk_summary(self, proposition):
    prompt = self._load_prompt("get_new_chunk_summary_prompt.txt").format(
        proposition=proposition
    )
    return self._call_ollama_api(prompt)
```
- _get_new_chunk_summary: This method generates a summary for a new chunk by formatting the appropriate prompt and sending it to the Ollama API.
```
def _get_new_chunk_title(self, summary):
    prompt = self._load_prompt("get_new_chunk_title_prompt.txt").format(
        summary=summary
    )
    return self._call_ollama_api(prompt)
```
- _get_new_chunk_title: This method generates a title for a new chunk by formatting the appropriate prompt and sending it to the Ollama API.

### Chunk Creation and Retrieval Methods
```
def _create_new_chunk(self, proposition):
    new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
    new_chunk_summary = self._get_new_chunk_summary(proposition)
    new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

    self.chunks[new_chunk_id] = {
        'chunk_id': new_chunk_id,
        'propositions': [proposition],
        'title': new_chunk_title,
        'summary': new_chunk_summary,
        'chunk_index': len(self.chunks)
    }
    if self.print_logging:
        print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")
```
- _create_new_chunk: This method creates a new chunk with a unique ID, a generated summary, and a title. It then adds the proposition to this new chunk.
```
def get_chunk_outline(self):
    chunk_outline = ""
    for chunk_id, chunk in self.chunks.items():
        single_chunk_string = f"""Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}\n\n"""
        chunk_outline += single_chunk_string
    return chunk_outline
```
- get_chunk_outline: This method generates and returns an outline of all current chunks.
```
def _find_relevant_chunk(self, proposition):
    current_chunk_outline = self.get_chunk_outline()

    prompt = self._load_prompt("find_relevant_chunk_prompt.txt").format(
        current_chunk_outline=current_chunk_outline,
        proposition=proposition
    )
    chunk_found = self._call_ollama_api(prompt)

    if "No chunks" in chunk_found:
        return None
    if "Chunk ID:" in chunk_found:
        chunk_id = chunk_found.split("Chunk ID:")[1].strip().split()[0]
        return chunk_id
    return None
```
- _find_relevant_chunk: This method finds the most relevant chunk for a given proposition by formatting the appropriate prompt and sending it to the Ollama API. It parses the response to extract the chunk ID.

### Output Methods
```
def get_chunks(self, get_type='dict'):
    if get_type == 'dict':
        return self.chunks
    if get_type == 'list_of_strings':
        return [" ".join(chunk['propositions']) for chunk in self.chunks.values()]
```
- get_chunks: This method returns the chunks either as a dictionary or as a list of strings, depending on the specified type.
```
def pretty_print_chunks(self):
    print(f"\nYou have {len(self.chunks)} chunks\n")
    for chunk_id, chunk in self.chunks.items():
        print(f"Chunk #{chunk['chunk_index']}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Summary: {chunk['summary']}")
        print(f"Propositions:")
        for prop in chunk['propositions']:
            print(f"    -{prop}")
        print("\n\n")
```
- pretty_print_chunks: This method prints a detailed overview of all chunks, including their IDs, summaries, and propositions.
```
def pretty_print_chunk_outline(self):
    print("Chunk Outline\n")
    print(self.get_chunk_outline())
```
- pretty_print_chunk_outline: This method prints the outline of all chunks.

### Document Reading Function
```
def read_documents_from_folder():
    folder_path = os.getenv("FOLDER_PATH")
    documents = []
    for file_path in glob.glob(os.path.join(folder_path, '*')):
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    return documents
```
- read_documents_from_folder: This function reads all text files from the specified folder and returns their contents as a list of strings.

### Main Script
```
if __name__ == "__main__":
    documents = read_documents_from_folder()

    ac = AgenticChunker()

    for document in documents:
        ac.add_propositions(document.split('. '))  # Splitting by sentences for chunking

    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    print(ac.get_chunks(get_type='list_of_strings'))
```
### Main Execution:
- Reads documents from the folder specified in the .env file.
- Initializes an AgenticChunker instance.
- Adds propositions from the documents to the chunker (splitting by sentences).
- Prints the detailed overview and outline of the chunks.
- Prints the chunks as a list of strings.

This script provides a comprehensive approach to processing and chunking text documents, leveraging an external API for summarization and categorization, and ensuring flexibility through environment configurations.
