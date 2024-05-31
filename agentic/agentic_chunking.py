import uuid
import requests
import os
import glob
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class AgenticChunker:
    def __init__(self):
        self.chunks = {}
        self.id_truncate_limit = 5
        self.generate_new_metadata_ind = True
        self.print_logging = True
        self.ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")
        self.prompt_path = os.getenv("PROMPT_PATH")

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

    def _load_prompt(self, filename):
        filepath = os.path.join(self.prompt_path, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()

    def add_propositions(self, propositions):
        for proposition in propositions:
            self.add_proposition(proposition)

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

    def add_proposition_to_chunk(self, chunk_id, proposition):
        self.chunks[chunk_id]['propositions'].append(proposition)
        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _update_chunk_summary(self, chunk):
        prompt = self._load_prompt("update_chunk_summary_prompt.txt").format(
            chunk_propositions='\n'.join(chunk['propositions']),
            current_summary=chunk['summary']
        )
        return self._call_ollama_api(prompt)

    def _update_chunk_title(self, chunk):
        prompt = self._load_prompt("update_chunk_title_prompt.txt").format(
            chunk_propositions='\n'.join(chunk['propositions']),
            current_summary=chunk['summary'],
            current_title=chunk['title']
        )
        return self._call_ollama_api(prompt)

    def _get_new_chunk_summary(self, proposition):
        prompt = self._load_prompt("get_new_chunk_summary_prompt.txt").format(
            proposition=proposition
        )
        return self._call_ollama_api(prompt)

    def _get_new_chunk_title(self, summary):
        prompt = self._load_prompt("get_new_chunk_title_prompt.txt").format(
            summary=summary
        )
        return self._call_ollama_api(prompt)

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

    def get_chunk_outline(self):
        chunk_outline = ""
        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}\n\n"""
            chunk_outline += single_chunk_string
        return chunk_outline

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

    def get_chunks(self, get_type='dict'):
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            return [" ".join(chunk['propositions']) for chunk in self.chunks.values()]

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

    def pretty_print_chunk_outline(self):
        print("Chunk Outline\n")
        print(self.get_chunk_outline())

def read_documents_from_folder():
    folder_path = os.getenv("FOLDER_PATH")
    documents = []
    for file_path in glob.glob(os.path.join(folder_path, '*')):
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    return documents

if __name__ == "__main__":
    documents = read_documents_from_folder()

    ac = AgenticChunker()

    for document in documents:
        ac.add_propositions(document.split('. '))  # Splitting by sentences for chunking

    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    print(ac.get_chunks(get_type='list_of_strings'))
