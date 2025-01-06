# question_extract

# Question Extraction and Processing System

This Python script provides a question extraction and processing system using FAISS for similarity search, Ollama for question rewriting, and modern NLP tools like spaCy and Sentence Transformers. It allows you to extract questions from a given text document, find similar questions, and generate variations of the questions in different styles.

## Features

- Extract questions from a text document using spaCy's linguistic features
- Compute embeddings for questions using Sentence Transformers
- Find similar questions using FAISS similarity search
- Rewrite questions in different styles (formal, casual) using Ollama
- Store and manage questions in a JSON database

## Requirements

- Python 3.6+
- spaCy
- Sentence Transformers
- FAISS
- Langchain Community
- Ollama

Install the required dependencies using:

```
pip install -r requirements.txt
```

## Usage

1. Initialize the `InMemoryQuestionProcessor` with desired configurations:

```python
processor = InMemoryQuestionProcessor(
    db_path="questions.json",
    ollama_model="llama2", 
    ollama_host="http://localhost:11434"
)
```

2. Process a text document to extract and analyze questions:

```python
sample_text = '''
This is a sample document. What is the meaning of life? 
How do we process text effectively? The system should handle
various types of questions. Would you please explain the process?
'''

results = processor.process_document(sample_text)
print(json.dumps(results, indent=2))
```

The `process_document` method extracts questions from the given text, finds similar questions using FAISS, and generates variations of the questions in formal and casual styles using Ollama.

3. The processed questions and their analysis are returned as a dictionary.

## Methods

- `extract_questions(text: str) -> List[str]`: Extract questions from a given text using spaCy's linguistic features.
- `compute_embeddings(text: str) -> np.ndarray`: Compute embeddings for a given text using Sentence Transformers.
- `find_similar_questions(question: str, threshold: float = 0.8, limit: int = 5) -> List[Dict]`: Find similar questions using FAISS similarity search.
- `rewrite_question(question: str, style: str = 'formal') -> str`: Rewrite a question in a specified style using Ollama.
- `add_question(question: str, category: Optional[str] = None) -> int`: Add a question to both FAISS and traditional storage.
- `process_document(text: str) -> Dict`: Process a document and extract questions with analysis.

## Configuration

- `db_path`: Path to store the JSON database (default: "questions.json")
- `ollama_model`: Name of the Ollama model to use (default: "llama2")
- `ollama_host`: Ollama API host (default: "http://localhost:11434")

## License

This project is licensed under the MIT License.