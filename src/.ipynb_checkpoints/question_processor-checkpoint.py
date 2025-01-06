"""
Question extraction system using FAISS, Ollama, and modern NLP tools.
"""
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.llms import Ollama
from typing import List, Dict, Optional
import json
from datetime import datetime
import os

class InMemoryQuestionProcessor:
    def __init__(self,
                 db_path: str = "questions.json",
                 ollama_model: str = "llama2",
                 ollama_host: str = "http://localhost:11434"):
        """
        Initialize the question processing system with FAISS.
        
        Args:
            db_path: Path to store the JSON database
            ollama_model: Name of the Ollama model to use
            ollama_host: Ollama API host
        """
        # Initialize NLP components
        self.nlp = spacy.load("en_core_web_sm")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Size for MiniLM embeddings
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product index
        
        # Initialize Ollama
        self.llm = Ollama(model=ollama_model, base_url=ollama_host)
        
        # Initialize storage
        self.db_path = db_path
        self.questions = self._load_db()
        
        # Add existing questions to FAISS
        self._load_vectors()

    def _load_vectors(self):
        """Load existing questions into FAISS index."""
        if self.questions['questions']:
            vectors = [
                self.compute_embeddings(q['text'])
                for q in self.questions['questions']
            ]
            if vectors:
                vectors = np.array(vectors).astype('float32')
                self.index.add(vectors)

    def extract_questions(self, text: str) -> List[str]:
        """Extract questions using spaCy's linguistic features."""
        doc = self.nlp(text)
        questions = []
        
        for sent in doc.sents:
            # Check for question marks
            if sent.text.strip().endswith('?'):
                questions.append(sent.text.strip())
                continue
            
            # Check for question words and structure
            doc_sent = self.nlp(sent.text)
            if any(token.tag_ in ['WDT', 'WP', 'WRB'] for token in doc_sent):
                questions.append(sent.text.strip())
        
        return questions

    def compute_embeddings(self, text: str) -> np.ndarray:
        """Compute embeddings using Sentence Transformers."""
        return self.encoder.encode([text])[0]

    def find_similar_questions(self,
                             question: str,
                             threshold: float = 0.8,
                             limit: int = 5) -> List[Dict]:
        """Find similar questions using FAISS similarity search."""
        if self.index.ntotal == 0:
            return []
            
        # Compute query vector
        query_vector = self.compute_embeddings(question)
        query_vector = np.array([query_vector]).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_vector, min(limit, self.index.ntotal))
        
        # Process results
        similar_questions = []
        for score, idx in zip(distances[0], indices[0]):
            if score >= threshold:
                similar_questions.append({
                    'id': int(idx),
                    'score': float(score),
                    'question': self.questions['questions'][idx]['text']
                })
        
        return similar_questions

    def rewrite_question(self, question: str, style: str = 'formal') -> str:
        """Rewrite question using Ollama with specified style."""
        prompt = f"""
        Task: Rewrite the following question in a {style} style.
        Keep the same meaning but adjust the tone and vocabulary to match the style.
        
        Question: {question}
        
        Rewritten question:"""
        
        return self.llm.predict(prompt).strip()

    def _load_db(self) -> Dict:
        """Load questions from JSON file."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {'questions': [], 'categories': {}}

    def _save_db(self):
        """Save questions to JSON file."""
        with open(self.db_path, 'w') as f:
            json.dump(self.questions, f, indent=2)

    def add_question(self,
                    question: str,
                    category: Optional[str] = None) -> int:
        """Add a question to both FAISS and traditional storage."""
        # Add to traditional storage
        question_id = len(self.questions['questions'])
        question_data = {
            'id': question_id,
            'text': question,
            'category': category,
            'created_at': datetime.now().isoformat()
        }
        
        self.questions['questions'].append(question_data)
        
        if category:
            if category not in self.questions['categories']:
                self.questions['categories'][category] = []
            self.questions['categories'][category].append(question_id)
        
        # Add to FAISS
        vector = self.compute_embeddings(question)
        self.index.add(np.array([vector]).astype('float32'))
        
        self._save_db()
        return question_id

    def process_document(self, text: str) -> Dict:
        """Process a document and extract questions with analysis."""
        questions = self.extract_questions(text)
        results = []
        
        for question in questions:
            # Find similar questions
            similar = self.find_similar_questions(question)
            
            # Generate variations using Ollama
            formal = self.rewrite_question(question, 'formal')
            casual = self.rewrite_question(question, 'casual')
            
            # Store if no similar questions found
            if not similar:
                question_id = self.add_question(question)
            
            results.append({
                'original': question,
                'similar': similar,
                'variations': {
                    'formal': formal,
                    'casual': casual
                }
            })
        
        return {'processed_questions': results}

# Example usage:
"""
processor = InMemoryQuestionProcessor()

sample_text = '''
This is a sample document. What is the meaning of life?
How do we process text effectively? The system should handle
various types of questions. Would you please explain the process?
'''

results = processor.process_document(sample_text)
print(json.dumps(results, indent=2))
"""

# Requirements for requirements.txt:
"""
spacy>=3.7.2
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
langchain>=0.1.0
ollama>=0.1.1
"""