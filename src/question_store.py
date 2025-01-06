"""
Simple database operations for storing questions.
"""
import json
import os
from datetime import datetime

class QuestionStore:
    def __init__(self, db_file='questions.json'):
        self.db_file = db_file
        self.questions = self._load_db()
    
    def _load_db(self):
        """Load questions from JSON file."""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r') as f:
                return json.load(f)
        return {'questions': [], 'categories': {}}
    
    def _save_db(self):
        """Save questions to JSON file."""
        with open(self.db_file, 'w') as f:
            json.dump(self.questions, f, indent=2)
    
    def add_question(self, question, category=None):
        """Add a new question to the store."""
        question_data = {
            'id': len(self.questions['questions']),
            'text': question,
            'category': category,
            'created_at': datetime.now().isoformat()
        }
        
        self.questions['questions'].append(question_data)
        
        if category:
            if category not in self.questions['categories']:
                self.questions['categories'][category] = []
            self.questions['categories'][category].append(question_data['id'])
        
        self._save_db()
        return question_data['id']
    
    def get_questions(self, category=None):
        """Retrieve questions, optionally filtered by category."""
        if category:
            question_ids = self.questions['categories'].get(category, [])
            return [q for q in self.questions['questions'] if q['id'] in question_ids]
        return self.questions['questions']