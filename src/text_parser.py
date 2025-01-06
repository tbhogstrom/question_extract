"""
Text parsing and question extraction module.
"""
import re

def extract_questions(text):
    """Extract questions from text using regex patterns."""
    # Match both question mark endings and implicit questions
    patterns = [
        r'[^.!?]*\?',  # Explicit questions ending with ?
        r'(what|who|where|when|why|how)\s+\w+[^?]*[.!]'  # Implicit questions
    ]
    
    questions = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        questions.extend(match.group().strip() for match in matches)
    
    return questions

def clean_question(question):
    """Clean and normalize question text."""
    # Remove extra whitespace and normalize punctuation
    cleaned = ' '.join(question.split())
    cleaned = re.sub(r'\s+([.,?!])', r'\1', cleaned)
    return cleaned