"""
Question rewriting based on style and tone analysis.
"""
import re
from collections import defaultdict

class StyleAnalyzer:
    def __init__(self):
        self.style_patterns = {
            'formal': {
                'words': ['please', 'kindly', 'would', 'could'],
                'structures': [
                    r'would you.*\?',
                    r'could you.*\?',
                    r'may I.*\?'
                ]
            },
            'casual': {
                'words': ['hey', 'thanks', 'cool', 'great'],
                'structures': [
                    r'what\'s.*\?',
                    r'how come.*\?',
                    r'why not.*\?'
                ]
            }
        }
    
    def analyze_style(self, text):
        """Analyze the writing style of a text."""
        style_scores = defaultdict(int)
        
        for style, patterns in self.style_patterns.items():
            # Check for style-specific words
            for word in patterns['words']:
                style_scores[style] += len(re.findall(r'\b' + word + r'\b', text.lower()))
            
            # Check for style-specific structures
            for structure in patterns['structures']:
                style_scores[style] += len(re.findall(structure, text.lower()))
        
        return max(style_scores.items(), key=lambda x: x[1])[0] if style_scores else 'neutral'

def rewrite_question(question, target_style='formal'):
    """Rewrite a question in the target style."""
    analyzer = StyleAnalyzer()
    current_style = analyzer.analyze_style(question)
    
    if current_style == target_style:
        return question
    
    # Basic style transformations
    if target_style == 'formal':
        # Convert casual to formal
        question = question.replace("what's", "what is")
        question = question.replace("how come", "why")
        question = re.sub(r'^(\w+)\s', 'Would you please \g<1> ', question.lower())
    else:
        # Convert formal to casual
        question = question.replace("would you please", "can you")
        question = question.replace("may I", "can I")
        question = re.sub(r'^would you.*?(\w+)\s', '\g<1> ', question.lower())
    
    return question[0].upper() + question[1:]