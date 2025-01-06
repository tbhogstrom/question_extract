"""
Main script to demonstrate the question processing system.
"""
from text_parser import extract_questions, clean_question
from vector_ops import create_word_vector, find_similar_questions
from question_store import QuestionStore
from question_rewriter import rewrite_question

def process_document(file_path, store):
    """Process a document and extract questions."""
    with open(file_path, 'r') as f:
        text = f.read()
    
    # Extract and clean questions
    questions = extract_questions(text)
    cleaned_questions = [clean_question(q) for q in questions]
    
    # Create vectors for similarity checking
    vectors = []
    vocabulary = None
    for question in cleaned_questions:
        vector, vocabulary = create_word_vector(question, vocabulary)
        vectors.append(vector)
    
    question_vectors = {
        'vectors': vectors,
        'vocabulary': vocabulary
    }
    
    # Process each question
    for question in cleaned_questions:
        # Check for similar questions
        similar = find_similar_questions(question, question_vectors)
        
        if not similar:  # No duplicates found
            # Store the question
            store.add_question(question)
            
            # Demonstrate style rewriting
            formal_version = rewrite_question(question, 'formal')
            casual_version = rewrite_question(question, 'casual')
            
            print(f"Original: {question}")
            print(f"Formal: {formal_version}")
            print(f"Casual: {casual_version}")
            print("-" * 50)

if __name__ == "__main__":
    # Initialize the question store
    store = QuestionStore()
    
    # Example usage
    sample_text = """
    This is a sample document. What is the meaning of life?
    How do we process text effectively? The system should handle
    various types of questions. Would you please explain the process?
    """
    
    with open("sample.txt", "w") as f:
        f.write(sample_text)
    
    process_document("sample.txt", store)