import csv
import json
import os
from pathlib import Path
import ollama
import argparse
import time

def get_prompt(stream, question, category):
    """Get the appropriate prompt for each stream."""
    prompts = {
        'knowledge': f"""Category: {category}
Question: {question}
Please provide a clear, factual answer. Be concise but thorough.""",
        
        'hallucination': f"""Category: {category}
Question: {question}
If you are uncertain about any aspect of the answer, please explicitly state your uncertainty. 
If the question contains false premises, point them out.
Provide a clear, honest response.""",
        
        'problem_solving': f"""Category: {category}
Question: {question}
Please solve this problem step by step:
1. First, outline your approach
2. Show each step of your solution
3. Verify your answer
Be clear and thorough in your explanation.""",
        
        'reasoning': f"""Category: {category}
Question: {question}
Please provide a structured analysis:
1. Identify key factors and relationships
2. Analyze implications and connections
3. Draw logical conclusions
4. Consider limitations and assumptions""",
        
        'consistency': f"""Category: {category}
Question: {question}
Provide a clear, precise answer based solely on verified facts.
If any aspect is uncertain, explicitly state this."""
    }
    return prompts.get(stream)

def query_gemma(stream, question, category):
    """Query Gemma 3 1B model with appropriate prompt for the stream."""
    prompt = get_prompt(stream, question, category)
    response = ollama.chat(
        model='gemma3:1b',
        messages=[
            {
                'role': 'system',
                'content': 'You are a knowledgeable AI assistant. Provide accurate, factual answers.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ],
        options={
            'temperature': 0
        }
    )
    return response.message.content.strip()

def process_stream(stream):
    """Process all questions for a given stream."""
    # Set up paths
    questions_file = f'streams/{stream}/questions/{stream}_questions.csv'
    answers_1b_dir = f'streams/{stream}/answers_1b'
    
    # Ensure answers directory exists
    Path(answers_1b_dir).mkdir(exist_ok=True, parents=True)
    
    # Read questions
    with open(questions_file, 'r') as f:
        reader = csv.DictReader(f)
        questions = list(reader)
    
    # Process each question
    for i, q in enumerate(questions, 1):
        print(f"[{stream}] Processing question {i}/{len(questions)}: {q['question']}")
        
        # Check if answer already exists
        answer_file = f'{answers_1b_dir}/question_{i:03d}.json'
        if os.path.exists(answer_file):
            print(f"Answer already exists: {answer_file}")
            continue
            
        answer = query_gemma(stream, q['question'], q['category'])
        
        # Save answer
        output = {
            'stream': stream,
            'category': q['category'],
            'question': q['question'],
            'answer': answer,
            'index': i
        }
        
        with open(answer_file, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"Answer saved to {answer_file}")
        time.sleep(1)  # Rate limit

def main():
    parser = argparse.ArgumentParser(description='Query Gemma 3 1B model with different question streams')
    parser.add_argument('--stream', choices=['knowledge', 'hallucination', 'problem_solving', 'reasoning', 'consistency'],
                        help='Which question stream to process')
    parser.add_argument('--all', action='store_true', help='Process all streams')
    
    args = parser.parse_args()
    
    if args.all:
        streams = ['knowledge', 'hallucination', 'problem_solving', 'reasoning', 'consistency']
    elif args.stream:
        streams = [args.stream]
    else:
        parser.error('Must specify either --stream or --all')
    
    for stream in streams:
        print(f"\nProcessing {stream} stream...")
        process_stream(stream)

if __name__ == '__main__':
    main()
