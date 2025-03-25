import csv
import json
import os
from pathlib import Path
import ollama

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    Path('answers').mkdir(exist_ok=True)

def read_questions(csv_file='questions.csv'):
    """Read questions from CSV file."""
    questions = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
    return questions

def query_gemma(question, category):
    """Query Gemma 3 1B model with temperature 0."""
    prompt = f"""Category: {category}
Question: {question}
Please provide a clear, factual answer. Be concise but thorough."""
    
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

def save_answer(category, question, answer, index):
    """Save question and answer to a JSON file."""
    output = {
        'category': category,
        'question': question,
        'answer': answer,
        'index': index
    }
    
    filename = f'answers/question_{index:03d}.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

def main():
    ensure_output_dir()
    questions = read_questions()
    
    for i, row in enumerate(questions, 1):
        print(f"Processing question {i}/100: {row['question']}")
        answer = query_gemma(row['question'], row['category'])
        save_answer(row['category'], row['question'], answer, i)
        print(f"Answer saved to question_{i:03d}.json")

if __name__ == '__main__':
    main()
