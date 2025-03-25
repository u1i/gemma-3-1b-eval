import json
import os
from pathlib import Path
import requests
import time
from collections import defaultdict
import statistics
import re

def read_answer_file(filename):
    """Read a single answer file."""
    with open(filename, 'r') as f:
        return json.load(f)

def validate_with_gemma27b(question, answer, category):
    """Validate an answer using Gemma 27B via OpenRouter."""
    headers = {
        'Authorization': f'Bearer {os.getenv("OPENROUTER_API_KEY")}',
        'Content-Type': 'application/json'
    }
    
    prompt = f"""Analyze this answer and provide scores and analysis in JSON format.

Category: {category}
Question: {question}
Answer: {answer}

Respond with ONLY a valid JSON object in this EXACT format (replace text in <> with actual values):
{{
    "accuracy": {{
        "score": <number 0-10>,
        "issues": [<list of specific issues found>],
        "strengths": [<list of specific strengths>]
    }},
    "reasoning": {{
        "score": <number 0-10>,
        "analysis": <single string analyzing logical flow>
    }},
    "completeness": {{
        "score": <number 0-10>,
        "analysis": <single string analyzing coverage>
    }},
    "knowledge_compression": {{
        "preserved": [<list of specific knowledge elements that survived>],
        "lost": [<list of specific knowledge elements that were lost>],
        "domain_handling": <single string analyzing domain expertise>
    }}    
}}

DO NOT include any text before or after the JSON object. Ensure all strings are properly escaped."""

    data = {
        'model': 'google/gemma-3-27b-it',
        'messages': [
            {
                'role': 'system',
                'content': 'You are an expert evaluator. You must ONLY output valid JSON matching the exact format requested.'
            },
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0,
        'response_format': { 'type': 'json_object' }
    }
    
    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        try:
            content = response.json()['choices'][0]['message']['content']
            # Try to clean up common JSON formatting issues
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {str(e)}")
            print(f"Raw response: {content}")
            return {
                "accuracy": {"score": 0, "issues": [f"Failed to parse response: {str(e)}"], "strengths": []},
                "reasoning": {"score": 0, "analysis": "Validation parsing failed"},
                "completeness": {"score": 0, "analysis": "Validation parsing failed"},
                "knowledge_compression": {
                    "preserved": [], "lost": [],
                    "domain_handling": "Validation parsing failed"
                }
            }
    else:
        print(f"API Error: {response.status_code}")
        print(f"Response: {response.text}")
        return {
            "accuracy": {"score": 0, "issues": [f"API Error: {response.status_code}"], "strengths": []},
            "reasoning": {"score": 0, "analysis": "API request failed"},
            "completeness": {"score": 0, "analysis": "API request failed"},
            "knowledge_compression": {
                "preserved": [], "lost": [],
                "domain_handling": "API request failed"
            }
        }

def analyze_results(validated_files):
    """Analyze validation results and create overall assessment."""
    metrics = ['accuracy', 'reasoning', 'completeness']
    category_scores = defaultdict(lambda: {metric: [] for metric in metrics})
    all_scores = {metric: [] for metric in metrics}
    knowledge_insights = defaultdict(list)
    error_count = 0
    processed_files = set()
    
    # Check for existing assessment
    if os.path.exists('validated/final_assessment.json'):
        try:
            with open('validated/final_assessment.json', 'r') as f:
                print("Found existing assessment, skipping processed files...")
                existing = json.load(f)
                if 'processed_files' in existing:
                    processed_files = set(existing['processed_files'])
        except json.JSONDecodeError:
            print("Warning: Could not read existing assessment file")
    
    for file in validated_files:
        if str(file) in processed_files:
            print(f"Skipping already processed file: {file}")
            continue
            
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                validation = data['validation']
                category = data['category']
                
                # Collect scores for each metric
                for metric in metrics:
                    try:
                        score = validation[metric]['score']
                        category_scores[category][metric].append(score)
                        all_scores[metric].append(score)
                    except (KeyError, TypeError):
                        error_count += 1
                
                # Collect knowledge compression insights
                if 'knowledge_compression' in validation:
                    knowledge_insights[category].append({
                        'question_index': data['index'],
                        'preserved': validation['knowledge_compression']['preserved'],
                        'lost': validation['knowledge_compression']['lost'],
                        'domain_handling': validation['knowledge_compression']['domain_handling']
                    })
                    
                processed_files.add(str(file))
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    
    # Calculate overall statistics for each metric
    overall_stats = {}
    for metric in metrics:
        scores = all_scores[metric]
        if scores:
            overall_stats[metric] = {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0
            }
    
    # Calculate per-category statistics
    category_stats = {}
    for category, metrics_scores in category_scores.items():
        category_stats[category] = {}
        for metric, scores in metrics_scores.items():
            if scores:
                category_stats[category][metric] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'count': len(scores)
                }
    
    # Sort categories by average performance across metrics
    def get_category_avg(category_data):
        scores = []
        for metric in metrics:  # Only look at our defined metrics
            if metric in category_data and isinstance(category_data[metric], dict):
                if 'mean' in category_data[metric]:
                    scores.append(category_data[metric]['mean'])
        return statistics.mean(scores) if scores else 0
    
    sorted_categories = sorted(
        category_stats.items(),
        key=lambda x: get_category_avg(x[1]),
        reverse=True
    )
    
    assessment = {
        'overall_stats': overall_stats,
        'category_performance': category_stats,
        'best_categories': [cat for cat, _ in sorted_categories[:3]],
        'worst_categories': [cat for cat, _ in sorted_categories[-3:]],
        'total_questions': len(processed_files),
        'error_count': error_count,
        'knowledge_compression_insights': dict(knowledge_insights),
        'processed_files': list(processed_files)  # Save processed files
    }
    
    return assessment

def generate_assessment_prompt(assessment):
    """Generate a prompt for the final assessment summary."""
    return f"""Based on the evaluation of a 1B parameter model's responses, analyze:

Overall Statistics:
{json.dumps(assessment['overall_stats'], indent=2)}

Best performing categories:
{', '.join(assessment['best_categories'])}

Worst performing categories:
{', '.join(assessment['worst_categories'])}

Total questions evaluated: {assessment['total_questions']}
Errors encountered: {assessment['error_count']}

Please analyze:
1. Overall performance across accuracy, reasoning, and completeness
2. Areas of strength and weakness by domain
3. Knowledge compression effectiveness (what survived/was lost)
4. Recommendations for model usage

Format your response as a detailed but concise analysis."""

def get_final_assessment(assessment):
    """Get final assessment from Gemma 27B."""
    headers = {
        'Authorization': f'Bearer {os.getenv("OPENROUTER_API_KEY")}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'google/gemma-3-27b-it',
        'messages': [
            {'role': 'user', 'content': generate_assessment_prompt(assessment)}
        ],
        'temperature': 0
    }
    
    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error generating final assessment: {response.status_code}"

def save_validation(original_data, validation_result, index):
    """Save validation results."""
    output = {
        **original_data,
        'validation': validation_result
    }
    
    filename = f'validated/question_{index:03d}_validated.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

def save_final_assessment(assessment, final_analysis):
    """Save final assessment and analysis."""
    output = {
        'raw_assessment': assessment,
        'analysis': final_analysis
    }
    
    with open('validated/final_assessment.json', 'w') as f:
        json.dump(output, f, indent=2)

def main():
    # Create validation directory
    Path('validated').mkdir(exist_ok=True)
    
    # Get all answer files and check which ones need validation
    answer_files = sorted(Path('answers').glob('question_*.json'))
    files_to_validate = []
    for file in answer_files:
        validated_file = Path('validated') / f"question_{int(file.stem.split('_')[1]):03d}_validated.json"
        if not validated_file.exists():
            files_to_validate.append((file, validated_file))
    
    # Show validation plan
    if not files_to_validate:
        print("All files already validated, proceeding to analysis")
    else:
        print(f"Need to validate {len(files_to_validate)} files")
    
    # Validate only unvalidated answers
    for answer_file, validated_file in files_to_validate:
        print(f"Validating {answer_file.name}")
        data = read_answer_file(answer_file)
        
        validation = validate_with_gemma27b(
            data['question'],
            data['answer'],
            data['category']
        )
        
        save_validation(data, validation, data['index'])
        time.sleep(1)  # Rate limit
    
    # Generate overall assessment
    validated_files = sorted(Path('validated').glob('question_*_validated.json'))
    assessment = analyze_results(validated_files)
    
    # Get final analysis from Gemma 27B
    final_analysis = get_final_assessment(assessment)
    
    # Save final assessment
    save_final_assessment(assessment, final_analysis)
    print("Validation complete. Results saved in validated/final_assessment.json")

if __name__ == '__main__':
    if not os.getenv('OPENROUTER_API_KEY'):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        exit(1)
    main()
