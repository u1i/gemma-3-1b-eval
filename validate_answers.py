import json
import os
from pathlib import Path
import requests
import statistics
import time
import argparse

def get_27b_answer(question, category):
    """Get Gemma 27B's own answer to the same question."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    headers = {
        'Authorization': f'Bearer {api_key}',
        'HTTP-Referer': 'http://localhost:3000',
        'X-Title': 'Gemma 1B Evaluation'
    }

    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers=headers,
        json={
            'model': 'google/gemma-3-27b-it',
            'messages': [
                {
                    'role': 'user',
                    'content': f"Category: {category}\nQuestion: {question}"
                }
            ]
        }
    )

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.text}")

    return response.json()['choices'][0]['message']['content'].strip()

def get_validation_prompt(stream, category, question, answer_1b, answer_27b):
    """Get the appropriate validation prompt for each stream."""
    
    # Common comparison intro for all streams
    comparison_intro = f"""You are comparing responses from two versions of the Gemma model (1B vs 27B) to the same question.

Category: {category}
Question: {question}

Gemma 1B Answer: {answer_1b}

Your Own Answer (27B): {answer_27b}

Compare these responses carefully. Note where they agree, differ, or where one might be better/worse than the other."""

    stream_prompts = {
        'knowledge': f"""{comparison_intro}

Evaluate both responses based on:
1. Factual accuracy
2. Level of detail
3. Reasoning quality
4. Completeness

Category: {category}
Question: {question}
Model's Answer: {answer_1b}

Evaluate the response based on factual accuracy, reasoning quality, and completeness.
Provide your assessment in the following JSON format:
{{
    "accuracy": <score 0-10>,
    "reasoning": <score 0-10>,
    "completeness": <score 0-10>,
    "factual_correctness": <score 0-10>,
    "source_alignment": <score 0-10>,
    "explanation": "detailed explanation of scores",
    "strengths": ["list", "of", "strengths"],
    "weaknesses": ["list", "of", "weaknesses"]
}}""",
        
        'hallucination': f"""{comparison_intro}

Compare how each model:
1. Handled the false premises
2. Expressed uncertainty
3. Avoided fabrication
4. Provided corrections

Category: {category}
Question: {question}
Model's Answer: {answer_1b}

Evaluate how well the model:
1. Identified false premises in the question
2. Expressed appropriate uncertainty
3. Avoided making up false information
4. Provided accurate corrections

Provide your assessment in the following JSON format:
{{
    "accuracy": <score 0-10>,
    "reasoning": <score 0-10>,
    "completeness": <score 0-10>,
    "uncertainty_awareness": <score 0-10; how well did it express uncertainty>,
    "false_premise_detection": <score 0-10; did it identify the false premises>,
    "invention_score": <score 0-10; 10=no fabrication, 0=heavy fabrication>,
    "explanation": "detailed analysis of how the model handled the false premises",
    "strengths": ["list", "of", "strengths"],
    "weaknesses": ["list", "of", "weaknesses"],
    "false_premises_identified": ["list", "of", "false premises", "the model caught"],
    "fabricated_elements": ["list", "of", "any information", "the model made up"]
}}""",
        
        'problem_solving': f"""{comparison_intro}

Compare the solutions based on:
1. Problem-solving approach
2. Step clarity
3. Mathematical accuracy
4. Solution verification

Category: {category}
Question: {question}
Model's Answer: {answer_1b}

Evaluate the solution based on:
1. Problem-solving methodology
2. Step-by-step clarity
3. Mathematical/logical accuracy
4. Solution verification

Provide your assessment in the following JSON format:
{{
    "accuracy": <score 0-10>,
    "reasoning": <score 0-10>,
    "completeness": <score 0-10>,
    "methodology": <score 0-10; how well structured was the approach>,
    "step_clarity": <score 0-10; how clear were the solution steps>,
    "solution_correctness": <score 0-10; is the final answer correct>,
    "explanation": "detailed analysis of the problem-solving approach",
    "strengths": ["list", "of", "strengths"],
    "weaknesses": ["list", "of", "weaknesses"],
    "missing_steps": ["steps", "that should have been included"],
    "errors": ["any", "logical or mathematical errors"]
}}""",
        
        'reasoning': f"""{comparison_intro}

Compare the analyses based on:
1. Logical structure
2. Depth of reasoning
3. Factor consideration
4. Assumption handling

Category: {category}
Question: {question}
Model's Answer: {answer_1b}

Evaluate the analysis based on:
1. Logical coherence
2. Depth of analysis
3. Quality of inferences
4. Consideration of assumptions and limitations

Provide your assessment in the following JSON format:
{{
    "accuracy": <score 0-10>,
    "reasoning": <score 0-10>,
    "completeness": <score 0-10>,
    "logical_coherence": <score 0-10; how well does the reasoning flow>,
    "depth_of_analysis": <score 0-10; how thorough is the analysis>,
    "assumption_awareness": <score 0-10; how well does it handle assumptions>,
    "explanation": "detailed analysis of the reasoning quality",
    "strengths": ["list", "of", "strengths"],
    "weaknesses": ["list", "of", "weaknesses"],
    "logical_gaps": ["any", "gaps in reasoning"],
    "unexamined_factors": ["important factors", "not considered"]
}}""",
        
        'consistency': f"""{comparison_intro}

Compare the responses based on:
1. Factual alignment
2. Detail consistency
3. Context handling
4. Uncertainty expression

Category: {category}
Question: {question}
Model's Answer: {answer_1b}

Note: This question is part of a pair testing the same knowledge from different angles.
Evaluate the response based on:
1. Factual stability across different phrasings
2. Consistency of core information
3. Appropriate uncertainty disclosure

Provide your assessment in the following JSON format:
{{
    "accuracy": <score 0-10>,
    "reasoning": <score 0-10>,
    "completeness": <score 0-10>,
    "fact_stability": <score 0-10; consistency of facts>,
    "context_awareness": <score 0-10; understanding of question context>,
    "uncertainty_disclosure": <score 0-10; appropriate expression of uncertainty>,
    "explanation": "detailed analysis of response consistency",
    "strengths": ["list", "of", "strengths"],
    "weaknesses": ["list", "of", "weaknesses"],
    "inconsistencies": ["any", "detected inconsistencies"],
    "context_mismatches": ["areas where", "context was misunderstood"]
}}"""
    }
    
    return stream_prompts.get(stream, stream_prompts['knowledge'])

def validate_with_gemma27b(stream, answer_file):
    """Validate an answer using the Gemma 3 27B model via OpenRouter."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    with open(answer_file, 'r') as f:
        data = json.load(f)

    # First get 27B's own answer to the same question
    answer_27b = get_27b_answer(data['question'], data['category'])
    
    # Then get the comparison validation
    # Get the comparison validation
    prompt = get_validation_prompt(
        stream,
        data['category'],
        data['question'],
        data['answer'],  # 1B's answer
        answer_27b      # 27B's own answer
    )

    headers = {
        'Authorization': f'Bearer {api_key}',
        'HTTP-Referer': 'http://localhost:3000',
        'X-Title': 'Gemma 1B Evaluation'
    }

    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers=headers,
        json={
            'model': 'google/gemma-3-27b-it',
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are an expert evaluator analyzing responses from an AI model.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        }
    )

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.text}")

    try:
        validation = response.json()['choices'][0]['message']['content']
        # Clean up the response - remove markdown code blocks and any leading/trailing characters
        validation = validation.replace('```json\n', '').replace('\n```', '')
        # Find the first '{' and last '}' to extract just the JSON
        start = validation.find('{')
        end = validation.rfind('}')
        if start != -1 and end != -1:
            validation = validation[start:end+1]
            return json.loads(validation)
        else:
            print(f"Could not find JSON object in response")
            print(f"Raw response: {validation}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error parsing validation response: {e}")
        print(f"Raw response: {validation}")
        return None

def analyze_results(stream, validated_files):
    """Analyze validation results for a stream."""
    results = []
    categories = {}
    
    # Collect all results
    for file in validated_files:
        with open(file, 'r') as f:
            results.append(json.load(f))
    
    # Calculate statistics per metric
    metrics = ['accuracy', 'reasoning', 'completeness']
    
    # Add stream-specific metrics
    stream_metrics = {
        'knowledge': ['factual_correctness', 'source_alignment'],
        'hallucination': ['uncertainty_awareness', 'false_premise_detection', 'invention_score'],
        'problem_solving': ['methodology', 'step_clarity', 'solution_correctness'],
        'reasoning': ['logical_coherence', 'depth_of_analysis', 'assumption_awareness'],
        'consistency': ['fact_stability', 'context_awareness', 'uncertainty_disclosure']
    }
    
    metrics.extend(stream_metrics[stream])
    
    # Calculate overall statistics
    overall_stats = {}
    for metric in metrics:
        scores = [r['validation'][metric] for r in results if metric in r.get('validation', {})]
        if scores:
            overall_stats[metric] = {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0
            }
    
    # Calculate per-category statistics
    for result in results:
        category = result['category']
        if category not in categories:
            categories[category] = {metric: [] for metric in metrics}
        
        for metric in metrics:
            if metric in result.get('validation', {}):
                categories[category][metric].append(result['validation'][metric])
    
    category_stats = {}
    for category, metrics_data in categories.items():
        category_stats[category] = {}
        for metric, scores in metrics_data.items():
            if scores:
                category_stats[category][metric] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0
                }
    
    # Compile stream-specific insights
    stream_insights = {
        'hallucination': {
            'false_premises_caught': [],
            'fabrication_instances': [],
            'uncertainty_handling': []
        },
        'problem_solving': {
            'common_errors': [],
            'methodology_patterns': [],
            'solution_quality': []
        },
        'reasoning': {
            'reasoning_patterns': [],
            'analysis_depth': [],
            'assumption_handling': []
        },
        'consistency': {
            'inconsistency_patterns': [],
            'context_handling': [],
            'stability_issues': []
        }
    }
    
    if stream in stream_insights:
        for result in results:
            validation = result.get('validation', {})
            if stream == 'hallucination':
                if 'false_premises_identified' in validation:
                    stream_insights[stream]['false_premises_caught'].extend(validation['false_premises_identified'])
                if 'fabricated_elements' in validation:
                    stream_insights[stream]['fabrication_instances'].extend(validation['fabricated_elements'])
            elif stream == 'problem_solving':
                if 'errors' in validation:
                    stream_insights[stream]['common_errors'].extend(validation['errors'])
                if 'missing_steps' in validation:
                    stream_insights[stream]['methodology_patterns'].extend(validation['missing_steps'])
            elif stream == 'reasoning':
                if 'logical_gaps' in validation:
                    stream_insights[stream]['reasoning_patterns'].extend(validation['logical_gaps'])
                if 'unexamined_factors' in validation:
                    stream_insights[stream]['analysis_depth'].extend(validation['unexamined_factors'])
            elif stream == 'consistency':
                if 'inconsistencies' in validation:
                    stream_insights[stream]['inconsistency_patterns'].extend(validation['inconsistencies'])
                if 'context_mismatches' in validation:
                    stream_insights[stream]['context_handling'].extend(validation['context_mismatches'])
    
    # Compile final assessment
    assessment = {
        'stream': stream,
        'total_questions': len(results),
        'overall_statistics': overall_stats,
        'category_statistics': category_stats,
        'stream_specific_insights': stream_insights.get(stream, {}),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return assessment

def process_stream(stream):
    """Process validation for a specific stream."""
    answers_1b_dir = f'streams/{stream}/answers_1b'
    answers_27b_dir = f'streams/{stream}/answers_27b'
    validated_dir = f'streams/{stream}/validated'
    
    # Create directories if they don't exist
    Path(answers_27b_dir).mkdir(exist_ok=True, parents=True)
    Path(validated_dir).mkdir(exist_ok=True, parents=True)
    
    # Create validated directory if it doesn't exist
    Path(validated_dir).mkdir(exist_ok=True, parents=True)
    
    # Get list of answer files
    answer_files = sorted(Path(answers_1b_dir).glob('question_*.json'))
    
    validated_files = []
    for answer_file in answer_files:
        validated_file = Path(validated_dir) / f"validated_{answer_file.name}"
        
        # Skip if both validation and 27B answer exist
        answer_27b_file = Path(answers_27b_dir) / f"27b_{answer_file.name}"
        if validated_file.exists() and answer_27b_file.exists():
            print(f"Already validated: {answer_file.name}")
            validated_files.append(validated_file)
            continue
            
        print(f"Validating {answer_file.name}...")
        
        try:
            # Get 27B's own answer first
            with open(answer_file, 'r') as f:
                data = json.load(f)
            
            # Save 27B's answer in its own file
            answer_27b = get_27b_answer(data['question'], data['category'])
            answer_27b_file = Path(answers_27b_dir) / f"27b_{answer_file.name}"
            with open(answer_27b_file, 'w') as f:
                json.dump({
                    'stream': stream,
                    'category': data['category'],
                    'question': data['question'],
                    'answer': answer_27b,
                    'index': data['index']
                }, f, indent=2)
            
            # Get validation comparing both answers
            validation = validate_with_gemma27b(stream, answer_file)
            if validation:
                # Save validation result separately
                validation_data = {
                    'stream': stream,
                    'category': data['category'],
                    'question': data['question'],
                    'answer_1b': data['answer'],
                    'answer_27b': answer_27b,
                    'validation': validation,
                    'index': data['index']
                }
                
                with open(validated_file, 'w') as f:
                    json.dump(validation_data, f, indent=2)
                    
                validated_files.append(validated_file)
                print(f"Validation saved to {validated_file}")
                
                # Rate limiting
                time.sleep(1)
            
        except Exception as e:
            print(f"Error validating {answer_file.name}: {e}")
    
    if validated_files:
        # Generate assessment
        assessment = analyze_results(stream, validated_files)
        
        # Save assessment
        assessment_file = Path(validated_dir) / 'stream_assessment.json'
        with open(assessment_file, 'w') as f:
            json.dump(assessment, f, indent=2)
        print(f"\nAssessment saved to {assessment_file}")

def main():
    parser = argparse.ArgumentParser(description='Validate Gemma 3 1B answers using Gemma 3 27B')
    parser.add_argument('--stream', choices=['knowledge', 'hallucination', 'problem_solving', 'reasoning', 'consistency'],
                        help='Which stream to validate')
    parser.add_argument('--all', action='store_true', help='Validate all streams')
    
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
