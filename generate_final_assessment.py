import json
from pathlib import Path
import statistics
import time

def load_stream_assessment(stream):
    """Load assessment for a specific stream."""
    assessment_file = Path(f'streams/{stream}/validated/stream_assessment.json')
    if not assessment_file.exists():
        print(f"Warning: No assessment found for {stream} stream")
        return None
    
    with open(assessment_file, 'r') as f:
        return json.load(f)

def generate_final_assessment():
    """Generate comprehensive assessment across all streams."""
    streams = ['knowledge', 'hallucination', 'problem_solving', 'reasoning', 'consistency']
    stream_assessments = {}
    
    # Load all stream assessments
    for stream in streams:
        assessment = load_stream_assessment(stream)
        if assessment:
            stream_assessments[stream] = assessment
    
    if not stream_assessments:
        raise ValueError("No stream assessments found")
    
    # Calculate cross-stream metrics
    total_questions = sum(sa['total_questions'] for sa in stream_assessments.values())
    
    # Common metrics across all streams
    common_metrics = ['accuracy', 'reasoning', 'completeness']
    overall_metrics = {}
    
    for metric in common_metrics:
        scores = []
        for assessment in stream_assessments.values():
            if metric in assessment['overall_statistics']:
                scores.append(assessment['overall_statistics'][metric]['mean'])
        
        if scores:
            overall_metrics[metric] = {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0
            }
    
    # Stream-specific performance analysis
    stream_performance = {}
    for stream, assessment in stream_assessments.items():
        # Get average of all metrics for this stream
        metrics = assessment['overall_statistics']
        scores = [stat['mean'] for stat in metrics.values()]
        
        stream_performance[stream] = {
            'average_score': statistics.mean(scores),
            'strongest_metrics': [],
            'weakest_metrics': []
        }
        
        # Find strongest and weakest metrics
        sorted_metrics = sorted(
            metrics.items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )
        
        # Top 3 strongest
        stream_performance[stream]['strongest_metrics'] = [
            {'metric': m[0], 'score': m[1]['mean']}
            for m in sorted_metrics[:3]
        ]
        
        # Bottom 3 weakest
        stream_performance[stream]['weakest_metrics'] = [
            {'metric': m[0], 'score': m[1]['mean']}
            for m in sorted_metrics[-3:]
        ]
    
    # Generate final assessment
    final_assessment = {
        'total_questions_evaluated': total_questions,
        'streams_evaluated': list(stream_assessments.keys()),
        'overall_metrics': overall_metrics,
        'stream_performance': stream_performance,
        'stream_assessments': stream_assessments,
        'model_strengths': [],
        'model_weaknesses': [],
        'recommendations': [],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Analyze overall strengths and weaknesses
    sorted_streams = sorted(
        stream_performance.items(),
        key=lambda x: x[1]['average_score'],
        reverse=True
    )
    
    # Add top performing streams to strengths
    final_assessment['model_strengths'].extend([
        f"Strong performance in {stream} tasks (avg score: {perf['average_score']:.2f})"
        for stream, perf in sorted_streams[:2]
    ])
    
    # Add top metrics to strengths
    for metric, stats in overall_metrics.items():
        if stats['mean'] >= 8.5:  # High performance threshold
            final_assessment['model_strengths'].append(
                f"Excellent {metric} across all streams (score: {stats['mean']:.2f})"
            )
    
    # Add lowest performing streams to weaknesses
    final_assessment['model_weaknesses'].extend([
        f"Lower performance in {stream} tasks (avg score: {perf['average_score']:.2f})"
        for stream, perf in sorted_streams[-2:]
    ])
    
    # Add low metrics to weaknesses
    for metric, stats in overall_metrics.items():
        if stats['mean'] < 7.0:  # Low performance threshold
            final_assessment['model_weaknesses'].append(
                f"Needs improvement in {metric} (score: {stats['mean']:.2f})"
            )
    
    # Generate recommendations based on analysis
    if final_assessment['model_weaknesses']:
        final_assessment['recommendations'].append(
            "Consider fine-tuning for specific weak areas"
        )
    
    high_variance_metrics = [
        metric for metric, stats in overall_metrics.items()
        if stats['std_dev'] > 1.5
    ]
    if high_variance_metrics:
        final_assessment['recommendations'].append(
            f"Improve consistency in: {', '.join(high_variance_metrics)}"
        )
    
    # Save final assessment
    output_file = Path('final_assessment.json')
    with open(output_file, 'w') as f:
        json.dump(final_assessment, f, indent=2)
    
    print(f"\nFinal assessment saved to {output_file}")
    return final_assessment

if __name__ == '__main__':
    generate_final_assessment()
