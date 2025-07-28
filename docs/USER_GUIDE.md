# ChatScope User Guide

This comprehensive guide will help you get the most out of ChatScope's conversation analysis capabilities.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Analysis](#basic-analysis)
- [Advanced Features](#advanced-features)
- [Command Line Interface](#command-line-interface)
- [Practical Examples](#practical-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Tips and Tricks](#tips-and-tricks)

## Getting Started

### Prerequisites

1. **Python 3.8 or higher**
2. **OpenAI API Key** - Get one from [OpenAI Platform](https://platform.openai.com/)
3. **ChatGPT conversation export** - Export your conversations from ChatGPT

### Installation

#### Basic Installation
```bash
pip install chatscope
```

#### Full Installation (with all features)
```bash
pip install "chatscope[all]"
```

#### Development Installation
```bash
git clone https://github.com/22smeargle/chatscope.git
cd chatscope
pip install -e ".[dev]"
```

### Setting Up Your API Key

#### Option 1: Environment Variable (Recommended)
```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### Option 2: In Your Code
```python
from chatscope import AdvancedChatGPTAnalyzer

analyzer = AdvancedChatGPTAnalyzer(api_key="your-api-key-here")
```

### Exporting ChatGPT Conversations

1. Go to [ChatGPT Settings](https://chat.openai.com/)
2. Navigate to "Data controls" → "Export data"
3. Request your data export
4. Download the `conversations.json` file when ready

## Basic Analysis

### Your First Analysis

```python
from chatscope import ChatGPTAnalyzer

# Initialize the analyzer
analyzer = ChatGPTAnalyzer()

# Run basic analysis
results = analyzer.analyze('conversations.json')

# View results
print(f"Total conversations: {results['total_conversations']}")
print(f"Categories found: {list(results['category_counts'].keys())}")
```

### Understanding Basic Results

The basic analysis provides:
- **Total conversation count**
- **Category distribution** (Programming, AI, Health, etc.)
- **Visual chart** of categories
- **Detailed categorization** of each conversation

### Customizing Categories

```python
# Define your own categories
custom_categories = [
    "Work Projects",
    "Learning", 
    "Creative Writing",
    "Problem Solving",
    "Personal",
    "Research"
]

analyzer = ChatGPTAnalyzer(categories=custom_categories)
results = analyzer.analyze('conversations.json')
```

## Advanced Features

### Comprehensive Analysis

```python
from chatscope import AdvancedChatGPTAnalyzer

# Initialize advanced analyzer
analyzer = AdvancedChatGPTAnalyzer()

# Run comprehensive analysis
results = analyzer.comprehensive_analysis(
    'conversations.json',
    include_sentiment=True,
    include_topics=True,
    include_temporal=True,
    create_visualizations=True
)
```

### Sentiment Analysis

#### Basic Sentiment Analysis
```python
# Load conversations
conversations = analyzer.load_conversations('conversations.json')

# Analyze sentiment
sentiment_results = analyzer.analyze_sentiment(conversations)

print(f"Average sentiment: {sentiment_results['average_polarity']:.2f}")
print(f"Emotional tone distribution: {sentiment_results['emotion_distribution']}")
```

#### Understanding Sentiment Scores
- **Polarity**: -1 (very negative) to +1 (very positive)
- **Subjectivity**: 0 (objective) to 1 (subjective)
- **Emotions**: joy, sadness, anger, fear, surprise, neutral

#### Sentiment Analysis Example
```python
# Analyze sentiment with emotional tone
sentiment_results = analyzer.analyze_sentiment(
    conversations, 
    include_emotional_tone=True
)

# Find most positive conversations
sentiments = sentiment_results['sentiments']
positive_convos = {
    title: score for title, score in sentiments.items() 
    if score['polarity'] > 0.5
}

print(f"Most positive conversations: {len(positive_convos)}")
for title, score in positive_convos.items():
    print(f"  {title}: {score['polarity']:.2f}")
```

### Topic Modeling

#### LDA Topic Modeling
```python
# Extract topics using LDA
topic_results = analyzer.extract_topics(
    conversations,
    num_topics=8,
    method='LDA'
)

# View discovered topics
for i, topic in enumerate(topic_results['topics']):
    words = ', '.join(topic['words'][:5])  # Top 5 words
    print(f"Topic {i+1}: {words}")
```

#### BERTopic (Advanced)
```python
# Use BERTopic for more sophisticated topic modeling
topic_results = analyzer.extract_topics(
    conversations,
    num_topics=10,
    method='BERTopic'
)

# BERTopic provides better semantic understanding
for topic in topic_results['topics']:
    print(f"Topic: {topic['label']}")
    print(f"Keywords: {', '.join(topic['words'][:7])}")
    print(f"Coherence: {topic.get('coherence', 'N/A')}")
    print()
```

#### Clustering-based Topics
```python
# Use K-means clustering for topic discovery
topic_results = analyzer.extract_topics(
    conversations,
    num_topics=6,
    method='clustering'
)

# Find conversations for each topic
topic_assignments = topic_results['topic_assignments']
for topic_id in range(6):
    convos_in_topic = [
        title for title, assigned_topic in topic_assignments.items()
        if assigned_topic == topic_id
    ]
    print(f"Topic {topic_id}: {len(convos_in_topic)} conversations")
```

### Temporal Analysis

#### Basic Temporal Patterns
```python
# Analyze when you chat most
temporal_results = analyzer.analyze_temporal_patterns(
    conversations,
    time_granularity='daily'
)

print(f"Peak hour: {temporal_results['peak_hour']}:00")
print(f"Most active day: {temporal_results['peak_day']}")
print(f"Overall trend: {temporal_results['trend']}")
```

#### Detailed Time Analysis
```python
# Hourly patterns
hourly = temporal_results['hourly_pattern']
print("\nHourly activity:")
for hour in range(24):
    count = hourly.get(str(hour), 0)
    bar = '█' * (count // 5)  # Simple bar chart
    print(f"{hour:2d}:00 {bar} ({count})")

# Weekly patterns
daily = temporal_results['daily_pattern']
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
print("\nWeekly activity:")
for day in days:
    count = daily.get(day, 0)
    print(f"{day}: {count} conversations")
```

#### Advanced Temporal Analysis
```python
# Detect trends and patterns
temporal_results = analyzer.analyze_temporal_patterns(
    conversations,
    time_granularity='weekly',
    detect_trends=True
)

# Access trend information
if 'trend_analysis' in temporal_results:
    trends = temporal_results['trend_analysis']
    print(f"Conversation frequency trend: {trends['frequency_trend']}")
    print(f"Seasonal patterns detected: {trends['seasonal_patterns']}")
```

### Creating Visualizations

#### Automatic Visualizations
```python
# Create all visualizations automatically
visualizations = analyzer.create_advanced_visualizations(
    results,
    output_dir='./my_analysis_charts'
)

print("Generated visualizations:")
for viz_type, file_path in visualizations.items():
    print(f"  {viz_type}: {file_path}")
```

#### Custom Visualization Settings
```python
# Run analysis with custom visualization settings
results = analyzer.comprehensive_analysis(
    'conversations.json',
    create_visualizations=True,
    output_dir='./detailed_analysis'
)

# Visualizations are saved in the output directory
# - sentiment_analysis.png
# - topic_distribution.png
# - temporal_patterns.png
# - activity_heatmap.png
# - word_cloud.png
```

## Command Line Interface

### Basic CLI Usage

```bash
# Basic analysis
chatscope conversations.json

# Advanced analysis
chatscope-advanced conversations.json --sentiment --topics --temporal
```

### CLI Options

#### Basic CLI
```bash
chatscope conversations.json \
  --output-dir ./results \
  --categories "Work,Learning,Creative,Personal" \
  --batch-size 15
```

#### Advanced CLI
```bash
chatscope-advanced conversations.json \
  --output-dir ./advanced_results \
  --sentiment \
  --topics \
  --temporal \
  --topic-method BERTopic \
  --num-topics 12 \
  --time-granularity hourly \
  --create-visualizations \
  --debug
```

### CLI Examples

#### Quick Analysis
```bash
# Fast basic categorization
chatscope conversations.json --output-dir quick_analysis
```

#### Comprehensive Analysis
```bash
# Full analysis with all features
chatscope-advanced conversations.json \
  --sentiment \
  --topics \
  --temporal \
  --create-visualizations \
  --output-dir comprehensive_analysis
```

#### Topic-Focused Analysis
```bash
# Focus on topic discovery
chatscope-advanced conversations.json \
  --topics \
  --topic-method BERTopic \
  --num-topics 15 \
  --create-visualizations
```

#### Sentiment-Focused Analysis
```bash
# Focus on sentiment and emotions
chatscope-advanced conversations.json \
  --sentiment \
  --create-visualizations \
  --output-dir sentiment_analysis
```

## Practical Examples

### Example 1: Personal Productivity Analysis

```python
from chatscope import AdvancedChatGPTAnalyzer
import json

# Initialize analyzer
analyzer = AdvancedChatGPTAnalyzer()

# Custom categories for productivity tracking
productivity_categories = [
    "Work Tasks",
    "Learning & Development", 
    "Problem Solving",
    "Creative Projects",
    "Planning & Organization",
    "Research",
    "Personal Growth"
]

analyzer.categories = productivity_categories

# Run comprehensive analysis
results = analyzer.comprehensive_analysis(
    'conversations.json',
    include_sentiment=True,
    include_topics=True,
    include_temporal=True
)

# Analyze productivity patterns
print("=== PRODUCTIVITY ANALYSIS ===")
print(f"Total conversations: {results['metadata']['total_conversations']}")

# Category breakdown
category_counts = results['basic_categorization']['category_counts']
print("\nCategory Distribution:")
for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / sum(category_counts.values())) * 100
    print(f"  {category}: {count} ({percentage:.1f}%)")

# Sentiment insights
if 'sentiment_analysis' in results:
    sentiment = results['sentiment_analysis']
    print(f"\nOverall Sentiment: {sentiment['average_polarity']:.2f}")
    print(f"Most positive category: {max(sentiment['category_sentiment'].items(), key=lambda x: x[1])[0]}")

# Temporal patterns
if 'temporal_analysis' in results:
    temporal = results['temporal_analysis']
    print(f"\nPeak productivity hour: {temporal['peak_hour']}:00")
    print(f"Most active day: {temporal['peak_day']}")
```

### Example 2: Learning Progress Tracking

```python
# Track learning conversations
learning_analyzer = AdvancedChatGPTAnalyzer()

# Focus on learning-related topics
learning_results = learning_analyzer.extract_topics(
    conversations,
    num_topics=10,
    method='BERTopic'
)

# Identify learning topics
learning_topics = []
for topic in learning_results['topics']:
    # Look for learning-related keywords
    learning_keywords = ['learn', 'study', 'understand', 'explain', 'tutorial', 'guide']
    if any(keyword in ' '.join(topic['words']).lower() for keyword in learning_keywords):
        learning_topics.append(topic)

print(f"Identified {len(learning_topics)} learning-focused topics:")
for topic in learning_topics:
    print(f"  - {', '.join(topic['words'][:5])}")

# Track learning sentiment over time
sentiment_results = learning_analyzer.analyze_sentiment(conversations)
learning_sentiment = {
    title: score for title, score in sentiment_results['sentiments'].items()
    if any(keyword in title.lower() for keyword in ['learn', 'study', 'how to', 'explain'])
}

avg_learning_sentiment = sum(s['polarity'] for s in learning_sentiment.values()) / len(learning_sentiment)
print(f"\nAverage learning sentiment: {avg_learning_sentiment:.2f}")
```

### Example 3: Content Creation Analysis

```python
# Analyze conversations for content creation insights
content_analyzer = AdvancedChatGPTAnalyzer()

# Custom categories for content creation
content_categories = [
    "Blog Writing",
    "Social Media",
    "Video Scripts",
    "Marketing Copy",
    "Technical Documentation",
    "Creative Writing",
    "SEO Content"
]

content_analyzer.categories = content_categories

# Run analysis
results = content_analyzer.comprehensive_analysis(
    'conversations.json',
    include_sentiment=True,
    include_topics=True
)

# Extract content themes
topic_results = results['topic_analysis']
content_themes = {}
for topic in topic_results['topics']:
    theme_words = topic['words'][:3]
    content_themes[f"Theme_{topic['topic_id']}"] = theme_words

print("Content Themes Discovered:")
for theme, words in content_themes.items():
    print(f"  {theme}: {', '.join(words)}")

# Analyze content sentiment
sentiment = results['sentiment_analysis']
print(f"\nContent Creation Sentiment: {sentiment['average_polarity']:.2f}")
print(f"Positive content ratio: {sentiment['sentiment_distribution']['positive']}%")
```

### Example 4: Research and Development Tracking

```python
# Track R&D conversations
rd_analyzer = AdvancedChatGPTAnalyzer(debug_mode=True)

# Filter for research-related conversations
conversations = rd_analyzer.load_conversations('conversations.json')
research_conversations = [
    conv for conv in conversations
    if any(keyword in conv.get('title', '').lower() 
           for keyword in ['research', 'analysis', 'study', 'investigate', 'explore'])
]

print(f"Found {len(research_conversations)} research conversations")

# Analyze research topics
research_topics = rd_analyzer.extract_topics(
    research_conversations,
    num_topics=8,
    method='LDA'
)

# Temporal analysis of research activity
research_temporal = rd_analyzer.analyze_temporal_patterns(
    research_conversations,
    time_granularity='weekly'
)

print(f"Research peak time: {research_temporal['peak_hour']}:00")
print(f"Research trend: {research_temporal['trend']}")

# Save research-specific results
research_results = {
    'research_conversations': len(research_conversations),
    'topics': research_topics,
    'temporal_patterns': research_temporal
}

with open('research_analysis.json', 'w') as f:
    json.dump(research_results, f, indent=2, default=str)
```

## Best Practices

### Data Preparation

1. **Clean Your Data**
   ```python
   # Remove incomplete conversations
   conversations = [
       conv for conv in conversations 
       if conv.get('title') and len(conv.get('mapping', {})) > 1
   ]
   ```

2. **Handle Large Datasets**
   ```python
   # Process in chunks for large datasets
   def process_in_chunks(conversations, chunk_size=100):
       for i in range(0, len(conversations), chunk_size):
           chunk = conversations[i:i + chunk_size]
           yield analyzer.analyze_sentiment(chunk)
   ```

3. **Validate Input Data**
   ```python
   # Check data quality before analysis
   def validate_conversations(conversations):
       valid_conversations = []
       for conv in conversations:
           if 'title' in conv and 'mapping' in conv:
               if len(conv['mapping']) > 0:
                   valid_conversations.append(conv)
       return valid_conversations
   ```

### Performance Optimization

1. **Optimize API Usage**
   ```python
   # Adjust batch size based on your data
   analyzer = AdvancedChatGPTAnalyzer(
       batch_size=10,  # Smaller for large datasets
       delay_between_requests=1.5,  # Avoid rate limits
       max_retries=5
   )
   ```

2. **Cache Results**
   ```python
   import pickle
   import os
   
   # Cache expensive operations
   cache_file = 'analysis_cache.pkl'
   if os.path.exists(cache_file):
       with open(cache_file, 'rb') as f:
           results = pickle.load(f)
   else:
       results = analyzer.comprehensive_analysis('conversations.json')
       with open(cache_file, 'wb') as f:
           pickle.dump(results, f)
   ```

3. **Memory Management**
   ```python
   # Clear large objects when done
   del conversations
   import gc
   gc.collect()
   ```

### Analysis Strategy

1. **Start Simple**
   - Begin with basic categorization
   - Add advanced features incrementally
   - Validate results at each step

2. **Customize for Your Use Case**
   ```python
   # Define domain-specific categories
   if domain == 'academic':
       categories = ['Research', 'Teaching', 'Writing', 'Administration']
   elif domain == 'business':
       categories = ['Strategy', 'Operations', 'Marketing', 'Finance']
   ```

3. **Iterate and Refine**
   ```python
   # Refine categories based on initial results
   initial_results = analyzer.analyze('conversations.json')
   
   # Review uncategorized items
   other_conversations = initial_results['categories']['Other']
   
   # Add new categories and re-run
   refined_categories = original_categories + ['New Category 1', 'New Category 2']
   analyzer.categories = refined_categories
   ```

### Result Interpretation

1. **Understand Confidence Levels**
   ```python
   # Check categorization confidence
   for title, category in results['categories'].items():
       if category == 'Other':
           print(f"Low confidence: {title}")
   ```

2. **Cross-Reference Results**
   ```python
   # Compare sentiment with topics
   positive_topics = []
   for topic_id, sentiment in topic_sentiments.items():
       if sentiment > 0.3:
           positive_topics.append(topic_id)
   ```

3. **Validate with Domain Knowledge**
   - Review sample categorizations manually
   - Check if topics make sense for your domain
   - Verify temporal patterns match your expectations

## Troubleshooting

### Common Issues

#### 1. API Key Problems
```python
# Test API key
try:
    analyzer = AdvancedChatGPTAnalyzer(api_key="your-key")
    # Try a small test
    test_result = analyzer.categorize_all_titles(["Test conversation"])
    print("API key is working!")
except Exception as e:
    print(f"API key issue: {e}")
```

#### 2. Data Format Issues
```python
# Debug data loading
try:
    conversations = analyzer.load_conversations('conversations.json')
    print(f"Loaded {len(conversations)} conversations")
    
    # Check first conversation structure
    if conversations:
        first_conv = conversations[0]
        print(f"Sample conversation keys: {list(first_conv.keys())}")
except Exception as e:
    print(f"Data loading error: {e}")
```

#### 3. Memory Issues
```python
# Monitor memory usage
import psutil
import os

def check_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

check_memory()
results = analyzer.analyze('conversations.json')
check_memory()
```

#### 4. Dependency Issues
```python
# Check optional dependencies
try:
    import sklearn
    print("✓ scikit-learn available")
except ImportError:
    print("✗ scikit-learn not installed - topic modeling unavailable")

try:
    import textblob
    print("✓ TextBlob available")
except ImportError:
    print("✗ TextBlob not installed - sentiment analysis unavailable")
```

### Error Recovery

```python
# Robust analysis with error handling
def robust_analysis(file_path):
    try:
        # Try comprehensive analysis first
        return analyzer.comprehensive_analysis(file_path)
    except APIError:
        print("API error - trying basic analysis only")
        return analyzer.analyze(file_path)
    except DataError as e:
        print(f"Data error: {e}")
        # Try to fix common data issues
        conversations = fix_conversation_data(file_path)
        return analyzer.analyze_conversations(conversations)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def fix_conversation_data(file_path):
    # Implement data cleaning logic
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Remove malformed conversations
    cleaned_data = [
        conv for conv in data 
        if isinstance(conv, dict) and 'title' in conv
    ]
    
    return cleaned_data
```

## Tips and Tricks

### Advanced Usage Patterns

#### 1. Batch Processing Multiple Files
```python
import glob
import os

# Process multiple conversation files
conversation_files = glob.glob('conversations_*.json')
all_results = {}

for file_path in conversation_files:
    file_name = os.path.basename(file_path)
    print(f"Processing {file_name}...")
    
    results = analyzer.comprehensive_analysis(file_path)
    all_results[file_name] = results

# Combine results
combined_stats = {
    'total_files': len(all_results),
    'total_conversations': sum(r['metadata']['total_conversations'] for r in all_results.values()),
    'average_sentiment': sum(r['sentiment_analysis']['average_polarity'] for r in all_results.values()) / len(all_results)
}
```

#### 2. Custom Analysis Pipelines
```python
class CustomAnalysisPipeline:
    def __init__(self):
        self.analyzer = AdvancedChatGPTAnalyzer()
        self.results = {}
    
    def run_pipeline(self, conversations):
        # Step 1: Basic categorization
        self.results['categories'] = self.analyzer.categorize_all_titles(
            self.analyzer.extract_unique_titles(conversations)
        )
        
        # Step 2: Sentiment analysis on specific categories
        work_conversations = self.filter_by_category(conversations, 'Work')
        self.results['work_sentiment'] = self.analyzer.analyze_sentiment(work_conversations)
        
        # Step 3: Topic modeling on learning conversations
        learning_conversations = self.filter_by_category(conversations, 'Learning')
        self.results['learning_topics'] = self.analyzer.extract_topics(learning_conversations)
        
        return self.results
    
    def filter_by_category(self, conversations, category):
        # Filter conversations by category
        category_titles = [
            title for title, cat in self.results['categories'].items()
            if cat == category
        ]
        return [
            conv for conv in conversations
            if conv.get('title') in category_titles
        ]
```

#### 3. Real-time Analysis Updates
```python
import time
from datetime import datetime

class RealTimeAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.analyzer = AdvancedChatGPTAnalyzer()
        self.last_modified = 0
        self.results = None
    
    def check_for_updates(self):
        current_modified = os.path.getmtime(self.file_path)
        if current_modified > self.last_modified:
            print(f"File updated at {datetime.now()}")
            self.results = self.analyzer.comprehensive_analysis(self.file_path)
            self.last_modified = current_modified
            return True
        return False
    
    def monitor(self, interval=300):  # Check every 5 minutes
        while True:
            if self.check_for_updates():
                self.generate_report()
            time.sleep(interval)
    
    def generate_report(self):
        if self.results:
            print(f"\n=== Analysis Report - {datetime.now()} ===")
            print(f"Total conversations: {self.results['metadata']['total_conversations']}")
            print(f"Average sentiment: {self.results['sentiment_analysis']['average_polarity']:.2f}")
```

#### 4. Export to Different Formats
```python
import pandas as pd
import csv

def export_results(results, format='json'):
    if format == 'csv':
        # Export category counts to CSV
        df = pd.DataFrame(list(results['category_counts'].items()), 
                         columns=['Category', 'Count'])
        df.to_csv('category_analysis.csv', index=False)
        
    elif format == 'excel':
        # Export comprehensive results to Excel
        with pd.ExcelWriter('analysis_results.xlsx') as writer:
            # Categories sheet
            cat_df = pd.DataFrame(list(results['categories'].items()),
                                columns=['Title', 'Category'])
            cat_df.to_excel(writer, sheet_name='Categories', index=False)
            
            # Sentiment sheet
            if 'sentiment_analysis' in results:
                sent_df = pd.DataFrame(results['sentiment_analysis']['sentiments']).T
                sent_df.to_excel(writer, sheet_name='Sentiment')
    
    elif format == 'html':
        # Generate HTML report
        html_content = generate_html_report(results)
        with open('analysis_report.html', 'w') as f:
            f.write(html_content)

def generate_html_report(results):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ChatScope Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .category {{ margin: 5px 0; }}
        </style>
    </head>
    <body>
        <h1>ChatScope Analysis Report</h1>
        <div class="metric">
            <h3>Overview</h3>
            <p>Total Conversations: {results['metadata']['total_conversations']}</p>
            <p>Analysis Date: {results['metadata']['timestamp']}</p>
        </div>
        
        <div class="metric">
            <h3>Category Distribution</h3>
    """
    
    for category, count in results['category_counts'].items():
        percentage = (count / results['metadata']['total_conversations']) * 100
        html += f'<div class="category">{category}: {count} ({percentage:.1f}%)</div>'
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html
```

### Performance Monitoring

```python
import time
from functools import wraps

def performance_monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"{func.__name__}:")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Memory: {end_memory - start_memory:.1f}MB")
        
        return result
    return wrapper

# Apply to analysis methods
AdvancedChatGPTAnalyzer.comprehensive_analysis = performance_monitor(
    AdvancedChatGPTAnalyzer.comprehensive_analysis
)
```

### Integration Examples

#### Jupyter Notebook Integration
```python
# In Jupyter notebook
%matplotlib inline
import matplotlib.pyplot as plt

# Run analysis
results = analyzer.comprehensive_analysis('conversations.json')

# Create inline visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Category distribution
axes[0,0].bar(results['category_counts'].keys(), results['category_counts'].values())
axes[0,0].set_title('Category Distribution')
axes[0,0].tick_params(axis='x', rotation=45)

# Sentiment over time
if 'temporal_analysis' in results:
    hourly = results['temporal_analysis']['hourly_pattern']
    hours = list(range(24))
    counts = [hourly.get(str(h), 0) for h in hours]
    axes[0,1].plot(hours, counts)
    axes[0,1].set_title('Activity by Hour')

plt.tight_layout()
plt.show()
```

#### Web Dashboard Integration
```python
# Flask web app example
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
analyzer = AdvancedChatGPTAnalyzer()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['conversations']
    file.save('temp_conversations.json')
    
    results = analyzer.comprehensive_analysis('temp_conversations.json')
    
    return jsonify({
        'total_conversations': results['metadata']['total_conversations'],
        'categories': results['category_counts'],
        'sentiment': results['sentiment_analysis']['average_polarity']
    })

if __name__ == '__main__':
    app.run(debug=True)
```

This user guide provides comprehensive coverage of ChatScope's capabilities with practical examples and best practices. Use it as a reference to get the most out of your conversation analysis projects!