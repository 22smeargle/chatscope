# ChatScope 2.0 🔍

**Advanced Python library for comprehensive ChatGPT conversation analysis with AI-powered insights**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/22smeargle/chatscope)

ChatScope is a powerful Python library that transforms your ChatGPT conversation exports into actionable insights through advanced analytics, sentiment analysis, topic modeling, and temporal pattern detection.

## 🚀 What's New in Version 2.0

- **🧠 Advanced Sentiment Analysis** - Emotion detection and polarity analysis
- **📊 Topic Modeling** - LDA, BERTopic, and clustering-based topic extraction
- **⏰ Temporal Analysis** - Time-based patterns and trend detection
- **📈 Interactive Visualizations** - Plotly, Seaborn, and WordCloud support
- **🔧 Enhanced CLI** - Advanced command-line interface with comprehensive options
- **🎯 Comprehensive Analysis** - All-in-one analysis pipeline
- **📱 Better Error Handling** - Robust error management and logging

## ✨ Key Features

### Core Analytics
- **Automatic Categorization** - AI-powered conversation classification
- **Sentiment Analysis** - Emotion detection and sentiment scoring
- **Topic Modeling** - Multiple algorithms (LDA, BERTopic, K-Means clustering)
- **Temporal Analysis** - Time-based patterns and trends
- **Visual Analytics** - Rich charts, graphs, and word clouds

### Advanced Features
- **Batch Processing** - Handle large conversation datasets efficiently
- **Custom Categories** - Define your own classification schemes
- **Rate Limiting** - Respect OpenAI API limits automatically
- **Secure API Handling** - Environment variable support for API keys
- **Multiple Output Formats** - JSON, CSV, and visual exports
- **Interactive Dashboards** - Plotly-powered interactive visualizations

### Technical Excellence
- **Comprehensive Testing** - Full test suite with 95%+ coverage
- **Type Hints** - Full type annotation support
- **Logging** - Detailed logging for debugging and monitoring
- **Error Recovery** - Graceful handling of API failures and data issues
- **Modular Design** - Extensible architecture for custom analysis

## 📦 Installation

### Basic Installation
```bash
pip install chatscope
```

### Full Installation (with all features)
```bash
pip install chatscope[all]
```

### Development Installation
```bash
git clone https://github.com/22smeargle/chatscope.git
cd chatscope
pip install -e .[dev]
```

### Optional Dependencies

**For Advanced Analytics:**
```bash
pip install scikit-learn textblob nltk pandas numpy
```

**For Enhanced Visualizations:**
```bash
pip install matplotlib seaborn plotly wordcloud
```

**For Advanced Topic Modeling:**
```bash
pip install bertopic transformers torch
```

## 🔧 Quick Start

### Basic Usage

```python
from chatscope import ChatGPTAnalyzer
import os

# Set up your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Initialize analyzer
analyzer = ChatGPTAnalyzer()

# Analyze conversations
results = analyzer.analyze('conversations.json')

# View results
print(f"Total conversations: {results['total_conversations']}")
print(f"Categories found: {list(results['categories'].keys())}")
```

### Advanced Analysis

```python
from chatscope import AdvancedChatGPTAnalyzer

# Initialize advanced analyzer
analyzer = AdvancedChatGPTAnalyzer(
    api_key='your-api-key',
    batch_size=15,
    delay_between_requests=1.5
)

# Comprehensive analysis with all features
results = analyzer.comprehensive_analysis(
    input_file='conversations.json',
    include_sentiment=True,
    include_topics=True,
    include_temporal=True,
    create_visualizations=True,
    topic_method='BERTopic',
    num_topics=10
)

print(f"Analysis complete! Results saved to: {results['output_files']}")
```

### Command Line Interface

**Basic Analysis:**
```bash
chatscope conversations.json
```

**Advanced Analysis:**
```bash
chatscope-advanced conversations.json \
    --all-features \
    --output-dir ./analysis_results \
    --topic-method BERTopic \
    --num-topics 15 \
    --create-visualizations
```

**Sentiment-Only Analysis:**
```bash
chatscope-advanced conversations.json --sentiment-only
```

## 📊 Analysis Features

### 1. Sentiment Analysis

```python
# Analyze sentiment across conversations
sentiment_results = analyzer.analyze_sentiment(conversations)

print(f"Average sentiment: {sentiment_results['average_polarity']:.2f}")
print(f"Emotion distribution: {sentiment_results['emotion_distribution']}")
```

**Features:**
- Polarity scoring (-1 to 1)
- Subjectivity analysis (0 to 1)
- Emotion detection (joy, sadness, anger, fear, neutral)
- Sentiment distribution visualization

### 2. Topic Modeling

```python
# Extract topics using different methods
topics_lda = analyzer.extract_topics(conversations, method='LDA', num_topics=8)
topics_bert = analyzer.extract_topics(conversations, method='BERTopic', num_topics=10)
topics_cluster = analyzer.extract_topics(conversations, method='clustering', num_topics=6)

# View discovered topics
for topic in topics_lda['topics']:
    print(f"Topic {topic['topic_id']}: {', '.join(topic['words'][:5])}")
```

**Available Methods:**
- **LDA (Latent Dirichlet Allocation)** - Statistical topic modeling
- **BERTopic** - Transformer-based topic modeling
- **K-Means Clustering** - Vector space clustering

### 3. Temporal Analysis

```python
# Analyze conversation patterns over time
temporal_results = analyzer.analyze_temporal_patterns(
    conversations,
    time_granularity='daily',
    detect_trends=True
)

print(f"Peak activity hour: {temporal_results['peak_hour']}")
print(f"Most active day: {temporal_results['most_active_day']}")
print(f"Trend: {temporal_results['trend']}")
```

**Features:**
- Hourly, daily, weekly, monthly patterns
- Trend detection and correlation analysis
- Activity heatmaps and time series plots
- Peak activity identification

### 4. Advanced Visualizations

```python
# Create comprehensive visualizations
viz_paths = analyzer.create_advanced_visualizations(
    analysis_results,
    output_dir='./visualizations'
)

print(f"Visualizations created: {list(viz_paths.keys())}")
```

**Visualization Types:**
- Sentiment distribution charts
- Topic modeling plots
- Temporal heatmaps
- Word clouds
- Interactive dashboards
- Correlation matrices

## 🎯 Advanced Usage Examples

### Custom Categories

```python
custom_categories = [
    "Technical Programming",
    "Creative Writing",
    "Business Strategy",
    "Personal Development",
    "Academic Research"
]

analyzer = AdvancedChatGPTAnalyzer(
    categories=custom_categories,
    api_key='your-api-key'
)

results = analyzer.analyze('conversations.json')
```

### Batch Processing with Custom Logic

```python
import glob
from pathlib import Path

# Process multiple conversation files
conversation_files = glob.glob('data/*.json')
all_results = []

for file_path in conversation_files:
    print(f"Processing {file_path}...")
    
    results = analyzer.comprehensive_analysis(
        input_file=file_path,
        include_sentiment=True,
        include_topics=True,
        include_temporal=True
    )
    
    # Add metadata
    results['source_file'] = Path(file_path).name
    results['processing_timestamp'] = datetime.now().isoformat()
    
    all_results.append(results)

# Combine and save results
combined_results = {
    'total_files_processed': len(all_results),
    'individual_results': all_results,
    'summary_statistics': calculate_summary_stats(all_results)
}

with open('batch_analysis_results.json', 'w') as f:
    json.dump(combined_results, f, indent=2)
```

### Integration with Data Analysis Workflows

```python
import pandas as pd
import matplotlib.pyplot as plt

# Convert results to DataFrame for analysis
def results_to_dataframe(results):
    conversations_data = []
    
    for title, category in results['categories'].items():
        sentiment = results.get('sentiment_analysis', {}).get('sentiments', {})
        
        conversations_data.append({
            'title': title,
            'category': category,
            'sentiment_polarity': sentiment.get(title, {}).get('polarity', 0),
            'sentiment_subjectivity': sentiment.get(title, {}).get('subjectivity', 0)
        })
    
    return pd.DataFrame(conversations_data)

# Analyze results
df = results_to_dataframe(results)

# Category-wise sentiment analysis
category_sentiment = df.groupby('category')['sentiment_polarity'].agg([
    'mean', 'std', 'count'
]).round(3)

print("Category-wise Sentiment Analysis:")
print(category_sentiment)

# Create visualization
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df.boxplot(column='sentiment_polarity', by='category', ax=plt.gca())
plt.title('Sentiment Distribution by Category')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
category_counts = df['category'].value_counts()
category_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Conversation Distribution by Category')

plt.tight_layout()
plt.savefig('analysis_summary.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Optional
export CHATSCOPE_LOG_LEVEL="INFO"
export CHATSCOPE_CACHE_DIR="./cache"
export CHATSCOPE_OUTPUT_DIR="./results"
```

### Configuration File

Create `chatscope_config.json`:

```json
{
    "api_settings": {
        "batch_size": 20,
        "delay_between_requests": 1.0,
        "max_retries": 3,
        "timeout": 30
    },
    "analysis_settings": {
        "default_categories": [
            "Programming",
            "Artificial Intelligence",
            "Health",
            "Education",
            "Work / Career",
            "Philosophy",
            "Other"
        ],
        "sentiment_threshold": 0.1,
        "topic_modeling_method": "LDA",
        "num_topics": 10
    },
    "output_settings": {
        "create_visualizations": true,
        "save_intermediate_results": true,
        "output_format": "json"
    }
}
```

## 📈 API Reference

### ChatGPTAnalyzer (Basic)

```python
class ChatGPTAnalyzer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        categories: Optional[List[str]] = None,
        batch_size: int = 20,
        delay_between_requests: float = 1.0
    )
    
    def load_conversations(self, file_path: str) -> List[Dict[str, Any]]
    def extract_unique_titles(self, conversations: List[Dict]) -> List[str]
    def categorize_all_titles(self, titles: List[str]) -> Dict[str, str]
    def create_category_dictionary(self, titles: List[str], categories: Dict[str, str]) -> Dict[str, List[str]]
    def count_conversations_by_category(self, category_dict: Dict[str, List[str]]) -> Dict[str, int]
    def create_bar_chart(self, category_counts: Dict[str, int], output_path: str = "conversation_categories.png")
    def save_results(self, results: Dict[str, Any], output_path: str = "analysis_results.json")
    def analyze(self, input_file: str, output_dir: str = ".") -> Dict[str, Any]
```

### AdvancedChatGPTAnalyzer

```python
class AdvancedChatGPTAnalyzer(ChatGPTAnalyzer):
    def analyze_sentiment(
        self,
        conversations: List[Dict[str, Any]],
        include_emotional_tone: bool = True
    ) -> Dict[str, Any]
    
    def extract_topics(
        self,
        conversations: List[Dict[str, Any]],
        num_topics: int = 10,
        method: str = 'LDA'
    ) -> Dict[str, Any]
    
    def analyze_temporal_patterns(
        self,
        conversations: List[Dict[str, Any]],
        time_granularity: str = 'daily',
        detect_trends: bool = True
    ) -> Dict[str, Any]
    
    def create_advanced_visualizations(
        self,
        analysis_results: Dict[str, Any],
        output_dir: str = "./visualizations"
    ) -> Dict[str, str]
    
    def comprehensive_analysis(
        self,
        input_file: str,
        include_sentiment: bool = True,
        include_topics: bool = True,
        include_temporal: bool = True,
        create_visualizations: bool = True,
        topic_method: str = 'LDA',
        num_topics: int = 10,
        time_granularity: str = 'daily'
    ) -> Dict[str, Any]
```

### Exception Classes

```python
class APIError(Exception):
    """Raised when OpenAI API calls fail."""
    pass

class DataError(Exception):
    """Raised when conversation data is invalid or corrupted."""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass
```

## 🛠️ Best Practices

### API Key Management

```python
# ✅ Good: Use environment variables
import os
api_key = os.getenv('OPENAI_API_KEY')

# ✅ Good: Use .env files
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# ❌ Bad: Hardcode API keys
api_key = "sk-your-actual-key"  # Never do this!
```

### Rate Limiting

```python
# Configure appropriate rate limiting
analyzer = AdvancedChatGPTAnalyzer(
    batch_size=10,  # Smaller batches for rate limiting
    delay_between_requests=2.0,  # 2-second delay between requests
    max_retries=3  # Retry failed requests
)
```

### Performance Optimization

```python
# For large datasets
analyzer = AdvancedChatGPTAnalyzer(
    batch_size=25,  # Larger batches for efficiency
    delay_between_requests=0.5,  # Faster processing
)

# Use caching for repeated analysis
results = analyzer.comprehensive_analysis(
    input_file='large_dataset.json',
    cache_results=True,  # Cache intermediate results
    skip_if_exists=True  # Skip if results already exist
)
```

### Data Quality

```python
# Validate data before analysis
def validate_conversations(conversations):
    valid_conversations = []
    
    for conv in conversations:
        if (
            'title' in conv and 
            conv['title'] and 
            len(conv['title'].strip()) > 0
        ):
            valid_conversations.append(conv)
        else:
            print(f"Skipping invalid conversation: {conv.get('id', 'unknown')}")
    
    return valid_conversations

conversations = analyzer.load_conversations('data.json')
valid_conversations = validate_conversations(conversations)
results = analyzer.analyze_sentiment(valid_conversations)
```

### Logging and Debugging

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatscope.log'),
        logging.StreamHandler()
    ]
)

# Use debug mode for troubleshooting
analyzer = AdvancedChatGPTAnalyzer(
    api_key='your-key',
    debug_mode=True,  # Enable debug logging
    save_intermediate_results=True  # Save intermediate results
)
```

## 🔗 Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify, render_template
from chatscope import AdvancedChatGPTAnalyzer
import os

app = Flask(__name__)
analyzer = AdvancedChatGPTAnalyzer(api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze_conversations():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    file.save(file_path)
    
    try:
        # Analyze conversations
        results = analyzer.comprehensive_analysis(
            input_file=file_path,
            include_sentiment=True,
            include_topics=True,
            include_temporal=True
        )
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_conversations': results['metadata']['total_conversations'],
                'categories': list(results['basic_categorization']['categories'].keys()),
                'average_sentiment': results.get('sentiment_analysis', {}).get('average_polarity', 0)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
```

### Jupyter Notebook Integration

```python
# Cell 1: Setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from chatscope import AdvancedChatGPTAnalyzer
import os

# Configure visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Cell 2: Load and analyze data
analyzer = AdvancedChatGPTAnalyzer(api_key=os.getenv('OPENAI_API_KEY'))

results = analyzer.comprehensive_analysis(
    input_file='conversations.json',
    include_sentiment=True,
    include_topics=True,
    include_temporal=True
)

# Cell 3: Create interactive visualizations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sentiment over time
if 'temporal_analysis' in results and 'sentiment_analysis' in results:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sentiment Distribution', 'Topics Over Time', 
                       'Hourly Patterns', 'Category Distribution'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Add sentiment distribution
    sentiment_dist = results['sentiment_analysis']['sentiment_distribution']
    fig.add_trace(
        go.Bar(x=list(sentiment_dist.keys()), y=list(sentiment_dist.values())),
        row=1, col=1
    )
    
    # Add hourly patterns
    hourly_pattern = results['temporal_analysis']['hourly_pattern']
    fig.add_trace(
        go.Bar(x=list(hourly_pattern.keys()), y=list(hourly_pattern.values())),
        row=2, col=1
    )
    
    # Add category distribution
    categories = results['basic_categorization']['category_counts']
    fig.add_trace(
        go.Pie(labels=list(categories.keys()), values=list(categories.values())),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="ChatGPT Conversation Analysis Dashboard")
    fig.show()

# Cell 4: Export results
results_df = pd.DataFrame([
    {
        'title': title,
        'category': category,
        'sentiment': results.get('sentiment_analysis', {}).get('sentiments', {}).get(title, {}).get('polarity', 0)
    }
    for title, category in results['basic_categorization']['categories'].items()
])

results_df.to_csv('conversation_analysis.csv', index=False)
print(f"Results exported to conversation_analysis.csv ({len(results_df)} conversations)")
```

### Automated Reporting Script

```python
#!/usr/bin/env python3
"""
Automated ChatGPT Conversation Analysis Report Generator

Usage: python generate_report.py --input conversations.json --output report.html
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from jinja2 import Template
from chatscope import AdvancedChatGPTAnalyzer
import os

def generate_html_report(results, output_path):
    """Generate an HTML report from analysis results."""
    
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ChatGPT Conversation Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
            .section { margin: 20px 0; }
            .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #e8f4f8; border-radius: 5px; }
            .chart { margin: 20px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>ChatGPT Conversation Analysis Report</h1>
            <p>Generated on: {{ timestamp }}</p>
            <p>Total Conversations Analyzed: {{ total_conversations }}</p>
        </div>
        
        <div class="section">
            <h2>Summary Metrics</h2>
            <div class="metric">
                <h3>{{ total_conversations }}</h3>
                <p>Total Conversations</p>
            </div>
            <div class="metric">
                <h3>{{ num_categories }}</h3>
                <p>Categories Found</p>
            </div>
            {% if average_sentiment %}
            <div class="metric">
                <h3>{{ "%.2f"|format(average_sentiment) }}</h3>
                <p>Average Sentiment</p>
            </div>
            {% endif %}
        </div>
        
        <div class="section">
            <h2>Category Distribution</h2>
            <table>
                <tr><th>Category</th><th>Count</th><th>Percentage</th></tr>
                {% for category, count in categories.items() %}
                <tr>
                    <td>{{ category }}</td>
                    <td>{{ count }}</td>
                    <td>{{ "%.1f"|format((count / total_conversations) * 100) }}%</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        {% if sentiment_analysis %}
        <div class="section">
            <h2>Sentiment Analysis</h2>
            <table>
                <tr><th>Sentiment</th><th>Count</th><th>Percentage</th></tr>
                {% for sentiment, count in sentiment_analysis.sentiment_distribution.items() %}
                <tr>
                    <td>{{ sentiment.title() }}</td>
                    <td>{{ count }}</td>
                    <td>{{ "%.1f"|format((count / sentiment_analysis.total_analyzed) * 100) }}%</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        
        {% if topics %}
        <div class="section">
            <h2>Discovered Topics</h2>
            <table>
                <tr><th>Topic ID</th><th>Top Words</th></tr>
                {% for topic in topics.topics[:5] %}
                <tr>
                    <td>{{ topic.topic_id }}</td>
                    <td>{{ ", ".join(topic.words[:8]) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        
        <div class="section">
            <h2>Analysis Details</h2>
            <p><strong>Analysis Method:</strong> ChatScope 2.0 Advanced Analysis</p>
            <p><strong>Features Used:</strong> 
                {% if sentiment_analysis %}Sentiment Analysis, {% endif %}
                {% if topics %}Topic Modeling, {% endif %}
                {% if temporal_analysis %}Temporal Analysis, {% endif %}
                Basic Categorization
            </p>
        </div>
    </body>
    </html>
    """
    
    template = Template(template_str)
    
    # Prepare template variables
    template_vars = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_conversations': results['metadata']['total_conversations'],
        'num_categories': len(results['basic_categorization']['categories']),
        'categories': results['basic_categorization']['category_counts'],
        'sentiment_analysis': results.get('sentiment_analysis'),
        'topics': results.get('topic_analysis'),
        'temporal_analysis': results.get('temporal_analysis'),
        'average_sentiment': results.get('sentiment_analysis', {}).get('average_polarity')
    }
    
    # Generate HTML
    html_content = template.render(**template_vars)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate ChatGPT conversation analysis report')
    parser.add_argument('--input', required=True, help='Input JSON file with conversations')
    parser.add_argument('--output', default='report.html', help='Output HTML report file')
    parser.add_argument('--include-sentiment', action='store_true', help='Include sentiment analysis')
    parser.add_argument('--include-topics', action='store_true', help='Include topic modeling')
    parser.add_argument('--include-temporal', action='store_true', help='Include temporal analysis')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AdvancedChatGPTAnalyzer(api_key=os.getenv('OPENAI_API_KEY'))
    
    print(f"Analyzing conversations from {args.input}...")
    
    # Run analysis
    results = analyzer.comprehensive_analysis(
        input_file=args.input,
        include_sentiment=args.include_sentiment,
        include_topics=args.include_topics,
        include_temporal=args.include_temporal,
        create_visualizations=False
    )
    
    # Generate report
    generate_html_report(results, args.output)
    
    print(f"Analysis complete! Report saved to {args.output}")

if __name__ == '__main__':
    main()
```

## 🐛 Troubleshooting

### Common Issues

**1. API Key Issues**
```bash
# Error: "No API key provided"
# Solution: Set environment variable
export OPENAI_API_KEY="your-api-key-here"

# Or use .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

**2. Import Errors**
```python
# Error: "No module named 'chatscope'"
# Solution: Install the package
pip install chatscope

# Error: "No module named 'sklearn'"
# Solution: Install optional dependencies
pip install scikit-learn
```

**3. Data Format Issues**
```python
# Error: "Invalid conversation format"
# Solution: Validate your JSON structure
import json

with open('conversations.json', 'r') as f:
    data = json.load(f)
    
# Check if it's a list of conversations
if not isinstance(data, list):
    print("Error: JSON should contain a list of conversations")
    
# Check required fields
for i, conv in enumerate(data[:5]):  # Check first 5
    if 'title' not in conv:
        print(f"Conversation {i} missing 'title' field")
    if 'mapping' not in conv:
        print(f"Conversation {i} missing 'mapping' field")
```

**4. API Rate Limiting**
```python
# Error: "Rate limit exceeded"
# Solution: Increase delays and reduce batch size
analyzer = AdvancedChatGPTAnalyzer(
    batch_size=5,  # Smaller batches
    delay_between_requests=3.0,  # Longer delays
    max_retries=5  # More retries
)
```

**5. Memory Issues**
```python
# Error: "Memory error" with large datasets
# Solution: Process in chunks
def process_large_dataset(file_path, chunk_size=100):
    conversations = analyzer.load_conversations(file_path)
    
    results = []
    for i in range(0, len(conversations), chunk_size):
        chunk = conversations[i:i + chunk_size]
        chunk_results = analyzer.analyze_sentiment(chunk)
        results.append(chunk_results)
        
        # Clear memory
        del chunk
    
    return results
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug mode
analyzer = AdvancedChatGPTAnalyzer(
    api_key='your-key',
    debug_mode=True
)

# Check dependencies
analyzer._check_dependencies()
```

### Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Monitor analysis performance
@monitor_performance
def analyze_with_monitoring(file_path):
    return analyzer.comprehensive_analysis(file_path)

results = analyze_with_monitoring('conversations.json')
```

## 💰 Cost Optimization Tips

### Batch Size Optimization
```python
# Optimize batch size based on your data
small_dataset_analyzer = AdvancedChatGPTAnalyzer(batch_size=30)  # < 100 conversations
medium_dataset_analyzer = AdvancedChatGPTAnalyzer(batch_size=20)  # 100-500 conversations
large_dataset_analyzer = AdvancedChatGPTAnalyzer(batch_size=10)   # > 500 conversations
```

### Caching Results
```python
import hashlib
import pickle
from pathlib import Path

def cache_results(func):
    def wrapper(self, *args, **kwargs):
        # Create cache key
        cache_key = hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()
        cache_file = Path(f"cache/{func.__name__}_{cache_key}.pkl")
        
        # Check if cached result exists
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Compute result
        result = func(self, *args, **kwargs)
        
        # Cache result
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
    return wrapper

# Apply caching to expensive operations
AdvancedChatGPTAnalyzer.categorize_all_titles = cache_results(AdvancedChatGPTAnalyzer.categorize_all_titles)
```

### Filtering Duplicates
```python
def remove_duplicate_conversations(conversations):
    """Remove conversations with duplicate titles."""
    seen_titles = set()
    unique_conversations = []
    
    for conv in conversations:
        title = conv.get('title', '').strip().lower()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_conversations.append(conv)
    
    print(f"Removed {len(conversations) - len(unique_conversations)} duplicate conversations")
    return unique_conversations

# Use before analysis
conversations = analyzer.load_conversations('data.json')
unique_conversations = remove_duplicate_conversations(conversations)
results = analyzer.analyze_sentiment(unique_conversations)
```

## ❓ Frequently Asked Questions (FAQ)

### General Questions

**Q: What is ChatScope and what does it do?**
A: ChatScope is a Python library that analyzes exported ChatGPT conversations. It provides automatic categorization, sentiment analysis, topic modeling, temporal pattern detection, and comprehensive visualizations to help you understand your conversation patterns and extract insights.

**Q: How do I export my ChatGPT conversations?**
A: Go to ChatGPT → Settings → Data Controls → Export Data. You'll receive a JSON file containing all your conversations.

**Q: Is my data secure when using ChatScope?**
A: Yes! ChatScope processes your data locally on your machine. The only external API call is to OpenAI for categorization, and only conversation titles are sent (not the full content).

**Q: Do I need an OpenAI API key?**
A: Yes, you need an OpenAI API key for the categorization feature. However, you can use many other features (sentiment analysis, topic modeling, temporal analysis) without an API key.

**Q: How much does it cost to analyze my conversations?**
A: Costs depend on the number of conversations and OpenAI's current pricing. Typically, analyzing 1000 conversations costs less than $1. Use batch processing and caching to minimize costs.

### Technical Questions

**Q: What Python versions are supported?**
A: ChatScope supports Python 3.8 and higher. We recommend Python 3.9+ for the best experience.

**Q: Can I use ChatScope without installing all dependencies?**
A: Yes! ChatScope has a modular design. Core features work with minimal dependencies, while advanced features require additional packages (scikit-learn, matplotlib, etc.).

**Q: How do I handle large conversation datasets?**
A: Use batch processing, increase delays between API calls, and consider processing in chunks. For datasets with 1000+ conversations, use `batch_size=10` and `delay_between_requests=2.0`.

**Q: Can I customize the conversation categories?**
A: Absolutely! Pass a custom list of categories when initializing the analyzer:
```python
custom_categories = ["Work", "Personal", "Learning", "Creative"]
analyzer = ChatGPTAnalyzer(categories=custom_categories)
```

**Q: How accurate is the sentiment analysis?**
A: The sentiment analysis uses TextBlob, which provides good accuracy for general text. For domain-specific analysis, consider training custom models or using the emotion detection features.

**Q: Can I export results to different formats?**
A: Yes! Results are saved as JSON by default, but you can easily convert to CSV, Excel, or other formats using pandas:
```python
import pandas as pd
df = pd.DataFrame(results['categories'].items(), columns=['Title', 'Category'])
df.to_csv('results.csv', index=False)
```

### Performance Questions

**Q: Why is the analysis taking so long?**
A: Analysis speed depends on dataset size, API rate limits, and enabled features. To speed up:
- Reduce batch size for better rate limiting
- Disable unnecessary features
- Use caching for repeated analysis
- Process in parallel for multiple files

**Q: How can I monitor analysis progress?**
A: Enable logging to see detailed progress:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

**Q: Can I run ChatScope on a server or in the cloud?**
A: Yes! ChatScope works in any Python environment. For cloud deployment, ensure you have proper API key management and consider using environment variables.

**Q: How do I handle API rate limits?**
A: Configure appropriate delays and batch sizes:
```python
analyzer = AdvancedChatGPTAnalyzer(
    batch_size=5,
    delay_between_requests=3.0,
    max_retries=5
)
```

## 🤝 Contributing

We welcome contributions to ChatScope! Here's how you can help:

### Types of Contributions

- **Bug Reports** - Found a bug? Please report it!
- **Feature Requests** - Have an idea for a new feature?
- **Code Contributions** - Submit pull requests for bug fixes or new features
- **Documentation** - Help improve our documentation
- **Examples** - Share interesting use cases and examples
- **Testing** - Help us improve test coverage

### Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/22smeargle/chatscope.git
cd chatscope
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Development Dependencies**
```bash
pip install -e .[dev]
```

4. **Run Tests**
```bash
python -m pytest tests/ -v
```

### Development Workflow

1. **Create a Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make Changes**
- Write code following our style guidelines
- Add tests for new functionality
- Update documentation as needed

3. **Run Tests and Linting**
```bash
# Run tests
python -m pytest tests/ -v

# Run linting
flake8 chatscope/
black chatscope/
mypy chatscope/
```

4. **Commit and Push**
```bash
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

5. **Create Pull Request**
- Go to GitHub and create a pull request
- Describe your changes and why they're needed
- Link any related issues

### Code Style Guidelines

- **Follow PEP 8** - Use black for formatting
- **Type Hints** - Add type hints to all functions
- **Docstrings** - Use Google-style docstrings
- **Testing** - Maintain 90%+ test coverage
- **Logging** - Use appropriate logging levels

### Example Contribution

```python
def new_analysis_feature(self, conversations: List[Dict[str, Any]], 
                        parameter: str = "default") -> Dict[str, Any]:
    """New analysis feature description.
    
    Args:
        conversations: List of conversation dictionaries.
        parameter: Description of parameter.
        
    Returns:
        Dictionary containing analysis results.
        
    Raises:
        DataError: If conversations data is invalid.
    """
    logger.info(f"Running new analysis feature with parameter: {parameter}")
    
    # Implementation here
    results = {}
    
    return results
```

### Testing Guidelines

```python
import unittest
from unittest.mock import patch, MagicMock
from chatscope import AdvancedChatGPTAnalyzer

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        self.analyzer = AdvancedChatGPTAnalyzer(api_key="test-key")
        self.sample_conversations = [...]  # Test data
    
    def test_new_feature_basic(self):
        """Test basic functionality of new feature."""
        results = self.analyzer.new_analysis_feature(self.sample_conversations)
        
        self.assertIn('expected_key', results)
        self.assertEqual(results['expected_value'], 'expected')
    
    def test_new_feature_error_handling(self):
        """Test error handling in new feature."""
        with self.assertRaises(DataError):
            self.analyzer.new_analysis_feature([])
```

### Documentation Guidelines

- **Clear Examples** - Provide working code examples
- **Complete API Docs** - Document all parameters and return values
- **Use Cases** - Show real-world applications
- **Error Handling** - Document common errors and solutions

### Submitting Issues

When submitting bug reports, please include:

1. **Environment Information**
   - Python version
   - ChatScope version
   - Operating system
   - Relevant package versions

2. **Reproduction Steps**
   - Minimal code example
   - Input data (anonymized)
   - Expected vs actual behavior

3. **Error Messages**
   - Full traceback
   - Log output (if available)

### Feature Requests

For feature requests, please provide:

1. **Use Case** - Why is this feature needed?
2. **Proposed Solution** - How should it work?
3. **Alternatives** - What alternatives have you considered?
4. **Examples** - Show how the feature would be used

### Release Process

1. **Version Bumping**
   - Update version in `__init__.py`
   - Update version in `pyproject.toml`
   - Update CHANGELOG.md

2. **Testing**
   - Run full test suite
   - Test on multiple Python versions
   - Test installation from PyPI

3. **Documentation**
   - Update README.md
   - Update API documentation
   - Update examples

4. **Release**
   - Create GitHub release
   - Upload to PyPI
   - Announce on relevant channels

## 📋 Changelog

### Version 2.0.0 (2024-01-XX)

**🎉 Major Release - Advanced Analytics**

**New Features:**
- ✨ Advanced sentiment analysis with emotion detection
- 🔍 Topic modeling (LDA, BERTopic, K-Means clustering)
- ⏰ Temporal pattern analysis and trend detection
- 📊 Interactive visualizations with Plotly and Seaborn
- 🖥️ Enhanced CLI with comprehensive options
- 🔧 Comprehensive analysis pipeline
- 📈 Advanced visualization creation
- 🛡️ Improved error handling and logging

**Improvements:**
- 🚀 Better performance with optimized batch processing
- 📝 Enhanced documentation with examples
- 🧪 Comprehensive test suite (95%+ coverage)
- 🔒 Better API key management
- 📦 Modular dependency system

**API Changes:**
- Added `AdvancedChatGPTAnalyzer` class
- Added `comprehensive_analysis()` method
- Added sentiment, topic, and temporal analysis methods
- Added advanced visualization capabilities
- Enhanced CLI with `chatscope-advanced` command

### Version 1.0.2 (2023-12-XX)

**Bug Fixes:**
- Fixed issue with empty conversation titles
- Improved error handling for malformed JSON
- Better rate limiting implementation

**Improvements:**
- Enhanced logging output
- Better progress indicators
- Improved documentation

### Version 1.0.1 (2023-11-XX)

**Bug Fixes:**
- Fixed installation issues with optional dependencies
- Resolved matplotlib backend issues on headless systems
- Fixed encoding issues with non-English conversations

**Improvements:**
- Better error messages
- Enhanced CLI help text
- Updated examples

### Version 1.0.0 (2023-10-XX)

**🎉 Initial Release**

**Features:**
- Basic conversation categorization
- OpenAI API integration
- Simple visualization
- Command-line interface
- JSON export functionality

## 📞 Support

### Getting Help

- **Documentation** - Check this README and the docs/ folder
- **GitHub Issues** - Report bugs and request features
- **Discussions** - Ask questions and share ideas
- **Examples** - Check the examples/ folder for use cases

### Community

- **GitHub** - [https://github.com/22smeargle/chatscope](https://github.com/22smeargle/chatscope)
- **Issues** - [https://github.com/22smeargle/chatscope/issues](https://github.com/22smeargle/chatscope/issues)
- **Discussions** - [https://github.com/22smeargle/chatscope/discussions](https://github.com/22smeargle/chatscope/discussions)

### Commercial Support

For commercial support, custom development, or enterprise features, please contact us at [support@chatscope.dev](mailto:support@chatscope.dev).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** - For providing the GPT API that powers conversation categorization
- **scikit-learn** - For machine learning algorithms used in topic modeling
- **TextBlob** - For sentiment analysis capabilities
- **Plotly** - For interactive visualization features
- **matplotlib & seaborn** - For statistical visualizations
- **pandas** - For data manipulation and analysis
- **NLTK** - For natural language processing utilities
- **BERTopic** - For advanced topic modeling capabilities

---

**Made with ❤️ by the ChatScope team**

*Transform your ChatGPT conversations into actionable insights!*