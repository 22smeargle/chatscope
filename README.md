# 🔍 ChatScope

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CNCL](https://img.shields.io/badge/License-CNCL-red.svg)](https://github.com/22wojciech/chatscope/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/chatscope.svg)](https://badge.fury.io/py/chatscope)

**Discover the scope of your conversations**

A powerful Python library for analyzing and categorizing ChatGPT conversation exports using OpenAI's API. This tool helps you understand your ChatGPT usage patterns by automatically categorizing your conversations into topics like Programming, AI, Psychology, Philosophy, and more.

## Features

- 📊 **Automatic Categorization**: Uses GPT-4 to intelligently categorize your conversations
- 📈 **Visual Analytics**: Generates beautiful bar charts showing conversation distribution
- 🔒 **Secure**: API keys loaded from environment variables
- ⚡ **Rate Limiting**: Built-in rate limiting to respect OpenAI API limits
- 🔄 **Batch Processing**: Efficiently processes large numbers of conversations
- 💾 **Export Results**: Saves detailed results in JSON format
- 🎨 **Customizable**: Custom categories and visualization options
- 🖥️ **CLI Support**: Command-line interface for easy automation
- 🛡️ **Error Handling**: Comprehensive error handling and logging
- 📦 **Easy Integration**: Simple Python API for seamless integration

## Installation

### From PyPI (Recommended)

```bash
pip install chatscope
```

### With plotting support

```bash
pip install chatscope[plotting]
```

### Development Installation

```bash
git clone https://github.com/22wojciech/chatscope.git
cd chatscope
pip install -e .[dev]
```

## Quick Start

### 1. Get Your ChatGPT Data

1. Go to [ChatGPT Settings](https://chat.openai.com/)
2. Navigate to "Data controls" → "Export data"
3. Download your data and extract `conversations.json`

### 2. Set Up OpenAI API Key

Create a `.env` file in your project directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Or set it as an environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 3. Run Analysis

#### Using Python API

```python
from chatscope import ChatGPTAnalyzer

# Initialize analyzer
analyzer = ChatGPTAnalyzer()

# Run analysis
results = analyzer.analyze('conversations.json')

# Print results
print(f"Total conversations: {results['total_conversations']}")
for category, count in results['counts'].items():
    if count > 0:
        print(f"{category}: {count}")
```

#### Using Command Line

```bash
# Basic usage
chatscope conversations.json

# Custom output paths
chatscope conversations.json -o my_chart.png -r my_results.json

# Don't show the plot
chatscope conversations.json --no-show
```

## Advanced Usage

### Custom Categories

```python
from chatscope import ChatGPTAnalyzer

custom_categories = [
    "Work",
    "Personal",
    "Learning",
    "Creative",
    "Technical",
    "Other"
]

analyzer = ChatGPTAnalyzer(categories=custom_categories)
results = analyzer.analyze('conversations.json')
```

### Rate Limiting Configuration

```python
analyzer = ChatGPTAnalyzer(
    batch_size=10,  # Process 10 titles per request
    delay_between_requests=2.0,  # Wait 2 seconds between requests
    max_tokens_per_request=3000  # Limit tokens per request
)
```

### Programmatic Chart Generation

```python
analyzer = ChatGPTAnalyzer()
results = analyzer.analyze(
    'conversations.json',
    output_chart='custom_chart.png',
    show_plot=False  # Don't display, just save
)
```

### Step-by-Step Processing

```python
from chatscope import ChatGPTAnalyzer

# Initialize analyzer
analyzer = ChatGPTAnalyzer()

# Load conversations manually
conversations = analyzer.load_conversations('conversations.json')
print(f"Loaded {len(conversations)} conversations")

# Extract unique titles
titles = analyzer.extract_unique_titles(conversations)
print(f"Found {len(titles)} unique titles")

# Categorize titles
categorizations = analyzer.categorize_all_titles(titles)

# Create category dictionary
category_dict = analyzer.create_category_dictionary(categorizations)

# Count conversations
counts = analyzer.count_conversations_by_category(category_dict)

# Generate chart
chart_path = analyzer.create_bar_chart(counts, figsize=(16, 10))

# Save results
results_path = analyzer.save_results(category_dict, counts)
```

### Batch Processing with Custom Logic

```python
from chatscope import ChatGPTAnalyzer
import time

analyzer = ChatGPTAnalyzer(batch_size=5, delay_between_requests=3.0)

# Process large datasets with custom progress tracking
titles = analyzer.extract_unique_titles(
    analyzer.load_conversations('large_conversations.json')
)

print(f"Processing {len(titles)} titles in batches of {analyzer.batch_size}")

start_time = time.time()
categorizations = analyzer.categorize_all_titles(titles)
end_time = time.time()

print(f"Processing completed in {end_time - start_time:.2f} seconds")
```

### Integration with Data Analysis Workflows

```python
import pandas as pd
from chatscope import ChatGPTAnalyzer

# Analyze conversations
analyzer = ChatGPTAnalyzer()
results = analyzer.analyze('conversations.json')

# Convert to pandas DataFrame for further analysis
data = []
for category, titles in results['categories'].items():
    for title in titles:
        data.append({'category': category, 'title': title})

df = pd.DataFrame(data)

# Perform additional analysis
category_stats = df.groupby('category').size().sort_values(ascending=False)
print("Top categories:")
print(category_stats.head())

# Calculate percentages
total_conversations = len(df)
percentages = (category_stats / total_conversations * 100).round(2)
print("\nCategory percentages:")
print(percentages)
```

## Best Practices

### API Key Management

1. **Environment Variables**: Always use environment variables for API keys:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

2. **`.env` Files**: For development, use `.env` files:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Never Hardcode**: Never hardcode API keys in your source code.

### Rate Limiting

1. **Respect API Limits**: Use appropriate batch sizes and delays:
   ```python
   # For large datasets
   analyzer = ChatGPTAnalyzer(
       batch_size=15,
       delay_between_requests=2.0
   )
   ```

2. **Monitor Usage**: Keep track of your API usage and costs.

3. **Error Handling**: Always implement proper error handling:
   ```python
   from chatscope.exceptions import APIError, DataError
   
   try:
       results = analyzer.analyze('conversations.json')
   except APIError as e:
       print(f"API error: {e}")
   except DataError as e:
       print(f"Data error: {e}")
   ```

### Performance Optimization

1. **Batch Size**: Optimize batch size based on your data:
   - Small datasets (< 100 titles): batch_size=20-30
   - Medium datasets (100-1000 titles): batch_size=15-20
   - Large datasets (> 1000 titles): batch_size=10-15

2. **Caching Results**: Save and reuse categorization results:
   ```python
   # Save results for later use
   results = analyzer.analyze('conversations.json')
   
   # Load previous results
   import json
   with open('categorization_results.json', 'r') as f:
       cached_results = json.load(f)
   ```

3. **Memory Management**: For very large datasets, consider processing in chunks.

### Data Quality

1. **Validate Input**: Ensure your conversations.json follows the expected format:
   ```json
   [
     {
       "title": "Conversation Title",
       "create_time": 1699123456.789,
       "update_time": 1699123456.789
     }
   ]
   ```

2. **Handle Missing Data**: The library automatically handles missing or empty titles.

3. **Custom Categories**: When using custom categories, ensure they are:
   - Mutually exclusive
   - Comprehensive (include an "Other" category)
   - Clear and specific

### Logging and Debugging

1. **Enable Verbose Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   
   analyzer = ChatGPTAnalyzer()
   results = analyzer.analyze('conversations.json')
   ```

2. **CLI Verbose Mode**:
   ```bash
   chatscope -v conversations.json
   ```

3. **Monitor Progress**: Use logging to track processing progress for large datasets.

## Command Line Interface

The package includes a comprehensive CLI:

```bash
chatscope --help
```

### CLI Examples

```bash
# Basic analysis
chatscope conversations.json

# Custom API key and batch size
chatscope --api-key sk-... --batch-size 15 conversations.json

# Custom categories
chatscope --categories "Work" "Personal" "Learning" conversations.json

# Verbose output
chatscope -v conversations.json

# Quiet mode (only errors)
chatscope -q conversations.json

# Custom figure size
chatscope --figsize 16 10 conversations.json
```

## Data Format

Your `conversations.json` should follow this structure:

```json
[
  {
    "title": "Python Data Analysis Tutorial",
    "create_time": 1699123456.789,
    "update_time": 1699123456.789
  },
  {
    "title": "Machine Learning Basics",
    "create_time": 1699123456.789,
    "update_time": 1699123456.789
  }
]
```

## Default Categories

The analyzer uses these categories by default:

- **Programming** - Code, software development, debugging
- **Artificial Intelligence** - AI, ML, data science topics
- **Psychology / Personal Development** - Mental health, self-improvement
- **Philosophy** - Philosophical discussions and questions
- **Astrology / Esoteric** - Spiritual and mystical topics
- **Work / Career** - Professional and career-related conversations
- **Health** - Medical, fitness, and wellness topics
- **Education** - Learning, academic subjects
- **Other** - Everything else

## Output Files

The analyzer generates several output files:

### 1. Chart (`conversation_categories.png`)
A bar chart showing the distribution of conversations across categories.

### 2. Detailed Results (`categorization_results.json`)
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "total_conversations": 150,
  "categories": {
    "Programming": ["Python Tutorial", "Debug Help"],
    "AI": ["ChatGPT Tips", "ML Basics"]
  },
  "counts": {
    "Programming": 45,
    "AI": 32,
    "Other": 73
  }
}
```

## Error Handling

The library includes comprehensive error handling:

```python
from chatscope import ChatGPTAnalyzer
from chatscope.exceptions import ChatGPTAnalyzerError

try:
    analyzer = ChatGPTAnalyzer()
    results = analyzer.analyze('conversations.json')
except APIError as e:
    print(f"OpenAI API error: {e}")
except DataError as e:
    print(f"Data processing error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages (automatically installed):
  - `openai>=0.28.0`
  - `python-dotenv>=0.19.0`
  - `requests>=2.25.0`
- Optional packages:
  - `matplotlib>=3.5.0` (for plotting)

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify, render_template
from chatscope import ChatGPTAnalyzer
from chatscope.exceptions import ChatGPTAnalyzerError
import os

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_conversations():
    try:
        # Get uploaded file
        file = request.files['conversations']
        file_path = f"temp_{file.filename}"
        file.save(file_path)
        
        # Analyze conversations
        analyzer = ChatGPTAnalyzer()
        results = analyzer.analyze(
            input_file=file_path,
            show_plot=False
        )
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'total_conversations': results['total_conversations'],
            'counts': results['counts']
        })
        
    except ChatGPTAnalyzerError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Jupyter Notebook Integration

```python
# Cell 1: Setup
import matplotlib.pyplot as plt
import pandas as pd
from chatscope import ChatGPTAnalyzer

# Configure matplotlib for inline plots
%matplotlib inline
plt.style.use('seaborn-v0_8')

# Cell 2: Analysis
analyzer = ChatGPTAnalyzer()
results = analyzer.analyze('conversations.json', show_plot=False)

print(f"Total conversations analyzed: {results['total_conversations']}")

# Cell 3: Custom Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Bar chart
counts = results['counts']
filtered_counts = {k: v for k, v in counts.items() if v > 0}
ax1.bar(filtered_counts.keys(), filtered_counts.values())
ax1.set_title('Conversations by Category')
ax1.tick_params(axis='x', rotation=45)

# Pie chart
ax2.pie(filtered_counts.values(), labels=filtered_counts.keys(), autopct='%1.1f%%')
ax2.set_title('Category Distribution')

plt.tight_layout()
plt.show()

# Cell 4: Data Export
df = pd.DataFrame([
    {'category': cat, 'count': count, 'percentage': count/results['total_conversations']*100}
    for cat, count in results['counts'].items() if count > 0
])

df.to_csv('conversation_analysis.csv', index=False)
print("Results exported to conversation_analysis.csv")
```

### Automated Reporting Script

```python
#!/usr/bin/env python3
"""
Automated ChatGPT conversation analysis and reporting script.
Usage: python report_generator.py conversations.json
"""

import sys
import os
from datetime import datetime
from chatscope import ChatGPTAnalyzer
from chatscope.exceptions import ChatGPTAnalyzerError

def generate_report(input_file: str, output_dir: str = "reports"):
    """Generate comprehensive analysis report."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Initialize analyzer
        analyzer = ChatGPTAnalyzer()
        
        # Run analysis
        print(f"Analyzing conversations from {input_file}...")
        results = analyzer.analyze(
            input_file=input_file,
            output_chart=f"{output_dir}/chart_{timestamp}.png",
            output_results=f"{output_dir}/results_{timestamp}.json",
            show_plot=False
        )
        
        # Generate text report
        report_path = f"{output_dir}/report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(f"ChatGPT Conversation Analysis Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source: {input_file}\n\n")
            
            f.write(f"SUMMARY\n")
            f.write(f"Total conversations: {results['total_conversations']}\n\n")
            
            f.write(f"CATEGORY BREAKDOWN\n")
            for category, count in sorted(results['counts'].items(), 
                                        key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / results['total_conversations']) * 100
                    f.write(f"{category}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nFILES GENERATED\n")
            f.write(f"Chart: {results['chart_path']}\n")
            f.write(f"Detailed results: {results['results_path']}\n")
            f.write(f"Report: {report_path}\n")
        
        print(f"Report generated successfully!")
        print(f"Files saved in: {output_dir}/")
        
        return results
        
    except ChatGPTAnalyzerError as e:
        print(f"Analysis error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python report_generator.py conversations.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
    
    generate_report(input_file)
```

## Troubleshooting

### Common Issues

#### 1. API Key Errors

**Problem**: `ConfigurationError: OpenAI API key not found`

**Solutions**:
```bash
# Set environment variable
export OPENAI_API_KEY="your_api_key_here"

# Or create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Or pass directly to constructor
analyzer = ChatGPTAnalyzer(api_key="your_api_key_here")
```

#### 2. Import Errors

**Problem**: `ImportError: openai package is required`

**Solution**:
```bash
pip install openai>=0.28.0
```

**Problem**: `ImportError: matplotlib is required for plotting`

**Solution**:
```bash
pip install matplotlib>=3.5.0
# Or install with plotting support
pip install chatscope[plotting]
```

#### 3. Data Format Issues

**Problem**: `DataError: Conversations file must contain a JSON array`

**Solution**: Ensure your JSON file has the correct format:
```json
[
  {
    "title": "Your conversation title",
    "create_time": 1699123456.789,
    "update_time": 1699123456.789
  }
]
```

#### 4. API Rate Limiting

**Problem**: API requests failing due to rate limits

**Solutions**:
```python
# Increase delay between requests
analyzer = ChatGPTAnalyzer(delay_between_requests=3.0)

# Reduce batch size
analyzer = ChatGPTAnalyzer(batch_size=10)

# Combine both
analyzer = ChatGPTAnalyzer(
    batch_size=10,
    delay_between_requests=2.0
)
```

#### 5. Memory Issues with Large Datasets

**Problem**: Out of memory errors with very large conversation files

**Solution**: Process in smaller chunks:
```python
import json
from chatscope import ChatGPTAnalyzer

def process_large_file(file_path, chunk_size=1000):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    analyzer = ChatGPTAnalyzer(batch_size=10)
    all_results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        
        # Save chunk to temporary file
        temp_file = f"temp_chunk_{i}.json"
        with open(temp_file, 'w') as f:
            json.dump(chunk, f)
        
        # Process chunk
        results = analyzer.analyze(temp_file, show_plot=False)
        all_results.append(results)
        
        # Clean up
        os.remove(temp_file)
    
    return all_results
```

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

analyzer = ChatGPTAnalyzer()
results = analyzer.analyze('conversations.json')
```

### Performance Monitoring

```python
import time
from chatscope import ChatGPTAnalyzer

start_time = time.time()
analyzer = ChatGPTAnalyzer()
results = analyzer.analyze('conversations.json')
end_time = time.time()

print(f"Analysis completed in {end_time - start_time:.2f} seconds")
print(f"Processed {results['total_conversations']} conversations")
print(f"Rate: {results['total_conversations'] / (end_time - start_time):.2f} conversations/second")
```

## API Costs

The tool uses OpenAI's GPT-4 API. Costs depend on:
- Number of conversation titles
- Batch size (larger batches = fewer API calls)
- Title length and complexity

Typical costs:
- ~100 conversations: $0.10-0.50
- ~1000 conversations: $1.00-5.00
- ~10000 conversations: $10.00-50.00

### Cost Optimization Tips

1. **Increase Batch Size**: Process more titles per request
   ```python
   analyzer = ChatGPTAnalyzer(batch_size=25)  # Higher batch size
   ```

2. **Cache Results**: Avoid re-processing the same data
   ```python
   # Check if results already exist
   if os.path.exists('categorization_results.json'):
       with open('categorization_results.json', 'r') as f:
           cached_results = json.load(f)
   else:
       results = analyzer.analyze('conversations.json')
   ```

3. **Filter Duplicates**: Remove duplicate titles before processing
   ```python
   conversations = analyzer.load_conversations('conversations.json')
   unique_titles = analyzer.extract_unique_titles(conversations)
   print(f"Reduced from {len(conversations)} to {len(unique_titles)} unique titles")
   ```

## Frequently Asked Questions (FAQ)

### General Questions

**Q: What data does ChatScope analyze?**
A: ChatScope analyzes the conversation titles from your ChatGPT export data. It does not analyze the actual conversation content, only the titles to categorize your conversations.

**Q: Is my conversation data sent to OpenAI?**
A: Only the conversation titles are sent to OpenAI's API for categorization. The actual conversation content is never transmitted.

**Q: Can I use this with other AI chat platforms?**
A: Currently, ChatScope is designed specifically for ChatGPT conversation exports. However, you can adapt it for other platforms by formatting your data to match the expected JSON structure.

**Q: How accurate is the categorization?**
A: The categorization uses GPT-4, which provides high accuracy. However, you can always customize the categories or manually review results for your specific needs.

### Technical Questions

**Q: Can I run this without an OpenAI API key?**
A: No, an OpenAI API key is required as the tool uses GPT-4 for intelligent categorization.

**Q: What's the maximum number of conversations I can analyze?**
A: There's no hard limit, but very large datasets (>10,000 conversations) may take longer and cost more. Consider using batch processing for large datasets.

**Q: Can I modify the default categories?**
A: Yes! You can provide custom categories when initializing the analyzer:
```python
custom_categories = ["Work", "Personal", "Learning", "Entertainment", "Other"]
analyzer = ChatGPTAnalyzer(categories=custom_categories)
```

**Q: How do I export my ChatGPT conversations?**
A: Go to ChatGPT Settings → Data controls → Export data. Download and extract the `conversations.json` file.

**Q: Can I use this in a commercial application?**
A: The current license is for non-commercial use only. Contact plus4822@icloud.com for commercial licensing.

**Q: Why am I getting rate limit errors?**
A: Reduce the batch size and increase the delay between requests:
```python
analyzer = ChatGPTAnalyzer(batch_size=10, delay_between_requests=2.0)
```

**Q: Can I save results in different formats?**
A: Currently, results are saved as JSON and charts as PNG. You can easily convert the results to other formats:
```python
import pandas as pd
results = analyzer.analyze('conversations.json')
df = pd.DataFrame(results['counts'].items(), columns=['Category', 'Count'])
df.to_csv('results.csv', index=False)
```

### Performance Questions

**Q: How long does analysis take?**
A: Processing time depends on the number of conversations and API response times. Typically:
- 100 conversations: 1-2 minutes
- 1000 conversations: 10-20 minutes
- 5000+ conversations: 1+ hours

**Q: How can I speed up the analysis?**
A: Increase the batch size (if you're not hitting rate limits):
```python
analyzer = ChatGPTAnalyzer(batch_size=30)
```

**Q: Can I pause and resume analysis?**
A: Currently, there's no built-in pause/resume functionality. For very large datasets, consider processing in chunks manually.

## Contributing

We welcome contributions! Here's how you can help improve ChatScope:

### Types of Contributions

- 🐛 **Bug Reports**: Report issues you encounter
- 💡 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve documentation and examples
- 🔧 **Code Contributions**: Submit bug fixes or new features
- 🧪 **Testing**: Help improve test coverage

### Development Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/yourusername/chatscope.git
   cd chatscope
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**:
   ```bash
   pip install -e .[dev]
   ```

4. **Set Up Pre-commit Hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Development Workflow

1. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**: Implement your feature or fix

3. **Run Tests**:
   ```bash
   pytest
   pytest --cov=chatscope  # With coverage
   ```

4. **Format Code**:
   ```bash
   black chatscope/
   flake8 chatscope/
   mypy chatscope/
   ```

5. **Update Documentation**: If you've added features, update the README and docstrings

6. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**: Submit a PR with a clear description of your changes

### Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public methods
- Keep functions focused and single-purpose
- Add unit tests for new functionality
- Use meaningful variable and function names

### Testing Guidelines

- Write tests for all new features
- Ensure existing tests pass
- Aim for high test coverage (>90%)
- Use pytest fixtures for common test data
- Mock external API calls in tests

### Documentation Guidelines

- Update README.md for new features
- Add docstrings to all public methods
- Include code examples in docstrings
- Update API reference section
- Add troubleshooting entries for common issues

### Submitting Issues

When submitting bug reports, please include:

- Python version
- ChatScope version
- Operating system
- Complete error message and stack trace
- Minimal code example to reproduce the issue
- Expected vs actual behavior

### Feature Requests

For feature requests, please provide:

- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Any relevant examples or mockups

### Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG in README.md
3. Create release tag
4. GitHub Actions will automatically publish to PyPI

### Getting Help

If you need help with development:

1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Join our community discussions
4. Contact maintainers directly for complex questions

## License

This project is licensed under the Custom Non-Commercial License (CNCL) v1.0 - see the [LICENSE](https://github.com/22wojciech/chatscope/blob/main/LICENSE) file for details.

**⚠️ Important:** This software is free for personal, academic, and research use only. Commercial use requires a separate license. Contact plus4822@icloud.com for commercial licensing.

## API Reference

### ChatGPTAnalyzer Class

The main class for analyzing ChatGPT conversations.

#### Constructor

```python
ChatGPTAnalyzer(
    api_key: Optional[str] = None,
    categories: Optional[List[str]] = None,
    batch_size: int = 20,
    delay_between_requests: float = 1.0,
    max_tokens_per_request: int = 4000
)
```

**Parameters:**
- `api_key` (str, optional): OpenAI API key. If None, loads from `OPENAI_API_KEY` environment variable.
- `categories` (List[str], optional): Custom categories for classification. Uses default categories if None.
- `batch_size` (int): Number of titles to process per API request. Default: 20.
- `delay_between_requests` (float): Delay in seconds between API requests. Default: 1.0.
- `max_tokens_per_request` (int): Maximum tokens per API request. Default: 4000.

**Raises:**
- `ConfigurationError`: If OpenAI API key is not provided or found.
- `ImportError`: If required dependencies are not installed.

#### Methods

##### analyze()

```python
analyze(
    input_file: str = "conversations.json",
    output_chart: str = "conversation_categories.png",
    output_results: str = "categorization_results.json",
    show_plot: bool = True
) -> Dict[str, Any]
```

Main analysis pipeline that processes conversations and generates results.

**Parameters:**
- `input_file` (str): Path to conversations JSON file.
- `output_chart` (str): Path to save the chart.
- `output_results` (str): Path to save detailed results.
- `show_plot` (bool): Whether to display the plot.

**Returns:**
- `Dict[str, Any]`: Analysis results containing:
  - `total_conversations` (int): Total number of unique conversations
  - `categories` (Dict[str, List[str]]): Categories with their conversation titles
  - `counts` (Dict[str, int]): Count of conversations per category
  - `chart_path` (str): Path to generated chart
  - `results_path` (str): Path to saved results

**Raises:**
- `DataError`: If input data is invalid.
- `APIError`: If API requests fail.

##### load_conversations()

```python
load_conversations(file_path: str) -> List[Dict[str, Any]]
```

Load conversations from JSON file.

**Parameters:**
- `file_path` (str): Path to the conversations JSON file.

**Returns:**
- `List[Dict[str, Any]]`: List of conversation dictionaries.

**Raises:**
- `DataError`: If file cannot be loaded or parsed.

##### extract_unique_titles()

```python
extract_unique_titles(conversations: List[Dict[str, Any]]) -> List[str]
```

Extract unique conversation titles from conversations list.

**Parameters:**
- `conversations` (List[Dict[str, Any]]): List of conversation dictionaries.

**Returns:**
- `List[str]`: List of unique conversation titles.

##### categorize_titles_batch()

```python
categorize_titles_batch(titles: List[str]) -> Dict[str, str]
```

Categorize a batch of titles using OpenAI API.

**Parameters:**
- `titles` (List[str]): List of titles to categorize.

**Returns:**
- `Dict[str, str]`: Dictionary mapping titles to categories.

**Raises:**
- `APIError`: If API request fails.

##### categorize_all_titles()

```python
categorize_all_titles(titles: List[str]) -> Dict[str, str]
```

Categorize all titles with batching and rate limiting.

**Parameters:**
- `titles` (List[str]): List of all titles to categorize.

**Returns:**
- `Dict[str, str]`: Dictionary mapping all titles to categories.

##### create_bar_chart()

```python
create_bar_chart(
    counts: Dict[str, int],
    output_path: str = "conversation_categories.png",
    figsize: tuple = (12, 8),
    show_plot: bool = True
) -> Optional[str]
```

Create and save a bar chart of conversation categories.

**Parameters:**
- `counts` (Dict[str, int]): Dictionary with category counts.
- `output_path` (str): Path to save the chart.
- `figsize` (tuple): Figure size as (width, height).
- `show_plot` (bool): Whether to display the plot.

**Returns:**
- `Optional[str]`: Path to saved chart or None if matplotlib is not available.

**Raises:**
- `ImportError`: If matplotlib is not installed.

##### save_results()

```python
save_results(
    category_dict: Dict[str, List[str]],
    counts: Dict[str, int],
    output_file: str = "categorization_results.json"
) -> str
```

Save categorization results to a JSON file.

**Parameters:**
- `category_dict` (Dict[str, List[str]]): Dictionary with categories and title lists.
- `counts` (Dict[str, int]): Dictionary with category counts.
- `output_file` (str): Path to save results.

**Returns:**
- `str`: Path to saved results file.

#### Default Categories

The analyzer uses these categories by default:

- **Programming** - Code, software development, debugging
- **Artificial Intelligence** - AI, ML, data science topics
- **Psychology / Personal Development** - Mental health, self-improvement
- **Philosophy** - Philosophical discussions and questions
- **Astrology / Esoteric** - Spiritual and mystical topics
- **Work / Career** - Professional and career-related conversations
- **Health** - Medical, fitness, and wellness topics
- **Education** - Learning, academic subjects
- **Other** - Everything else

### Exception Classes

#### ChatGPTAnalyzerError

Base exception class for ChatGPT Analyzer.

#### APIError

Raised when OpenAI API requests fail.

**Inherits from:** `ChatGPTAnalyzerError`

#### DataError

Raised when there are issues with input data.

**Inherits from:** `ChatGPTAnalyzerError`

#### ConfigurationError

Raised when there are configuration issues.

**Inherits from:** `ChatGPTAnalyzerError`

### Command Line Interface

The package includes a comprehensive CLI accessible via the `chatscope` command.

#### Basic Usage

```bash
chatscope conversations.json
```

#### CLI Arguments

**Positional Arguments:**
- `input_file`: Path to the conversations JSON file

**Optional Arguments:**
- `-o, --output-chart`: Output path for the chart (default: conversation_categories.png)
- `-r, --output-results`: Output path for detailed results (default: categorization_results.json)
- `--api-key`: OpenAI API key (can also be set via OPENAI_API_KEY environment variable)
- `--batch-size`: Number of titles to process in each API request (default: 20)
- `--delay`: Delay in seconds between API requests (default: 1.0)
- `--max-tokens`: Maximum tokens per API request (default: 4000)
- `--categories`: Custom categories to use (space-separated)
- `--no-show`: Don't display the chart after creation
- `--figsize WIDTH HEIGHT`: Figure size for the chart (default: 12 8)
- `-v, --verbose`: Enable verbose logging
- `-q, --quiet`: Enable quiet mode (only errors)

#### CLI Examples

```bash
# Basic analysis
chatscope conversations.json

# Custom API key and batch size
chatscope --api-key sk-... --batch-size 15 conversations.json

# Custom categories
chatscope --categories "Work" "Personal" "Learning" conversations.json

# Verbose output
chatscope -v conversations.json

# Quiet mode (only errors)
chatscope -q conversations.json

# Custom figure size
chatscope --figsize 16 10 conversations.json

# Save without showing plot
chatscope --no-show -o my_chart.png conversations.json
```

## Changelog

### v1.0.2
- Improved error handling
- Enhanced CLI functionality
- Better documentation
- Bug fixes and optimizations

### v1.0.1
- Added comprehensive logging
- Improved rate limiting
- CLI enhancements
- Documentation updates

### v1.0.0
- Initial release
- Basic conversation categorization
- CLI interface
- Plotting support
- Rate limiting
- Comprehensive error handling

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/22wojciech/chatscope/issues) page
2. Create a new issue with detailed information
3. Include your Python version, OS, and error messages

## Acknowledgments

- OpenAI for providing the GPT-4 API
- The Python community for excellent libraries
- Contributors and users who provide feedback

---

**Made with ❤️ for the ChatGPT community**