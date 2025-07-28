# ChatScope API Reference

This document provides detailed API reference for all ChatScope classes and methods.

## Table of Contents

- [ChatGPTAnalyzer (Basic)](#chatgptanalyzer-basic)
- [AdvancedChatGPTAnalyzer](#advancedchatgptanalyzer)
- [Exception Classes](#exception-classes)
- [Utility Functions](#utility-functions)
- [Configuration](#configuration)

## ChatGPTAnalyzer (Basic)

The base analyzer class for basic conversation categorization.

### Constructor

```python
ChatGPTAnalyzer(
    api_key: Optional[str] = None,
    categories: Optional[List[str]] = None,
    batch_size: int = 20,
    delay_between_requests: float = 1.0
)
```

**Parameters:**
- `api_key` (str, optional): OpenAI API key. If None, loads from environment variable.
- `categories` (List[str], optional): Custom categories for classification.
- `batch_size` (int): Number of titles to process per API request.
- `delay_between_requests` (float): Delay in seconds between API requests.

**Raises:**
- `ConfigurationError`: If API key is not provided or found.
- `ImportError`: If required dependencies are not installed.

### Methods

#### load_conversations()

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

#### extract_unique_titles()

```python
extract_unique_titles(conversations: List[Dict]) -> List[str]
```

Extract unique conversation titles from conversations list.

**Parameters:**
- `conversations` (List[Dict]): List of conversation dictionaries.

**Returns:**
- `List[str]`: List of unique conversation titles.

#### categorize_all_titles()

```python
categorize_all_titles(titles: List[str]) -> Dict[str, str]
```

Categorize all titles using OpenAI API with batching and rate limiting.

**Parameters:**
- `titles` (List[str]): List of titles to categorize.

**Returns:**
- `Dict[str, str]`: Dictionary mapping titles to categories.

**Raises:**
- `APIError`: If API requests fail.

#### create_category_dictionary()

```python
create_category_dictionary(
    titles: List[str], 
    categories: Dict[str, str]
) -> Dict[str, List[str]]
```

Create a dictionary organizing titles by category.

**Parameters:**
- `titles` (List[str]): List of conversation titles.
- `categories` (Dict[str, str]): Mapping of titles to categories.

**Returns:**
- `Dict[str, List[str]]`: Dictionary with categories as keys and title lists as values.

#### count_conversations_by_category()

```python
count_conversations_by_category(
    category_dict: Dict[str, List[str]]
) -> Dict[str, int]
```

Count conversations in each category.

**Parameters:**
- `category_dict` (Dict[str, List[str]]): Dictionary with categories and title lists.

**Returns:**
- `Dict[str, int]`: Dictionary with category counts.

#### create_bar_chart()

```python
create_bar_chart(
    category_counts: Dict[str, int],
    output_path: str = "conversation_categories.png"
) -> Optional[str]
```

Create and save a bar chart of conversation categories.

**Parameters:**
- `category_counts` (Dict[str, int]): Dictionary with category counts.
- `output_path` (str): Path to save the chart.

**Returns:**
- `Optional[str]`: Path to saved chart or None if matplotlib is not available.

#### save_results()

```python
save_results(
    results: Dict[str, Any],
    output_path: str = "analysis_results.json"
) -> str
```

Save analysis results to a JSON file.

**Parameters:**
- `results` (Dict[str, Any]): Analysis results dictionary.
- `output_path` (str): Path to save results.

**Returns:**
- `str`: Path to saved results file.

#### analyze()

```python
analyze(
    input_file: str,
    output_dir: str = "."
) -> Dict[str, Any]
```

Main analysis pipeline that processes conversations and generates results.

**Parameters:**
- `input_file` (str): Path to conversations JSON file.
- `output_dir` (str): Directory to save output files.

**Returns:**
- `Dict[str, Any]`: Analysis results containing:
  - `total_conversations` (int): Total number of conversations
  - `categories` (Dict[str, str]): Mapping of titles to categories
  - `category_counts` (Dict[str, int]): Count per category
  - `metadata` (Dict): Analysis metadata

## AdvancedChatGPTAnalyzer

Extended analyzer class with advanced analytics capabilities.

### Constructor

```python
AdvancedChatGPTAnalyzer(
    api_key: Optional[str] = None,
    categories: Optional[List[str]] = None,
    batch_size: int = 20,
    delay_between_requests: float = 1.0,
    max_retries: int = 3,
    debug_mode: bool = False
)
```

Inherits all parameters from `ChatGPTAnalyzer` plus:

**Additional Parameters:**
- `max_retries` (int): Maximum number of retries for failed API calls.
- `debug_mode` (bool): Enable debug logging and intermediate result saving.

### Advanced Analysis Methods

#### analyze_sentiment()

```python
analyze_sentiment(
    conversations: List[Dict[str, Any]],
    include_emotional_tone: bool = True
) -> Dict[str, Any]
```

Analyze sentiment and emotional tone of conversations.

**Parameters:**
- `conversations` (List[Dict[str, Any]]): List of conversation dictionaries.
- `include_emotional_tone` (bool): Whether to include emotion detection.

**Returns:**
- `Dict[str, Any]`: Sentiment analysis results containing:
  - `sentiments` (Dict): Per-conversation sentiment scores
  - `average_polarity` (float): Overall sentiment polarity (-1 to 1)
  - `average_subjectivity` (float): Overall subjectivity (0 to 1)
  - `sentiment_distribution` (Dict): Distribution of sentiment categories
  - `emotion_distribution` (Dict): Distribution of detected emotions

#### extract_topics()

```python
extract_topics(
    conversations: List[Dict[str, Any]],
    num_topics: int = 10,
    method: str = 'LDA'
) -> Dict[str, Any]
```

Extract topics from conversation content using various methods.

**Parameters:**
- `conversations` (List[Dict[str, Any]]): List of conversation dictionaries.
- `num_topics` (int): Number of topics to extract.
- `method` (str): Topic modeling method ('LDA', 'BERTopic', 'clustering').

**Returns:**
- `Dict[str, Any]`: Topic analysis results containing:
  - `topics` (List[Dict]): List of discovered topics with words and weights
  - `topic_assignments` (Dict): Mapping of conversations to topics
  - `method_used` (str): Topic modeling method used
  - `coherence_score` (float): Topic coherence score (if available)

#### analyze_temporal_patterns()

```python
analyze_temporal_patterns(
    conversations: List[Dict[str, Any]],
    time_granularity: str = 'daily',
    detect_trends: bool = True
) -> Dict[str, Any]
```

Analyze temporal patterns in conversation data.

**Parameters:**
- `conversations` (List[Dict[str, Any]]): List of conversation dictionaries.
- `time_granularity` (str): Time granularity ('hourly', 'daily', 'weekly', 'monthly').
- `detect_trends` (bool): Whether to detect trends and patterns.

**Returns:**
- `Dict[str, Any]`: Temporal analysis results containing:
  - `hourly_pattern` (Dict): Conversation counts by hour
  - `daily_pattern` (Dict): Conversation counts by day of week
  - `monthly_pattern` (Dict): Conversation counts by month
  - `peak_hour` (int): Hour with most activity
  - `peak_day` (str): Day with most activity
  - `trend` (str): Overall trend direction
  - `activity_heatmap` (Dict): Time-based activity heatmap data

#### create_advanced_visualizations()

```python
create_advanced_visualizations(
    analysis_results: Dict[str, Any],
    output_dir: str = "./visualizations"
) -> Dict[str, str]
```

Create comprehensive visualizations from analysis results.

**Parameters:**
- `analysis_results` (Dict[str, Any]): Complete analysis results.
- `output_dir` (str): Directory to save visualizations.

**Returns:**
- `Dict[str, str]`: Dictionary mapping visualization types to file paths.

#### comprehensive_analysis()

```python
comprehensive_analysis(
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

Run comprehensive analysis with all available features.

**Parameters:**
- `input_file` (str): Path to conversations JSON file.
- `include_sentiment` (bool): Include sentiment analysis.
- `include_topics` (bool): Include topic modeling.
- `include_temporal` (bool): Include temporal analysis.
- `create_visualizations` (bool): Create visualization files.
- `topic_method` (str): Topic modeling method to use.
- `num_topics` (int): Number of topics to extract.
- `time_granularity` (str): Time granularity for temporal analysis.

**Returns:**
- `Dict[str, Any]`: Comprehensive analysis results containing all enabled analyses.

## Exception Classes

### APIError

```python
class APIError(Exception):
    """Raised when OpenAI API calls fail."""
    pass
```

Raised when:
- API key is invalid
- Rate limits are exceeded
- API service is unavailable
- Request format is invalid

### DataError

```python
class DataError(Exception):
    """Raised when conversation data is invalid or corrupted."""
    pass
```

Raised when:
- JSON file is malformed
- Required fields are missing
- Data format is unexpected
- File cannot be read

### ConfigurationError

```python
class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass
```

Raised when:
- API key is not provided
- Invalid parameters are passed
- Required dependencies are missing
- Configuration file is malformed

## Utility Functions

### validate_conversations()

```python
validate_conversations(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]
```

Validate and filter conversation data.

**Parameters:**
- `conversations` (List[Dict[str, Any]]): Raw conversation data.

**Returns:**
- `List[Dict[str, Any]]`: Validated conversation data.

### extract_conversation_text()

```python
extract_conversation_text(conversation: Dict[str, Any]) -> str
```

Extract text content from a conversation dictionary.

**Parameters:**
- `conversation` (Dict[str, Any]): Single conversation dictionary.

**Returns:**
- `str`: Extracted text content.

### calculate_summary_stats()

```python
calculate_summary_stats(results_list: List[Dict[str, Any]]) -> Dict[str, Any]
```

Calculate summary statistics from multiple analysis results.

**Parameters:**
- `results_list` (List[Dict[str, Any]]): List of analysis results.

**Returns:**
- `Dict[str, Any]`: Summary statistics.

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key (required)
- `CHATSCOPE_LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `CHATSCOPE_CACHE_DIR`: Directory for caching results
- `CHATSCOPE_OUTPUT_DIR`: Default output directory

### Configuration File Format

ChatScope can be configured using a JSON configuration file:

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
            "AI",
            "Health",
            "Education",
            "Work",
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

### Loading Configuration

```python
from chatscope import AdvancedChatGPTAnalyzer
import json

# Load configuration from file
with open('chatscope_config.json', 'r') as f:
    config = json.load(f)

# Initialize analyzer with configuration
analyzer = AdvancedChatGPTAnalyzer(
    batch_size=config['api_settings']['batch_size'],
    delay_between_requests=config['api_settings']['delay_between_requests'],
    categories=config['analysis_settings']['default_categories']
)
```

## Data Formats

### Input Data Format

ChatGPT conversation export format:

```json
[
  {
    "title": "Conversation Title",
    "create_time": 1699123456.789,
    "update_time": 1699123456.789,
    "mapping": {
      "message_id": {
        "id": "message_id",
        "message": {
          "id": "message_id",
          "author": {
            "role": "user"
          },
          "content": {
            "content_type": "text",
            "parts": ["Message content"]
          },
          "create_time": 1699123456.789
        }
      }
    }
  }
]
```

### Output Data Format

Analysis results format:

```json
{
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "total_conversations": 150,
    "analysis_version": "2.0.0",
    "features_used": ["categorization", "sentiment", "topics"]
  },
  "basic_categorization": {
    "categories": {
      "Title 1": "Programming",
      "Title 2": "AI"
    },
    "category_counts": {
      "Programming": 45,
      "AI": 32,
      "Other": 73
    }
  },
  "sentiment_analysis": {
    "sentiments": {
      "Title 1": {
        "polarity": 0.2,
        "subjectivity": 0.6,
        "emotion": "neutral"
      }
    },
    "average_polarity": 0.15,
    "sentiment_distribution": {
      "positive": 60,
      "neutral": 70,
      "negative": 20
    }
  },
  "topic_analysis": {
    "topics": [
      {
        "topic_id": 0,
        "words": ["python", "code", "programming"],
        "weights": [0.3, 0.25, 0.2]
      }
    ],
    "method_used": "LDA",
    "coherence_score": 0.45
  },
  "temporal_analysis": {
    "hourly_pattern": {
      "0": 2, "1": 1, "9": 15, "14": 20
    },
    "peak_hour": 14,
    "trend": "increasing"
  }
}
```

## Error Handling

### Best Practices

```python
from chatscope import AdvancedChatGPTAnalyzer, APIError, DataError, ConfigurationError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    analyzer = AdvancedChatGPTAnalyzer(api_key='your-key')
    results = analyzer.comprehensive_analysis('conversations.json')
    
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    # Handle configuration issues
    
except DataError as e:
    logger.error(f"Data error: {e}")
    # Handle data format issues
    
except APIError as e:
    logger.error(f"API error: {e}")
    # Handle API-related issues
    
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected errors
```

### Common Error Scenarios

1. **Missing API Key**
   ```python
   # Error: ConfigurationError: OpenAI API key not found
   # Solution: Set OPENAI_API_KEY environment variable
   ```

2. **Invalid JSON Format**
   ```python
   # Error: DataError: Invalid JSON format in conversations file
   # Solution: Validate JSON structure
   ```

3. **Rate Limit Exceeded**
   ```python
   # Error: APIError: Rate limit exceeded
   # Solution: Increase delay_between_requests or reduce batch_size
   ```

4. **Missing Dependencies**
   ```python
   # Error: ImportError: scikit-learn is required for topic modeling
   # Solution: pip install scikit-learn
   ```

## Performance Considerations

### Optimization Tips

1. **Batch Size Optimization**
   - Small datasets (< 100): batch_size=25-30
   - Medium datasets (100-1000): batch_size=15-20
   - Large datasets (> 1000): batch_size=10-15

2. **Memory Management**
   - Process large datasets in chunks
   - Clear intermediate results when not needed
   - Use generators for large data processing

3. **API Rate Limiting**
   - Monitor API usage and costs
   - Implement exponential backoff for retries
   - Cache results to avoid reprocessing

4. **Parallel Processing**
   - Use multiprocessing for independent analyses
   - Implement async processing for I/O operations
   - Balance parallelism with API rate limits

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

# Apply to analysis methods
AdvancedChatGPTAnalyzer.comprehensive_analysis = monitor_performance(
    AdvancedChatGPTAnalyzer.comprehensive_analysis
)
```

---

*This API reference is for ChatScope 2.0. For the latest updates, check the [GitHub repository](https://github.com/22smeargle/chatscope).*