# ChatScope Examples

This document contains practical examples and code snippets for various ChatScope use cases.

## Table of Contents

- [Quick Start Examples](#quick-start-examples)
- [Basic Analysis Examples](#basic-analysis-examples)
- [Advanced Analysis Examples](#advanced-analysis-examples)
- [Visualization Examples](#visualization-examples)
- [Integration Examples](#integration-examples)
- [Custom Analysis Examples](#custom-analysis-examples)
- [Batch Processing Examples](#batch-processing-examples)
- [Real-world Use Cases](#real-world-use-cases)

## Quick Start Examples

### 1. Basic Analysis in 3 Lines

```python
from chatscope import ChatGPTAnalyzer

analyzer = ChatGPTAnalyzer()
results = analyzer.analyze('conversations.json')
print(f"Found {results['total_conversations']} conversations in {len(results['category_counts'])} categories")
```

### 2. Advanced Analysis in 5 Lines

```python
from chatscope import AdvancedChatGPTAnalyzer

analyzer = AdvancedChatGPTAnalyzer()
results = analyzer.comprehensive_analysis('conversations.json')
print(f"Sentiment: {results['sentiment_analysis']['average_polarity']:.2f}")
print(f"Topics found: {len(results['topic_analysis']['topics'])}")
print(f"Peak activity: {results['temporal_analysis']['peak_hour']}:00")
```

### 3. Command Line Quick Start

```bash
# Basic analysis
chatscope conversations.json

# Advanced analysis with visualizations
chatscope-advanced conversations.json --sentiment --topics --temporal --create-visualizations
```

## Basic Analysis Examples

### Custom Categories

```python
from chatscope import ChatGPTAnalyzer

# Define custom categories for your domain
custom_categories = [
    "Software Development",
    "Data Science",
    "Machine Learning",
    "Web Development",
    "DevOps",
    "Career Advice",
    "Personal Projects"
]

analyzer = ChatGPTAnalyzer(categories=custom_categories)
results = analyzer.analyze('conversations.json')

# Print category distribution
print("Category Distribution:")
for category, count in sorted(results['category_counts'].items(), key=lambda x: x[1], reverse=True):
    percentage = (count / results['total_conversations']) * 100
    print(f"  {category}: {count} conversations ({percentage:.1f}%)")
```

### Batch Size Optimization

```python
from chatscope import ChatGPTAnalyzer
import time

# Test different batch sizes for optimal performance
batch_sizes = [5, 10, 15, 20, 25]
results = {}

for batch_size in batch_sizes:
    print(f"Testing batch size: {batch_size}")
    
    analyzer = ChatGPTAnalyzer(
        batch_size=batch_size,
        delay_between_requests=1.0
    )
    
    start_time = time.time()
    analysis_results = analyzer.analyze('conversations.json')
    end_time = time.time()
    
    results[batch_size] = {
        'time': end_time - start_time,
        'conversations': analysis_results['total_conversations']
    }
    
    print(f"  Time: {results[batch_size]['time']:.2f}s")
    print(f"  Rate: {results[batch_size]['conversations'] / results[batch_size]['time']:.1f} conv/s")

# Find optimal batch size
optimal_batch = min(results.keys(), key=lambda x: results[x]['time'])
print(f"\nOptimal batch size: {optimal_batch}")
```

### Error Handling and Retry Logic

```python
from chatscope import ChatGPTAnalyzer, APIError, DataError
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_analysis(file_path, max_retries=3):
    """Perform analysis with robust error handling."""
    
    for attempt in range(max_retries):
        try:
            analyzer = ChatGPTAnalyzer(
                batch_size=10,  # Smaller batch for reliability
                delay_between_requests=2.0,  # Longer delay
                max_retries=3
            )
            
            logger.info(f"Attempt {attempt + 1}: Starting analysis")
            results = analyzer.analyze(file_path)
            logger.info("Analysis completed successfully")
            return results
            
        except APIError as e:
            logger.warning(f"API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5  # Exponential backoff
                logger.info(f"Waiting {wait_time}s before retry")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached for API errors")
                raise
                
        except DataError as e:
            logger.error(f"Data error: {e}")
            # Data errors are not retryable
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise
    
    return None

# Usage
try:
    results = robust_analysis('conversations.json')
    if results:
        print(f"Successfully analyzed {results['total_conversations']} conversations")
except Exception as e:
    print(f"Analysis failed: {e}")
```

## Advanced Analysis Examples

### Comprehensive Sentiment Analysis

```python
from chatscope import AdvancedChatGPTAnalyzer
import json

analyzer = AdvancedChatGPTAnalyzer()

# Load conversations
conversations = analyzer.load_conversations('conversations.json')

# Detailed sentiment analysis
sentiment_results = analyzer.analyze_sentiment(
    conversations, 
    include_emotional_tone=True
)

print("=== SENTIMENT ANALYSIS RESULTS ===")
print(f"Overall Polarity: {sentiment_results['average_polarity']:.3f} (-1 to +1)")
print(f"Overall Subjectivity: {sentiment_results['average_subjectivity']:.3f} (0 to 1)")

# Sentiment distribution
print("\nSentiment Distribution:")
for sentiment, count in sentiment_results['sentiment_distribution'].items():
    print(f"  {sentiment.capitalize()}: {count} conversations")

# Emotion distribution
if 'emotion_distribution' in sentiment_results:
    print("\nEmotion Distribution:")
    for emotion, count in sentiment_results['emotion_distribution'].items():
        print(f"  {emotion.capitalize()}: {count} conversations")

# Find most positive and negative conversations
sentiments = sentiment_results['sentiments']
most_positive = max(sentiments.items(), key=lambda x: x[1]['polarity'])
most_negative = min(sentiments.items(), key=lambda x: x[1]['polarity'])

print(f"\nMost Positive: '{most_positive[0]}' (score: {most_positive[1]['polarity']:.3f})")
print(f"Most Negative: '{most_negative[0]}' (score: {most_negative[1]['polarity']:.3f})")

# Save detailed sentiment results
with open('detailed_sentiment.json', 'w') as f:
    json.dump(sentiment_results, f, indent=2, default=str)
```

### Advanced Topic Modeling

```python
from chatscope import AdvancedChatGPTAnalyzer
import matplotlib.pyplot as plt
import numpy as np

analyzer = AdvancedChatGPTAnalyzer()
conversations = analyzer.load_conversations('conversations.json')

# Compare different topic modeling methods
methods = ['LDA', 'clustering']
topic_results = {}

for method in methods:
    print(f"\n=== {method.upper()} TOPIC MODELING ===")
    
    results = analyzer.extract_topics(
        conversations,
        num_topics=8,
        method=method
    )
    
    topic_results[method] = results
    
    # Display topics
    for i, topic in enumerate(results['topics']):
        words = ', '.join(topic['words'][:5])
        print(f"Topic {i+1}: {words}")
        if 'weights' in topic:
            weights = ', '.join([f"{w:.3f}" for w in topic['weights'][:3]])
            print(f"  Weights: {weights}")

# Topic assignment analysis
print("\n=== TOPIC ASSIGNMENTS ===")
for method in methods:
    assignments = topic_results[method]['topic_assignments']
    topic_counts = {}
    
    for title, topic_id in assignments.items():
        topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
    
    print(f"\n{method} Topic Distribution:")
    for topic_id, count in sorted(topic_counts.items()):
        print(f"  Topic {topic_id}: {count} conversations")

# Visualize topic distributions
fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))
if len(methods) == 1:
    axes = [axes]

for i, method in enumerate(methods):
    assignments = topic_results[method]['topic_assignments']
    topic_counts = {}
    
    for topic_id in assignments.values():
        topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
    
    topics = list(topic_counts.keys())
    counts = list(topic_counts.values())
    
    axes[i].bar(topics, counts)
    axes[i].set_title(f'{method} Topic Distribution')
    axes[i].set_xlabel('Topic ID')
    axes[i].set_ylabel('Number of Conversations')

plt.tight_layout()
plt.savefig('topic_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### BERTopic Advanced Example

```python
from chatscope import AdvancedChatGPTAnalyzer

# Note: Requires bertopic installation
# pip install bertopic

analyzer = AdvancedChatGPTAnalyzer()
conversations = analyzer.load_conversations('conversations.json')

try:
    # Advanced BERTopic analysis
    bertopic_results = analyzer.extract_topics(
        conversations,
        num_topics=12,
        method='BERTopic'
    )
    
    print("=== BERTOPIC ANALYSIS ===")
    print(f"Number of topics found: {len(bertopic_results['topics'])}")
    
    # Display topics with semantic labels
    for topic in bertopic_results['topics']:
        print(f"\nTopic {topic['topic_id']}: {topic.get('label', 'Unlabeled')}")
        print(f"Keywords: {', '.join(topic['words'][:7])}")
        
        if 'coherence' in topic:
            print(f"Coherence Score: {topic['coherence']:.3f}")
        
        # Show sample conversations for this topic
        topic_conversations = [
            title for title, assigned_topic in bertopic_results['topic_assignments'].items()
            if assigned_topic == topic['topic_id']
        ]
        
        print(f"Sample conversations ({len(topic_conversations)} total):")
        for conv_title in topic_conversations[:3]:
            print(f"  - {conv_title}")
    
    # Topic evolution over time (if timestamps available)
    if 'topic_evolution' in bertopic_results:
        print("\n=== TOPIC EVOLUTION ===")
        evolution = bertopic_results['topic_evolution']
        for period, topics in evolution.items():
            print(f"{period}: {len(topics)} active topics")
            
except ImportError:
    print("BERTopic not available. Install with: pip install bertopic")
except Exception as e:
    print(f"BERTopic analysis failed: {e}")
    print("Falling back to LDA...")
    
    # Fallback to LDA
    lda_results = analyzer.extract_topics(
        conversations,
        num_topics=10,
        method='LDA'
    )
    print(f"LDA found {len(lda_results['topics'])} topics")
```

### Temporal Pattern Analysis

```python
from chatscope import AdvancedChatGPTAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

analyzer = AdvancedChatGPTAnalyzer()
conversations = analyzer.load_conversations('conversations.json')

# Comprehensive temporal analysis
temporal_results = analyzer.analyze_temporal_patterns(
    conversations,
    time_granularity='hourly',
    detect_trends=True
)

print("=== TEMPORAL ANALYSIS ===")
print(f"Peak Hour: {temporal_results['peak_hour']}:00")
print(f"Peak Day: {temporal_results['peak_day']}")
print(f"Overall Trend: {temporal_results['trend']}")

# Detailed hourly analysis
print("\nHourly Activity Pattern:")
hourly_pattern = temporal_results['hourly_pattern']
for hour in range(24):
    count = hourly_pattern.get(str(hour), 0)
    bar = '█' * min(count // 2, 50)  # Visual bar
    print(f"{hour:2d}:00 │{bar:<50}│ {count}")

# Weekly pattern analysis
print("\nWeekly Activity Pattern:")
daily_pattern = temporal_results['daily_pattern']
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for day in days:
    count = daily_pattern.get(day, 0)
    bar = '█' * min(count // 5, 30)
    print(f"{day:9} │{bar:<30}│ {count}")

# Create comprehensive temporal visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Hourly heatmap
hours = list(range(24))
hourly_counts = [hourly_pattern.get(str(h), 0) for h in hours]
axes[0, 0].plot(hours, hourly_counts, marker='o')
axes[0, 0].set_title('Activity by Hour of Day')
axes[0, 0].set_xlabel('Hour')
axes[0, 0].set_ylabel('Number of Conversations')
axes[0, 0].grid(True, alpha=0.3)

# Weekly bar chart
weekly_counts = [daily_pattern.get(day, 0) for day in days]
axes[0, 1].bar(range(len(days)), weekly_counts)
axes[0, 1].set_title('Activity by Day of Week')
axes[0, 1].set_xlabel('Day')
axes[0, 1].set_ylabel('Number of Conversations')
axes[0, 1].set_xticks(range(len(days)))
axes[0, 1].set_xticklabels([d[:3] for d in days])

# Monthly trend (if available)
if 'monthly_pattern' in temporal_results:
    monthly_pattern = temporal_results['monthly_pattern']
    months = sorted(monthly_pattern.keys())
    monthly_counts = [monthly_pattern[month] for month in months]
    
    axes[1, 0].plot(range(len(months)), monthly_counts, marker='s')
    axes[1, 0].set_title('Monthly Activity Trend')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Number of Conversations')
    axes[1, 0].set_xticks(range(len(months)))
    axes[1, 0].set_xticklabels(months, rotation=45)

# Activity heatmap (hour vs day)
if 'activity_heatmap' in temporal_results:
    heatmap_data = temporal_results['activity_heatmap']
    # Convert to matrix format for visualization
    heatmap_matrix = np.zeros((7, 24))
    
    for day_idx, day in enumerate(days):
        for hour in range(24):
            key = f"{day}_{hour}"
            heatmap_matrix[day_idx, hour] = heatmap_data.get(key, 0)
    
    im = axes[1, 1].imshow(heatmap_matrix, cmap='YlOrRd', aspect='auto')
    axes[1, 1].set_title('Activity Heatmap (Day vs Hour)')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Day of Week')
    axes[1, 1].set_yticks(range(7))
    axes[1, 1].set_yticklabels([d[:3] for d in days])
    axes[1, 1].set_xticks(range(0, 24, 4))
    plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save temporal data for further analysis
temporal_df = pd.DataFrame({
    'hour': hours,
    'conversations': hourly_counts
})
temporal_df.to_csv('hourly_activity.csv', index=False)

print("\nTemporal analysis saved to:")
print("  - temporal_analysis.png")
print("  - hourly_activity.csv")
```

## Visualization Examples

### Custom Visualization Suite

```python
from chatscope import AdvancedChatGPTAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

analyzer = AdvancedChatGPTAnalyzer()
results = analyzer.comprehensive_analysis('conversations.json')

def create_custom_visualizations(results, output_dir='./custom_viz'):
    """Create a comprehensive set of custom visualizations."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Enhanced Category Pie Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    category_counts = results['basic_categorization']['category_counts']
    
    # Create pie chart with custom styling
    wedges, texts, autotexts = ax.pie(
        category_counts.values(),
        labels=category_counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.05 if count == max(category_counts.values()) else 0 
                for count in category_counts.values()]
    )
    
    # Enhance text styling
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Conversation Categories Distribution', 
                fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{output_dir}/enhanced_categories.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sentiment Analysis Dashboard
    if 'sentiment_analysis' in results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        sentiment_data = results['sentiment_analysis']
        
        # Sentiment distribution
        sent_dist = sentiment_data['sentiment_distribution']
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        axes[0, 0].bar(sent_dist.keys(), sent_dist.values(), color=colors)
        axes[0, 0].set_title('Sentiment Distribution')
        axes[0, 0].set_ylabel('Number of Conversations')
        
        # Polarity histogram
        sentiments = sentiment_data['sentiments']
        polarities = [s['polarity'] for s in sentiments.values()]
        axes[0, 1].hist(polarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(np.mean(polarities), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(polarities):.3f}')
        axes[0, 1].set_title('Polarity Distribution')
        axes[0, 1].set_xlabel('Polarity Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Subjectivity vs Polarity scatter
        polarities = [s['polarity'] for s in sentiments.values()]
        subjectivities = [s['subjectivity'] for s in sentiments.values()]
        scatter = axes[1, 0].scatter(polarities, subjectivities, 
                                   alpha=0.6, c=polarities, cmap='RdYlGn')
        axes[1, 0].set_xlabel('Polarity')
        axes[1, 0].set_ylabel('Subjectivity')
        axes[1, 0].set_title('Sentiment Scatter Plot')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # Emotion distribution (if available)
        if 'emotion_distribution' in sentiment_data:
            emotion_dist = sentiment_data['emotion_distribution']
            axes[1, 1].pie(emotion_dist.values(), labels=emotion_dist.keys(), 
                          autopct='%1.1f%%')
            axes[1, 1].set_title('Emotion Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_dashboard.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Topic Analysis Visualization
    if 'topic_analysis' in results:
        topic_data = results['topic_analysis']
        topics = topic_data['topics']
        
        # Topic word clouds
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, topic in enumerate(topics[:8]):
            if i < len(axes):
                # Create word frequency dict
                word_freq = dict(zip(topic['words'], topic.get('weights', [1]*len(topic['words']))))
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='viridis'
                ).generate_from_frequencies(word_freq)
                
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'Topic {i+1}')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(topics), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Topic Word Clouds', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/topic_wordclouds.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Temporal Heatmap
    if 'temporal_analysis' in results:
        temporal_data = results['temporal_analysis']
        
        # Create activity heatmap
        if 'activity_heatmap' in temporal_data:
            heatmap_data = temporal_data['activity_heatmap']
            
            # Convert to matrix
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hours = list(range(24))
            matrix = np.zeros((len(days), len(hours)))
            
            for day_idx, day in enumerate(days):
                for hour_idx, hour in enumerate(hours):
                    key = f"{day}_{hour}"
                    matrix[day_idx, hour_idx] = heatmap_data.get(key, 0)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(15, 8))
            sns.heatmap(matrix, 
                       xticklabels=hours,
                       yticklabels=days,
                       cmap='YlOrRd',
                       annot=False,
                       fmt='d',
                       cbar_kws={'label': 'Number of Conversations'})
            
            ax.set_title('Activity Heatmap: Day vs Hour', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Day of Week')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/activity_heatmap.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Custom visualizations saved to {output_dir}/")
    return output_dir

# Create visualizations
viz_dir = create_custom_visualizations(results)
print(f"Visualizations created in: {viz_dir}")
```

### Interactive Plotly Visualizations

```python
from chatscope import AdvancedChatGPTAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Note: Requires plotly installation
# pip install plotly

analyzer = AdvancedChatGPTAnalyzer()
results = analyzer.comprehensive_analysis('conversations.json')

def create_interactive_dashboard(results):
    """Create interactive Plotly dashboard."""
    
    # 1. Interactive Category Sunburst
    category_counts = results['basic_categorization']['category_counts']
    
    fig_sunburst = go.Figure(go.Sunburst(
        labels=list(category_counts.keys()),
        values=list(category_counts.values()),
        parents=[""] * len(category_counts),
        branchvalues="total"
    ))
    
    fig_sunburst.update_layout(
        title="Interactive Category Distribution",
        font_size=12
    )
    
    fig_sunburst.write_html("interactive_categories.html")
    
    # 2. Sentiment Analysis Dashboard
    if 'sentiment_analysis' in results:
        sentiment_data = results['sentiment_analysis']
        sentiments = sentiment_data['sentiments']
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame([
            {
                'title': title,
                'polarity': data['polarity'],
                'subjectivity': data['subjectivity'],
                'emotion': data.get('emotion', 'neutral')
            }
            for title, data in sentiments.items()
        ])
        
        # Interactive scatter plot
        fig_scatter = px.scatter(
            df, 
            x='polarity', 
            y='subjectivity',
            color='emotion',
            hover_data=['title'],
            title="Sentiment Analysis: Polarity vs Subjectivity"
        )
        
        fig_scatter.update_layout(
            xaxis_title="Polarity (Negative ← → Positive)",
            yaxis_title="Subjectivity (Objective ← → Subjective)"
        )
        
        fig_scatter.write_html("interactive_sentiment.html")
    
    # 3. Temporal Analysis Dashboard
    if 'temporal_analysis' in results:
        temporal_data = results['temporal_analysis']
        
        # Hourly activity
        hourly_pattern = temporal_data['hourly_pattern']
        hours = list(range(24))
        hourly_counts = [hourly_pattern.get(str(h), 0) for h in hours]
        
        # Daily activity
        daily_pattern = temporal_data['daily_pattern']
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts = [daily_pattern.get(day, 0) for day in days]
        
        # Create subplots
        fig_temporal = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hourly Activity', 'Daily Activity', 
                          'Activity Heatmap', 'Trends'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Hourly line chart
        fig_temporal.add_trace(
            go.Scatter(x=hours, y=hourly_counts, mode='lines+markers',
                      name='Hourly Activity'),
            row=1, col=1
        )
        
        # Daily bar chart
        fig_temporal.add_trace(
            go.Bar(x=days, y=daily_counts, name='Daily Activity'),
            row=1, col=2
        )
        
        # Activity heatmap
        if 'activity_heatmap' in temporal_data:
            heatmap_data = temporal_data['activity_heatmap']
            matrix = []
            
            for day in days:
                row = []
                for hour in hours:
                    key = f"{day}_{hour}"
                    row.append(heatmap_data.get(key, 0))
                matrix.append(row)
            
            fig_temporal.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=hours,
                    y=days,
                    colorscale='YlOrRd',
                    showscale=False
                ),
                row=2, col=1
            )
        
        fig_temporal.update_layout(
            title="Temporal Analysis Dashboard",
            height=800
        )
        
        fig_temporal.write_html("interactive_temporal.html")
    
    # 4. Topic Analysis Network
    if 'topic_analysis' in results:
        topic_data = results['topic_analysis']
        topics = topic_data['topics']
        
        # Create topic network visualization
        # This is a simplified version - you could use networkx for more complex networks
        
        topic_labels = []
        topic_sizes = []
        topic_colors = []
        
        for i, topic in enumerate(topics):
            label = f"Topic {i+1}\n{', '.join(topic['words'][:3])}"
            topic_labels.append(label)
            
            # Size based on number of assigned conversations
            assignments = topic_data.get('topic_assignments', {})
            size = sum(1 for t in assignments.values() if t == i)
            topic_sizes.append(size)
            topic_colors.append(i)
        
        # Create bubble chart as topic overview
        fig_topics = go.Figure(data=go.Scatter(
            x=list(range(len(topics))),
            y=[1] * len(topics),
            mode='markers+text',
            marker=dict(
                size=topic_sizes,
                sizemode='diameter',
                sizeref=max(topic_sizes)/50,
                color=topic_colors,
                colorscale='Viridis',
                showscale=True
            ),
            text=topic_labels,
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>Conversations: %{marker.size}<extra></extra>'
        ))
        
        fig_topics.update_layout(
            title="Topic Overview",
            xaxis_title="Topic Index",
            yaxis=dict(showticklabels=False),
            height=400
        )
        
        fig_topics.write_html("interactive_topics.html")
    
    print("Interactive visualizations created:")
    print("  - interactive_categories.html")
    print("  - interactive_sentiment.html")
    print("  - interactive_temporal.html")
    print("  - interactive_topics.html")

# Create interactive dashboard
create_interactive_dashboard(results)
```

## Integration Examples

### Jupyter Notebook Integration

```python
# Jupyter notebook cell
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display, HTML, Markdown
from chatscope import AdvancedChatGPTAnalyzer

# Initialize analyzer
analyzer = AdvancedChatGPTAnalyzer()

# Run analysis with progress tracking
print("Starting comprehensive analysis...")
results = analyzer.comprehensive_analysis(
    'conversations.json',
    include_sentiment=True,
    include_topics=True,
    include_temporal=True
)

# Display results with rich formatting
display(Markdown("## 📊 Analysis Results"))

# Summary statistics
summary_html = f"""
<div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px; margin: 10px 0;">
    <h3>📈 Summary Statistics</h3>
    <ul>
        <li><strong>Total Conversations:</strong> {results['metadata']['total_conversations']}</li>
        <li><strong>Categories Found:</strong> {len(results['basic_categorization']['category_counts'])}</li>
        <li><strong>Average Sentiment:</strong> {results['sentiment_analysis']['average_polarity']:.3f}</li>
        <li><strong>Topics Discovered:</strong> {len(results['topic_analysis']['topics'])}</li>
        <li><strong>Peak Activity Hour:</strong> {results['temporal_analysis']['peak_hour']}:00</li>
    </ul>
</div>
"""

display(HTML(summary_html))

# Create inline visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Category distribution
category_counts = results['basic_categorization']['category_counts']
axes[0, 0].pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
axes[0, 0].set_title('Category Distribution')

# Sentiment distribution
sentiment_dist = results['sentiment_analysis']['sentiment_distribution']
axes[0, 1].bar(sentiment_dist.keys(), sentiment_dist.values(), 
               color=['red', 'gray', 'green'])
axes[0, 1].set_title('Sentiment Distribution')

# Hourly activity
hourly_pattern = results['temporal_analysis']['hourly_pattern']
hours = list(range(24))
counts = [hourly_pattern.get(str(h), 0) for h in hours]
axes[1, 0].plot(hours, counts, marker='o')
axes[1, 0].set_title('Activity by Hour')
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Conversations')

# Topic word frequency
topics = results['topic_analysis']['topics']
if topics:
    top_topic = topics[0]
    words = top_topic['words'][:10]
    weights = top_topic.get('weights', [1]*len(words))[:10]
    
    axes[1, 1].barh(words, weights)
    axes[1, 1].set_title(f'Top Topic Words')
    axes[1, 1].set_xlabel('Weight')

plt.tight_layout()
plt.show()

# Interactive widgets for exploration
from ipywidgets import interact, Dropdown

def explore_topics(topic_index):
    if topic_index < len(topics):
        topic = topics[topic_index]
        print(f"Topic {topic_index + 1}:")
        print(f"Keywords: {', '.join(topic['words'][:10])}")
        
        # Show conversations assigned to this topic
        assignments = results['topic_analysis']['topic_assignments']
        topic_conversations = [
            title for title, assigned_topic in assignments.items()
            if assigned_topic == topic_index
        ]
        
        print(f"\nConversations in this topic ({len(topic_conversations)}):")
        for conv in topic_conversations[:5]:
            print(f"  - {conv}")
        
        if len(topic_conversations) > 5:
            print(f"  ... and {len(topic_conversations) - 5} more")

# Create interactive topic explorer
topic_dropdown = Dropdown(
    options=[(f"Topic {i+1}", i) for i in range(len(topics))],
    description='Topic:'
)

interact(explore_topics, topic_index=topic_dropdown)
```

### Flask Web Application

```python
# app.py
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
from chatscope import AdvancedChatGPTAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

analyzer = AdvancedChatGPTAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.json'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    filename = data.get('filename')
    options = data.get('options', {})
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Run analysis based on options
        results = analyzer.comprehensive_analysis(
            filepath,
            include_sentiment=options.get('sentiment', True),
            include_topics=options.get('topics', True),
            include_temporal=options.get('temporal', True),
            create_visualizations=False  # Don't create files in web app
        )
        
        # Prepare response data
        response_data = {
            'success': True,
            'summary': {
                'total_conversations': results['metadata']['total_conversations'],
                'categories': len(results['basic_categorization']['category_counts']),
                'average_sentiment': results['sentiment_analysis']['average_polarity'],
                'topics_found': len(results['topic_analysis']['topics'])
            },
            'categories': results['basic_categorization']['category_counts'],
            'sentiment': {
                'distribution': results['sentiment_analysis']['sentiment_distribution'],
                'average_polarity': results['sentiment_analysis']['average_polarity']
            },
            'topics': [
                {
                    'id': i,
                    'words': topic['words'][:5],
                    'weight': topic.get('weights', [0])[0] if topic.get('weights') else 0
                }
                for i, topic in enumerate(results['topic_analysis']['topics'][:10])
            ],
            'temporal': {
                'peak_hour': results['temporal_analysis']['peak_hour'],
                'peak_day': results['temporal_analysis']['peak_day'],
                'hourly_pattern': results['temporal_analysis']['hourly_pattern']
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/export/<filename>')
def export_results(filename):
    # Export analysis results
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        results = analyzer.comprehensive_analysis(filepath)
        
        # Save results to JSON
        export_path = os.path.join(app.config['UPLOAD_FOLDER'], f'results_{filename}')
        with open(export_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return send_file(export_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatScope Web Analyzer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
        }
        .results {
            margin-top: 30px;
            display: none;
        }
        .chart-container {
            width: 100%;
            height: 400px;
            margin: 20px 0;
        }
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 ChatScope Web Analyzer</h1>
        <p>Upload your ChatGPT conversations JSON file for comprehensive analysis.</p>
        
        <div class="upload-area">
            <input type="file" id="fileInput" accept=".json" style="display: none;">
            <button onclick="document.getElementById('fileInput').click()">Choose File</button>
            <p>or drag and drop your conversations.json file here</p>
        </div>
        
        <div class="analysis-options">
            <h3>Analysis Options</h3>
            <label><input type="checkbox" id="sentimentCheck" checked> Sentiment Analysis</label><br>
            <label><input type="checkbox" id="topicsCheck" checked> Topic Modeling</label><br>
            <label><input type="checkbox" id="temporalCheck" checked> Temporal Analysis</label><br>
        </div>
        
        <button id="analyzeBtn" onclick="runAnalysis()" disabled>Run Analysis</button>
        
        <div class="loading" id="loading">
            <p>🔄 Analyzing your conversations...</p>
        </div>
        
        <div class="results" id="results">
            <h2>📊 Analysis Results</h2>
            
            <div class="summary-stats" id="summaryStats">
                <!-- Summary statistics will be populated here -->
            </div>
            
            <div class="chart-container">
                <canvas id="categoryChart"></canvas>
            </div>
            
            <div class="chart-container">
                <canvas id="sentimentChart"></canvas>
            </div>
            
            <div class="chart-container">
                <canvas id="temporalChart"></canvas>
            </div>
            
            <button onclick="exportResults()">📥 Export Results</button>
        </div>
    </div>
    
    <script>
        let uploadedFilename = null;
        
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                uploadFile(file);
            }
        });
        
        function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    uploadedFilename = data.filename;
                    document.getElementById('analyzeBtn').disabled = false;
                    alert('File uploaded successfully!');
                }
            })
            .catch(error => {
                alert('Upload failed: ' + error);
            });
        }
        
        function runAnalysis() {
            if (!uploadedFilename) {
                alert('Please upload a file first');
                return;
            }
            
            const options = {
                sentiment: document.getElementById('sentimentCheck').checked,
                topics: document.getElementById('topicsCheck').checked,
                temporal: document.getElementById('temporalCheck').checked
            };
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    filename: uploadedFilename,
                    options: options
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                if (data.success) {
                    displayResults(data);
                } else {
                    alert('Analysis failed: ' + data.error);
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert('Analysis failed: ' + error);
            });
        }
        
        function displayResults(data) {
            // Display summary statistics
            const summaryHtml = `
                <div class="stat-card">
                    <div class="stat-value">${data.summary.total_conversations}</div>
                    <div>Total Conversations</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.summary.categories}</div>
                    <div>Categories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.summary.average_sentiment.toFixed(3)}</div>
                    <div>Avg Sentiment</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.summary.topics_found}</div>
                    <div>Topics Found</div>
                </div>
            `;
            document.getElementById('summaryStats').innerHTML = summaryHtml;
            
            // Create category chart
            const categoryCtx = document.getElementById('categoryChart').getContext('2d');
            new Chart(categoryCtx, {
                type: 'pie',
                data: {
                    labels: Object.keys(data.categories),
                    datasets: [{
                        data: Object.values(data.categories),
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                            '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Category Distribution'
                        }
                    }
                }
            });
            
            // Create sentiment chart
            const sentimentCtx = document.getElementById('sentimentChart').getContext('2d');
            new Chart(sentimentCtx, {
                type: 'bar',
                data: {
                    labels: Object.keys(data.sentiment.distribution),
                    datasets: [{
                        label: 'Conversations',
                        data: Object.values(data.sentiment.distribution),
                        backgroundColor: ['#FF6384', '#C9CBCF', '#36A2EB']
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Sentiment Distribution'
                        }
                    }
                }
            });
            
            // Create temporal chart
            const temporalCtx = document.getElementById('temporalChart').getContext('2d');
            const hours = Array.from({length: 24}, (_, i) => i);
            const hourlyData = hours.map(h => data.temporal.hourly_pattern[h.toString()] || 0);
            
            new Chart(temporalCtx, {
                type: 'line',
                data: {
                    labels: hours.map(h => h + ':00'),
                    datasets: [{
                        label: 'Conversations',
                        data: hourlyData,
                        borderColor: '#36A2EB',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Activity by Hour of Day'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            document.getElementById('results').style.display = 'block';
        }
        
        function exportResults() {
            if (uploadedFilename) {
                window.location.href = '/export/' + uploadedFilename;
            }
        }
    </script>
</body>
</html>
```

### Streamlit Dashboard

```python
# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from chatscope import AdvancedChatGPTAnalyzer
import json
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="ChatScope Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔍 ChatScope Analysis Dashboard")
st.markdown("""
Upload your ChatGPT conversations JSON file to get comprehensive insights about your chat patterns,
sentiment, topics, and temporal behavior.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# API key input
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key. You can also set the OPENAI_API_KEY environment variable."
)

# Analysis options
st.sidebar.header("Analysis Options")
include_sentiment = st.sidebar.checkbox("Sentiment Analysis", value=True)
include_topics = st.sidebar.checkbox("Topic Modeling", value=True)
include_temporal = st.sidebar.checkbox("Temporal Analysis", value=True)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    topic_method = st.selectbox(
        "Topic Modeling Method",
        ["LDA", "clustering", "BERTopic"],
        index=0
    )
    num_topics = st.slider("Number of Topics", 3, 20, 8)
    batch_size = st.slider("API Batch Size", 5, 30, 15)

# File upload
st.header("📁 Upload Conversations")
uploaded_file = st.file_uploader(
    "Choose your conversations.json file",
    type="json",
    help="Export your ChatGPT conversations and upload the JSON file here."
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Initialize analyzer
        analyzer = AdvancedChatGPTAnalyzer(
            api_key=api_key if api_key else None,
            batch_size=batch_size
        )
        
        # Load and validate data
        with st.spinner("Loading conversations..."):
            conversations = analyzer.load_conversations(tmp_file_path)
        
        st.success(f"✅ Loaded {len(conversations)} conversations")
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Running comprehensive analysis..."):
                try:
                    results = analyzer.comprehensive_analysis(
                        tmp_file_path,
                        include_sentiment=include_sentiment,
                        include_topics=include_topics,
                        include_temporal=include_temporal,
                        topic_method=topic_method,
                        num_topics=num_topics,
                        create_visualizations=False
                    )
                    
                    # Store results in session state
                    st.session_state.analysis_results = results
                    st.success("✅ Analysis completed!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.stop()
        
        # Display results if available
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results
            
            # Summary metrics
            st.header("📊 Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Conversations",
                    results['metadata']['total_conversations']
                )
            
            with col2:
                st.metric(
                    "Categories",
                    len(results['basic_categorization']['category_counts'])
                )
            
            with col3:
                if include_sentiment:
                    avg_sentiment = results['sentiment_analysis']['average_polarity']
                    st.metric(
                        "Average Sentiment",
                        f"{avg_sentiment:.3f}",
                        delta=f"{avg_sentiment:.3f}" if avg_sentiment > 0 else None
                    )
            
            with col4:
                if include_topics:
                    st.metric(
                        "Topics Found",
                        len(results['topic_analysis']['topics'])
                    )
            
            # Category Analysis
            st.header("📂 Category Analysis")
            category_counts = results['basic_categorization']['category_counts']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Category pie chart
                fig_pie = px.pie(
                    values=list(category_counts.values()),
                    names=list(category_counts.keys()),
                    title="Category Distribution"
                )
                st.plotly_