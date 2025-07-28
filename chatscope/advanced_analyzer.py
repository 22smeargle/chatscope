"""Advanced analyzer module with sentiment analysis, topic modeling, and temporal analysis."""

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
import re

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
except ImportError:
    TfidfVectorizer = None
    KMeans = None
    LatentDirichletAllocation = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError:
    nltk = None
    stopwords = None
    word_tokenize = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    px = None
    make_subplots = None

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

try:
    from bertopic import BERTopic
except ImportError:
    BERTopic = None

try:
    import openai
except ImportError:
    openai = None

from .analyzer import ChatGPTAnalyzer
from .exceptions import APIError, DataError, ConfigurationError

# Configure logging
logger = logging.getLogger(__name__)


class AdvancedChatGPTAnalyzer(ChatGPTAnalyzer):
    """Advanced analyzer with sentiment analysis, topic modeling, and temporal analysis."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the advanced analyzer."""
        super().__init__(*args, **kwargs)
        self._check_dependencies()
        self._initialize_nltk()
    
    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        missing_deps = []
        
        if np is None or pd is None:
            missing_deps.append("numpy, pandas")
        if TfidfVectorizer is None:
            missing_deps.append("scikit-learn")
        if TextBlob is None:
            missing_deps.append("textblob")
        if nltk is None:
            missing_deps.append("nltk")
        
        if missing_deps:
            logger.warning(f"Some advanced features may not work. Missing: {', '.join(missing_deps)}")
    
    def _initialize_nltk(self):
        """Initialize NLTK data."""
        if nltk is not None:
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                logger.info("Downloading NLTK data...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
    
    def analyze_sentiment(self, conversations: List[Dict[str, Any]], 
                         include_emotional_tone: bool = True) -> Dict[str, Any]:
        """Analyze sentiment of conversations.
        
        Args:
            conversations: List of conversation dictionaries.
            include_emotional_tone: Whether to include emotional tone analysis.
            
        Returns:
            Dictionary with sentiment analysis results.
        """
        if TextBlob is None:
            raise ImportError("textblob is required for sentiment analysis")
        
        logger.info("Analyzing sentiment...")
        
        sentiments = []
        emotions = []
        
        for conv in conversations:
            if 'mapping' in conv:
                # Extract text from conversation messages
                text_content = self._extract_conversation_text(conv)
                
                if text_content:
                    # Analyze sentiment
                    blob = TextBlob(text_content)
                    polarity = blob.sentiment.polarity  # -1 to 1
                    subjectivity = blob.sentiment.subjectivity  # 0 to 1
                    
                    # Categorize sentiment
                    if polarity > 0.1:
                        sentiment_label = "positive"
                    elif polarity < -0.1:
                        sentiment_label = "negative"
                    else:
                        sentiment_label = "neutral"
                    
                    sentiments.append({
                        'title': conv.get('title', 'Unknown'),
                        'sentiment': sentiment_label,
                        'polarity': polarity,
                        'subjectivity': subjectivity,
                        'text_length': len(text_content)
                    })
                    
                    if include_emotional_tone:
                        # Simple emotion detection based on keywords
                        emotion = self._detect_emotion(text_content)
                        emotions.append(emotion)
        
        # Calculate statistics
        sentiment_counts = Counter([s['sentiment'] for s in sentiments])
        avg_polarity = np.mean([s['polarity'] for s in sentiments]) if sentiments else 0
        avg_subjectivity = np.mean([s['subjectivity'] for s in sentiments]) if sentiments else 0
        
        result = {
            'sentiments': sentiments,
            'sentiment_distribution': dict(sentiment_counts),
            'average_polarity': avg_polarity,
            'average_subjectivity': avg_subjectivity,
            'total_analyzed': len(sentiments)
        }
        
        if include_emotional_tone:
            emotion_counts = Counter(emotions)
            result['emotion_distribution'] = dict(emotion_counts)
        
        logger.info(f"Sentiment analysis completed for {len(sentiments)} conversations")
        return result
    
    def _extract_conversation_text(self, conversation: Dict[str, Any]) -> str:
        """Extract text content from a conversation."""
        text_parts = []
        
        if 'mapping' in conversation:
            for node_id, node in conversation['mapping'].items():
                if node and 'message' in node and node['message']:
                    message = node['message']
                    if 'content' in message and message['content']:
                        if 'parts' in message['content']:
                            for part in message['content']['parts']:
                                if isinstance(part, str):
                                    text_parts.append(part)
        
        return ' '.join(text_parts)
    
    def _detect_emotion(self, text: str) -> str:
        """Simple emotion detection based on keywords."""
        text_lower = text.lower()
        
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'pleased', 'delighted', 'cheerful'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated'],
            'sadness': ['sad', 'depressed', 'disappointed', 'unhappy', 'melancholy'],
            'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'concerned'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'unexpected'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened']
        }
        
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotion_scores[emotion] = score
        
        if max(emotion_scores.values()) > 0:
            return max(emotion_scores, key=emotion_scores.get)
        else:
            return 'neutral'
    
    def extract_topics(self, conversations: List[Dict[str, Any]], 
                      num_topics: int = 10, method: str = 'LDA') -> Dict[str, Any]:
        """Extract topics from conversations using topic modeling.
        
        Args:
            conversations: List of conversation dictionaries.
            num_topics: Number of topics to extract.
            method: Topic modeling method ('LDA', 'BERTopic', or 'clustering').
            
        Returns:
            Dictionary with topic modeling results.
        """
        logger.info(f"Extracting topics using {method}...")
        
        # Extract text from conversations
        texts = []
        titles = []
        
        for conv in conversations:
            text = self._extract_conversation_text(conv)
            if text and len(text.strip()) > 50:  # Filter out very short texts
                texts.append(text)
                titles.append(conv.get('title', 'Unknown'))
        
        if not texts:
            raise DataError("No suitable text content found for topic modeling")
        
        if method.upper() == 'LDA':
            return self._extract_topics_lda(texts, titles, num_topics)
        elif method.upper() == 'BERTOPIC':
            return self._extract_topics_bertopic(texts, titles, num_topics)
        elif method.upper() == 'CLUSTERING':
            return self._extract_topics_clustering(texts, titles, num_topics)
        else:
            raise ValueError(f"Unknown topic modeling method: {method}")
    
    def _extract_topics_lda(self, texts: List[str], titles: List[str], 
                           num_topics: int) -> Dict[str, Any]:
        """Extract topics using Latent Dirichlet Allocation."""
        if LatentDirichletAllocation is None:
            raise ImportError("scikit-learn is required for LDA topic modeling")
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=100
        )
        
        lda.fit(tfidf_matrix)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': top_weights
            })
        
        # Assign documents to topics
        doc_topic_probs = lda.transform(tfidf_matrix)
        document_topics = []
        
        for i, (title, probs) in enumerate(zip(titles, doc_topic_probs)):
            dominant_topic = np.argmax(probs)
            document_topics.append({
                'title': title,
                'dominant_topic': int(dominant_topic),
                'topic_probabilities': probs.tolist()
            })
        
        return {
            'method': 'LDA',
            'num_topics': num_topics,
            'topics': topics,
            'document_topics': document_topics,
            'perplexity': lda.perplexity(tfidf_matrix)
        }
    
    def _extract_topics_bertopic(self, texts: List[str], titles: List[str], 
                                num_topics: int) -> Dict[str, Any]:
        """Extract topics using BERTopic."""
        if BERTopic is None:
            raise ImportError("bertopic is required for BERTopic modeling")
        
        # Initialize BERTopic model
        topic_model = BERTopic(
            nr_topics=num_topics,
            verbose=False
        )
        
        # Fit model and predict topics
        topics, probs = topic_model.fit_transform(texts)
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        
        # Format results
        formatted_topics = []
        for topic_id in topic_info['Topic'].values:
            if topic_id != -1:  # Skip outlier topic
                topic_words = topic_model.get_topic(topic_id)
                words = [word for word, _ in topic_words]
                weights = [weight for _, weight in topic_words]
                
                formatted_topics.append({
                    'topic_id': topic_id,
                    'words': words,
                    'weights': weights
                })
        
        # Document topics
        document_topics = []
        for i, (title, topic_id, prob) in enumerate(zip(titles, topics, probs)):
            document_topics.append({
                'title': title,
                'dominant_topic': int(topic_id),
                'topic_probability': float(prob)
            })
        
        return {
            'method': 'BERTopic',
            'num_topics': len(formatted_topics),
            'topics': formatted_topics,
            'document_topics': document_topics
        }
    
    def _extract_topics_clustering(self, texts: List[str], titles: List[str], 
                                  num_topics: int) -> Dict[str, Any]:
        """Extract topics using K-means clustering."""
        if KMeans is None or TfidfVectorizer is None:
            raise ImportError("scikit-learn is required for clustering-based topic modeling")
        
        # Preprocess and vectorize
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Extract topics from cluster centers
        topics = []
        for cluster_id in range(num_topics):
            center = kmeans.cluster_centers_[cluster_id]
            top_indices = center.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [center[i] for i in top_indices]
            
            topics.append({
                'topic_id': cluster_id,
                'words': top_words,
                'weights': top_weights
            })
        
        # Document topics
        document_topics = []
        for i, (title, cluster_id) in enumerate(zip(titles, cluster_labels)):
            document_topics.append({
                'title': title,
                'dominant_topic': int(cluster_id)
            })
        
        return {
            'method': 'Clustering',
            'num_topics': num_topics,
            'topics': topics,
            'document_topics': document_topics,
            'inertia': kmeans.inertia_
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for topic modeling."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_temporal_patterns(self, conversations: List[Dict[str, Any]], 
                                 time_granularity: str = 'daily',
                                 detect_trends: bool = True) -> Dict[str, Any]:
        """Analyze temporal patterns in conversations.
        
        Args:
            conversations: List of conversation dictionaries.
            time_granularity: Time granularity ('hourly', 'daily', 'weekly', 'monthly').
            detect_trends: Whether to detect trends.
            
        Returns:
            Dictionary with temporal analysis results.
        """
        if pd is None:
            raise ImportError("pandas is required for temporal analysis")
        
        logger.info("Analyzing temporal patterns...")
        
        # Extract timestamps
        timestamps = []
        for conv in conversations:
            if 'create_time' in conv and conv['create_time']:
                try:
                    timestamp = datetime.fromtimestamp(conv['create_time'])
                    timestamps.append({
                        'timestamp': timestamp,
                        'title': conv.get('title', 'Unknown'),
                        'conversation_id': conv.get('id', 'unknown')
                    })
                except (ValueError, TypeError):
                    continue
        
        if not timestamps:
            raise DataError("No valid timestamps found in conversations")
        
        # Create DataFrame
        df = pd.DataFrame(timestamps)
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month_name()
        
        # Aggregate by time granularity
        if time_granularity == 'hourly':
            time_series = df.groupby(df['timestamp'].dt.floor('H')).size()
        elif time_granularity == 'daily':
            time_series = df.groupby('date').size()
        elif time_granularity == 'weekly':
            time_series = df.groupby(df['timestamp'].dt.to_period('W')).size()
        elif time_granularity == 'monthly':
            time_series = df.groupby(df['timestamp'].dt.to_period('M')).size()
        else:
            raise ValueError(f"Unknown time granularity: {time_granularity}")
        
        # Calculate patterns
        hourly_pattern = df.groupby('hour').size().to_dict()
        daily_pattern = df.groupby('day_of_week').size().to_dict()
        monthly_pattern = df.groupby('month').size().to_dict()
        
        # Convert time_series keys to strings for JSON serialization
        time_series_dict = {str(k): v for k, v in time_series.to_dict().items()}
        
        result = {
            'time_granularity': time_granularity,
            'time_series': time_series_dict,
            'hourly_pattern': hourly_pattern,
            'daily_pattern': daily_pattern,
            'monthly_pattern': monthly_pattern,
            'total_conversations': len(timestamps),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            }
        }
        
        if detect_trends:
            # Simple trend detection
            if len(time_series) > 1:
                values = list(time_series.values)
                trend = 'increasing' if values[-1] > values[0] else 'decreasing'
                
                # Calculate correlation with time
                x = np.arange(len(values))
                correlation = np.corrcoef(x, values)[0, 1] if len(values) > 1 else 0
                
                result['trend_analysis'] = {
                    'trend': trend,
                    'correlation_with_time': correlation,
                    'peak_period': time_series.idxmax(),
                    'peak_value': time_series.max()
                }
        
        logger.info(f"Temporal analysis completed for {len(timestamps)} conversations")
        return result
    
    def create_advanced_visualizations(self, analysis_results: Dict[str, Any], 
                                     output_dir: str = "visualizations") -> Dict[str, str]:
        """Create advanced visualizations for analysis results.
        
        Args:
            analysis_results: Results from various analysis methods.
            output_dir: Directory to save visualizations.
            
        Returns:
            Dictionary with paths to created visualizations.
        """
        os.makedirs(output_dir, exist_ok=True)
        created_files = {}
        
        # Sentiment visualization
        if 'sentiment_analysis' in analysis_results:
            sentiment_path = self._create_sentiment_visualization(
                analysis_results['sentiment_analysis'], output_dir
            )
            if sentiment_path:
                created_files['sentiment_chart'] = sentiment_path
        
        # Topic modeling visualization
        if 'topic_analysis' in analysis_results:
            topic_path = self._create_topic_visualization(
                analysis_results['topic_analysis'], output_dir
            )
            if topic_path:
                created_files['topic_chart'] = topic_path
        
        # Temporal analysis visualization
        if 'temporal_analysis' in analysis_results:
            temporal_path = self._create_temporal_visualization(
                analysis_results['temporal_analysis'], output_dir
            )
            if temporal_path:
                created_files['temporal_chart'] = temporal_path
        
        # Word cloud
        if 'topic_analysis' in analysis_results:
            wordcloud_path = self._create_wordcloud(
                analysis_results['topic_analysis'], output_dir
            )
            if wordcloud_path:
                created_files['wordcloud'] = wordcloud_path
        
        return created_files
    
    def _create_sentiment_visualization(self, sentiment_data: Dict[str, Any], 
                                       output_dir: str) -> Optional[str]:
        """Create sentiment analysis visualization."""
        if plt is None or sns is None:
            logger.warning("matplotlib/seaborn not available, skipping sentiment visualization")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sentiment Analysis Results', fontsize=16, fontweight='bold')
        
        # Sentiment distribution pie chart
        sentiment_dist = sentiment_data['sentiment_distribution']
        axes[0, 0].pie(sentiment_dist.values(), labels=sentiment_dist.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Sentiment Distribution')
        
        # Polarity histogram
        polarities = [s['polarity'] for s in sentiment_data['sentiments']]
        axes[0, 1].hist(polarities, bins=20, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Polarity Distribution')
        axes[0, 1].set_xlabel('Polarity')
        axes[0, 1].set_ylabel('Frequency')
        
        # Subjectivity histogram
        subjectivities = [s['subjectivity'] for s in sentiment_data['sentiments']]
        axes[1, 0].hist(subjectivities, bins=20, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Subjectivity Distribution')
        axes[1, 0].set_xlabel('Subjectivity')
        axes[1, 0].set_ylabel('Frequency')
        
        # Emotion distribution (if available)
        if 'emotion_distribution' in sentiment_data:
            emotion_dist = sentiment_data['emotion_distribution']
            axes[1, 1].bar(emotion_dist.keys(), emotion_dist.values(), color='coral')
            axes[1, 1].set_title('Emotion Distribution')
            axes[1, 1].set_xlabel('Emotions')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'Emotion data not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'sentiment_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_topic_visualization(self, topic_data: Dict[str, Any], 
                                   output_dir: str) -> Optional[str]:
        """Create topic modeling visualization."""
        if plt is None:
            logger.warning("matplotlib not available, skipping topic visualization")
            return None
        
        topics = topic_data['topics']
        if not topics:
            return None
        
        # Create topic word importance heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for heatmap
        max_words = 10
        topic_words = []
        topic_weights = []
        
        for topic in topics[:10]:  # Show top 10 topics
            words = topic['words'][:max_words]
            weights = topic['weights'][:max_words]
            
            # Pad with zeros if needed
            while len(words) < max_words:
                words.append('')
                weights.append(0)
            
            topic_words.append(words)
            topic_weights.append(weights)
        
        # Create heatmap
        if sns is not None:
            sns.heatmap(topic_weights, 
                       xticklabels=[f'Word {i+1}' for i in range(max_words)],
                       yticklabels=[f'Topic {i}' for i in range(len(topic_weights))],
                       annot=True, fmt='.3f', cmap='YlOrRd')
        else:
            im = ax.imshow(topic_weights, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(max_words))
            ax.set_xticklabels([f'Word {i+1}' for i in range(max_words)])
            ax.set_yticks(range(len(topic_weights)))
            ax.set_yticklabels([f'Topic {i}' for i in range(len(topic_weights))])
            plt.colorbar(im)
        
        plt.title('Topic-Word Importance Heatmap')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'topic_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_temporal_visualization(self, temporal_data: Dict[str, Any], 
                                      output_dir: str) -> Optional[str]:
        """Create temporal analysis visualization."""
        if plt is None:
            logger.warning("matplotlib not available, skipping temporal visualization")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Analysis Results', fontsize=16, fontweight='bold')
        
        # Time series plot
        time_series = temporal_data['time_series']
        if time_series:
            dates = list(time_series.keys())
            values = list(time_series.values())
            
            axes[0, 0].plot(dates, values, marker='o')
            axes[0, 0].set_title(f'Conversations Over Time ({temporal_data["time_granularity"]})')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Number of Conversations')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Hourly pattern
        hourly_pattern = temporal_data['hourly_pattern']
        if hourly_pattern:
            hours = list(hourly_pattern.keys())
            counts = list(hourly_pattern.values())
            
            axes[0, 1].bar(hours, counts, color='skyblue')
            axes[0, 1].set_title('Hourly Pattern')
            axes[0, 1].set_xlabel('Hour of Day')
            axes[0, 1].set_ylabel('Number of Conversations')
        
        # Daily pattern
        daily_pattern = temporal_data['daily_pattern']
        if daily_pattern:
            days = list(daily_pattern.keys())
            counts = list(daily_pattern.values())
            
            axes[1, 0].bar(days, counts, color='lightgreen')
            axes[1, 0].set_title('Daily Pattern')
            axes[1, 0].set_xlabel('Day of Week')
            axes[1, 0].set_ylabel('Number of Conversations')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Monthly pattern
        monthly_pattern = temporal_data['monthly_pattern']
        if monthly_pattern:
            months = list(monthly_pattern.keys())
            counts = list(monthly_pattern.values())
            
            axes[1, 1].bar(months, counts, color='coral')
            axes[1, 1].set_title('Monthly Pattern')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Number of Conversations')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'temporal_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_wordcloud(self, topic_data: Dict[str, Any], 
                         output_dir: str) -> Optional[str]:
        """Create word cloud from topic data."""
        if WordCloud is None:
            logger.warning("wordcloud not available, skipping word cloud creation")
            return None
        
        # Collect all words and their weights
        word_freq = {}
        for topic in topic_data['topics']:
            for word, weight in zip(topic['words'], topic['weights']):
                if word in word_freq:
                    word_freq[word] += weight
                else:
                    word_freq[word] = weight
        
        if not word_freq:
            return None
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate_from_frequencies(word_freq)
        
        # Save word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Topic Words Cloud', fontsize=16, fontweight='bold')
        
        output_path = os.path.join(output_dir, 'wordcloud.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def comprehensive_analysis(self, input_file: str, 
                             include_sentiment: bool = True,
                             include_topics: bool = True,
                             include_temporal: bool = True,
                             topic_method: str = 'LDA',
                             num_topics: int = 10,
                             create_visualizations: bool = True) -> Dict[str, Any]:
        """Perform comprehensive analysis with all advanced features.
        
        Args:
            input_file: Path to conversations JSON file.
            include_sentiment: Whether to include sentiment analysis.
            include_topics: Whether to include topic modeling.
            include_temporal: Whether to include temporal analysis.
            topic_method: Topic modeling method.
            num_topics: Number of topics to extract.
            create_visualizations: Whether to create visualizations.
            
        Returns:
            Dictionary with all analysis results.
        """
        logger.info("Starting comprehensive analysis...")
        
        # Load conversations
        conversations = self.load_conversations(input_file)
        
        results = {
            'metadata': {
                'total_conversations': len(conversations),
                'analysis_timestamp': datetime.now().isoformat(),
                'analyzer_version': '2.0.0'
            }
        }
        
        # Basic categorization
        logger.info("Performing basic categorization...")
        basic_results = self.analyze(input_file, show_plot=False)
        results['basic_categorization'] = basic_results
        
        # Sentiment analysis
        if include_sentiment:
            try:
                sentiment_results = self.analyze_sentiment(conversations)
                results['sentiment_analysis'] = sentiment_results
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                results['sentiment_analysis'] = {'error': str(e)}
        
        # Topic modeling
        if include_topics:
            try:
                topic_results = self.extract_topics(conversations, num_topics, topic_method)
                results['topic_analysis'] = topic_results
            except Exception as e:
                logger.error(f"Topic analysis failed: {e}")
                results['topic_analysis'] = {'error': str(e)}
        
        # Temporal analysis
        if include_temporal:
            try:
                temporal_results = self.analyze_temporal_patterns(conversations)
                results['temporal_analysis'] = temporal_results
            except Exception as e:
                logger.error(f"Temporal analysis failed: {e}")
                results['temporal_analysis'] = {'error': str(e)}
        
        # Create visualizations
        if create_visualizations:
            try:
                viz_paths = self.create_advanced_visualizations(results)
                results['visualizations'] = viz_paths
            except Exception as e:
                logger.error(f"Visualization creation failed: {e}")
                results['visualizations'] = {'error': str(e)}
        
        # Save comprehensive results
        output_file = "comprehensive_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        results['results_file'] = output_file
        
        logger.info("Comprehensive analysis completed!")
        return results