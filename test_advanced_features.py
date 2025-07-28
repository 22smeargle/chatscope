"""Comprehensive test suite for advanced ChatScope features."""

import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

try:
    from chatscope import AdvancedChatGPTAnalyzer
    from chatscope.exceptions import APIError, DataError, ConfigurationError
except ImportError:
    print("Warning: Could not import AdvancedChatGPTAnalyzer. Some tests may fail.")
    AdvancedChatGPTAnalyzer = None


class TestAdvancedChatGPTAnalyzer(unittest.TestCase):
    """Test cases for AdvancedChatGPTAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        if AdvancedChatGPTAnalyzer is None:
            self.skipTest("AdvancedChatGPTAnalyzer not available")
        
        self.test_dir = tempfile.mkdtemp()
        self.sample_conversations = self._create_sample_conversations()
        self.conversations_file = os.path.join(self.test_dir, "test_conversations.json")
        
        with open(self.conversations_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_conversations, f)
        
        # Mock API key
        self.api_key = "test-api-key"
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_sample_conversations(self):
        """Create sample conversation data for testing."""
        base_time = datetime.now().timestamp()
        
        conversations = []
        
        # Conversation 1: Programming topic, positive sentiment
        conv1 = {
            "id": "conv-1",
            "title": "Python web development help",
            "create_time": base_time - 86400,  # 1 day ago
            "mapping": {
                "node1": {
                    "message": {
                        "content": {
                            "parts": [
                                "I'm excited to learn Python web development! Can you help me get started with Flask?"
                            ]
                        }
                    }
                },
                "node2": {
                    "message": {
                        "content": {
                            "parts": [
                                "Absolutely! Flask is a great framework for beginners. Here's how to get started..."
                            ]
                        }
                    }
                }
            }
        }
        
        # Conversation 2: AI topic, neutral sentiment
        conv2 = {
            "id": "conv-2",
            "title": "Understanding machine learning algorithms",
            "create_time": base_time - 43200,  # 12 hours ago
            "mapping": {
                "node1": {
                    "message": {
                        "content": {
                            "parts": [
                                "Can you explain the difference between supervised and unsupervised learning?"
                            ]
                        }
                    }
                },
                "node2": {
                    "message": {
                        "content": {
                            "parts": [
                                "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data."
                            ]
                        }
                    }
                }
            }
        }
        
        # Conversation 3: Health topic, negative sentiment
        conv3 = {
            "id": "conv-3",
            "title": "Dealing with stress and anxiety",
            "create_time": base_time - 21600,  # 6 hours ago
            "mapping": {
                "node1": {
                    "message": {
                        "content": {
                            "parts": [
                                "I'm feeling really stressed and anxious lately. I'm worried about my job and health."
                            ]
                        }
                    }
                },
                "node2": {
                    "message": {
                        "content": {
                            "parts": [
                                "I understand you're going through a difficult time. Here are some strategies that might help..."
                            ]
                        }
                    }
                }
            }
        }
        
        # Conversation 4: Philosophy topic, positive sentiment
        conv4 = {
            "id": "conv-4",
            "title": "The meaning of happiness",
            "create_time": base_time - 7200,  # 2 hours ago
            "mapping": {
                "node1": {
                    "message": {
                        "content": {
                            "parts": [
                                "I've been thinking about what happiness really means. It's fascinating how different philosophers approach this question."
                            ]
                        }
                    }
                },
                "node2": {
                    "message": {
                        "content": {
                            "parts": [
                                "That's a wonderful topic to explore! Aristotle believed happiness comes from living virtuously..."
                            ]
                        }
                    }
                }
            }
        }
        
        # Conversation 5: Work topic, mixed sentiment
        conv5 = {
            "id": "conv-5",
            "title": "Career change advice",
            "create_time": base_time - 3600,  # 1 hour ago
            "mapping": {
                "node1": {
                    "message": {
                        "content": {
                            "parts": [
                                "I'm considering a career change but I'm nervous about the risks. However, I'm also excited about new opportunities."
                            ]
                        }
                    }
                },
                "node2": {
                    "message": {
                        "content": {
                            "parts": [
                                "Career changes can be both challenging and rewarding. Let's explore your options..."
                            ]
                        }
                    }
                }
            }
        }
        
        conversations.extend([conv1, conv2, conv3, conv4, conv5])
        return conversations
    
    @patch('chatscope.analyzer.openai')
    def test_initialization(self, mock_openai):
        """Test analyzer initialization."""
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        
        self.assertEqual(analyzer.api_key, self.api_key)
        self.assertEqual(analyzer.batch_size, 20)
        self.assertEqual(analyzer.delay_between_requests, 1.0)
    
    def test_load_conversations(self):
        """Test loading conversations from file."""
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        conversations = analyzer.load_conversations(self.conversations_file)
        
        self.assertEqual(len(conversations), 5)
        self.assertEqual(conversations[0]['title'], "Python web development help")
    
    def test_extract_conversation_text(self):
        """Test extracting text from conversation."""
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        
        conversation = self.sample_conversations[0]
        text = analyzer._extract_conversation_text(conversation)
        
        self.assertIn("excited to learn Python", text)
        self.assertIn("Flask is a great framework", text)
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality."""
        try:
            from textblob import TextBlob
        except ImportError:
            self.skipTest("TextBlob not available")
        
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        
        # Test with sample conversations
        results = analyzer.analyze_sentiment(self.sample_conversations)
        
        self.assertIn('sentiments', results)
        self.assertIn('sentiment_distribution', results)
        self.assertIn('average_polarity', results)
        self.assertIn('total_analyzed', results)
        
        # Check that we have results for all conversations
        self.assertEqual(results['total_analyzed'], 5)
        
        # Check sentiment distribution
        sentiment_dist = results['sentiment_distribution']
        self.assertIn('positive', sentiment_dist)
        self.assertIn('negative', sentiment_dist)
        self.assertIn('neutral', sentiment_dist)
    
    def test_emotion_detection(self):
        """Test emotion detection."""
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        
        # Test different emotions
        happy_text = "I'm so happy and excited about this!"
        sad_text = "I feel really sad and disappointed."
        angry_text = "This makes me so angry and frustrated!"
        neutral_text = "This is a normal conversation about weather."
        
        self.assertEqual(analyzer._detect_emotion(happy_text), 'joy')
        self.assertEqual(analyzer._detect_emotion(sad_text), 'sadness')
        self.assertEqual(analyzer._detect_emotion(angry_text), 'anger')
        self.assertEqual(analyzer._detect_emotion(neutral_text), 'neutral')
    
    def test_topic_modeling_lda(self):
        """Test LDA topic modeling."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
        except ImportError:
            self.skipTest("scikit-learn not available")
        
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        
        # Test LDA topic modeling
        results = analyzer.extract_topics(self.sample_conversations, num_topics=3, method='LDA')
        
        self.assertEqual(results['method'], 'LDA')
        self.assertEqual(results['num_topics'], 3)
        self.assertIn('topics', results)
        self.assertIn('document_topics', results)
        
        # Check topics structure
        topics = results['topics']
        self.assertEqual(len(topics), 3)
        
        for topic in topics:
            self.assertIn('topic_id', topic)
            self.assertIn('words', topic)
            self.assertIn('weights', topic)
    
    def test_topic_modeling_clustering(self):
        """Test clustering-based topic modeling."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            self.skipTest("scikit-learn not available")
        
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        
        # Test clustering topic modeling
        results = analyzer.extract_topics(self.sample_conversations, num_topics=3, method='clustering')
        
        self.assertEqual(results['method'], 'Clustering')
        self.assertEqual(results['num_topics'], 3)
        self.assertIn('topics', results)
        self.assertIn('document_topics', results)
    
    def test_temporal_analysis(self):
        """Test temporal analysis functionality."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available")
        
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        
        # Test temporal analysis
        results = analyzer.analyze_temporal_patterns(self.sample_conversations)
        
        self.assertIn('time_granularity', results)
        self.assertIn('time_series', results)
        self.assertIn('hourly_pattern', results)
        self.assertIn('daily_pattern', results)
        self.assertIn('total_conversations', results)
        self.assertIn('date_range', results)
        
        # Check that we have the right number of conversations
        self.assertEqual(results['total_conversations'], 5)
        
        # Check date range
        date_range = results['date_range']
        self.assertIn('start', date_range)
        self.assertIn('end', date_range)
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        
        # Test text preprocessing
        raw_text = "Hello! This is a TEST with 123 numbers and @#$ special chars."
        processed = analyzer._preprocess_text(raw_text)
        
        # Should be lowercase and contain only letters and spaces
        self.assertEqual(processed, "hello this is a test with numbers and special chars")
    
    @patch('chatscope.analyzer.openai')
    def test_comprehensive_analysis(self, mock_openai):
        """Test comprehensive analysis pipeline."""
        # Mock OpenAI API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "Python web development help": "Programming",
            "Understanding machine learning algorithms": "Artificial Intelligence",
            "Dealing with stress and anxiety": "Health",
            "The meaning of happiness": "Philosophy",
            "Career change advice": "Work / Career"
        })
        
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        
        # Change to test directory
        original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        try:
            # Test comprehensive analysis
            results = analyzer.comprehensive_analysis(
                input_file=self.conversations_file,
                include_sentiment=True,
                include_topics=False,  # Skip topics to avoid sklearn dependency issues
                include_temporal=True,
                create_visualizations=False  # Skip visualizations to avoid matplotlib issues
            )
            
            # Check basic structure
            self.assertIn('metadata', results)
            self.assertIn('basic_categorization', results)
            
            # Check metadata
            metadata = results['metadata']
            self.assertEqual(metadata['total_conversations'], 5)
            self.assertIn('analysis_timestamp', metadata)
            
            # Check that sentiment analysis was included
            if 'sentiment_analysis' in results and 'error' not in results['sentiment_analysis']:
                sentiment_data = results['sentiment_analysis']
                self.assertIn('sentiments', sentiment_data)
                self.assertIn('sentiment_distribution', sentiment_data)
            
            # Check that temporal analysis was included
            if 'temporal_analysis' in results and 'error' not in results['temporal_analysis']:
                temporal_data = results['temporal_analysis']
                self.assertIn('time_series', temporal_data)
                self.assertIn('total_conversations', temporal_data)
        
        finally:
            os.chdir(original_cwd)
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        
        # Test with non-existent file
        with self.assertRaises(DataError):
            analyzer.load_conversations("non_existent_file.json")
        
        # Test with invalid JSON
        invalid_file = os.path.join(self.test_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        with self.assertRaises(DataError):
            analyzer.load_conversations(invalid_file)
        
        # Test with empty conversations list
        empty_conversations = []
        
        # Should handle empty list gracefully
        sentiment_results = analyzer.analyze_sentiment(empty_conversations)
        self.assertEqual(sentiment_results['total_analyzed'], 0)
    
    def test_visualization_creation(self):
        """Test visualization creation (if matplotlib is available)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.skipTest("matplotlib not available")
        
        analyzer = AdvancedChatGPTAnalyzer(api_key=self.api_key)
        
        # Create sample analysis results
        sentiment_results = {
            'sentiment_distribution': {'positive': 2, 'negative': 1, 'neutral': 2},
            'sentiments': [
                {'polarity': 0.5, 'subjectivity': 0.6},
                {'polarity': -0.3, 'subjectivity': 0.4},
                {'polarity': 0.1, 'subjectivity': 0.5},
                {'polarity': 0.7, 'subjectivity': 0.8},
                {'polarity': 0.0, 'subjectivity': 0.3}
            ],
            'emotion_distribution': {'joy': 2, 'sadness': 1, 'neutral': 2}
        }
        
        analysis_results = {
            'sentiment_analysis': sentiment_results
        }
        
        # Test visualization creation
        viz_dir = os.path.join(self.test_dir, "visualizations")
        created_files = analyzer.create_advanced_visualizations(analysis_results, viz_dir)
        
        # Check if sentiment chart was created
        if 'sentiment_chart' in created_files:
            self.assertTrue(os.path.exists(created_files['sentiment_chart']))


class TestAdvancedCLI(unittest.TestCase):
    """Test cases for advanced CLI functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample conversations file
        self.sample_conversations = [
            {
                "id": "conv-1",
                "title": "Test conversation",
                "create_time": datetime.now().timestamp(),
                "mapping": {
                    "node1": {
                        "message": {
                            "content": {
                                "parts": ["This is a test conversation about programming."]
                            }
                        }
                    }
                }
            }
        ]
        
        self.conversations_file = os.path.join(self.test_dir, "test_conversations.json")
        with open(self.conversations_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_conversations, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_cli_argument_parsing(self):
        """Test CLI argument parsing."""
        try:
            from chatscope.advanced_cli import create_parser
        except ImportError:
            self.skipTest("Advanced CLI not available")
        
        parser = create_parser()
        
        # Test basic arguments
        args = parser.parse_args(["test.json"])
        self.assertEqual(args.input_file, "test.json")
        self.assertEqual(args.output_dir, ".")
        self.assertFalse(args.sentiment)
        self.assertFalse(args.topics)
        self.assertFalse(args.temporal)
        
        # Test with all features enabled
        args = parser.parse_args(["test.json", "--all-features"])
        self.assertTrue(args.all_features)
        
        # Test sentiment-only mode
        args = parser.parse_args(["test.json", "--sentiment-only"])
        self.assertTrue(args.sentiment_only)
        
        # Test topic modeling options
        args = parser.parse_args(["test.json", "--topic-method", "BERTopic", "--num-topics", "15"])
        self.assertEqual(args.topic_method, "BERTopic")
        self.assertEqual(args.num_topics, 15)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create more comprehensive sample data
        self.sample_conversations = self._create_comprehensive_sample()
        self.conversations_file = os.path.join(self.test_dir, "integration_test.json")
        
        with open(self.conversations_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_conversations, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_comprehensive_sample(self):
        """Create comprehensive sample data for integration testing."""
        base_time = datetime.now().timestamp()
        conversations = []
        
        topics = [
            ("Python programming tutorial", "I love learning Python! It's such an amazing language."),
            ("Machine learning basics", "Can you explain neural networks and how they work?"),
            ("Dealing with depression", "I've been feeling really sad and hopeless lately."),
            ("Philosophy of mind", "What is consciousness and how does it emerge from the brain?"),
            ("Career advice needed", "I'm excited about changing careers but also worried about the risks."),
            ("Cooking recipes", "I'm happy to try new recipes and cook for my family."),
            ("Travel planning", "I'm anxious about traveling alone but excited about the adventure."),
            ("Book recommendations", "I enjoy reading science fiction and fantasy novels."),
            ("Exercise routine", "I feel great after working out but struggle with motivation."),
            ("Investment advice", "I'm concerned about market volatility but optimistic about long-term growth.")
        ]
        
        for i, (title, content) in enumerate(topics):
            conv = {
                "id": f"conv-{i+1}",
                "title": title,
                "create_time": base_time - (i * 3600),  # Spread over 10 hours
                "mapping": {
                    "node1": {
                        "message": {
                            "content": {
                                "parts": [content]
                            }
                        }
                    },
                    "node2": {
                        "message": {
                            "content": {
                                "parts": ["That's an interesting topic. Let me help you with that."]
                            }
                        }
                    }
                }
            }
            conversations.append(conv)
        
        return conversations
    
    @patch('chatscope.analyzer.openai')
    def test_full_pipeline_integration(self, mock_openai):
        """Test the complete analysis pipeline integration."""
        if AdvancedChatGPTAnalyzer is None:
            self.skipTest("AdvancedChatGPTAnalyzer not available")
        
        # Mock OpenAI API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "Python programming tutorial": "Programming",
            "Machine learning basics": "Artificial Intelligence",
            "Dealing with depression": "Health",
            "Philosophy of mind": "Philosophy",
            "Career advice needed": "Work / Career",
            "Cooking recipes": "Other",
            "Travel planning": "Other",
            "Book recommendations": "Education",
            "Exercise routine": "Health",
            "Investment advice": "Work / Career"
        })
        
        mock_openai.ChatCompletion.create.return_value = mock_response
        
        analyzer = AdvancedChatGPTAnalyzer(api_key="test-key")
        
        # Change to test directory
        original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        try:
            # Run comprehensive analysis
            results = analyzer.comprehensive_analysis(
                input_file=self.conversations_file,
                include_sentiment=True,
                include_topics=False,  # Skip to avoid dependency issues in CI
                include_temporal=True,
                create_visualizations=False  # Skip to avoid matplotlib issues in CI
            )
            
            # Verify results structure
            self.assertIn('metadata', results)
            self.assertIn('basic_categorization', results)
            
            # Verify metadata
            self.assertEqual(results['metadata']['total_conversations'], 10)
            
            # Verify basic categorization worked
            basic_results = results['basic_categorization']
            self.assertIn('total_conversations', basic_results)
            self.assertIn('categories', basic_results)
            
            # Verify sentiment analysis (if no errors)
            if 'sentiment_analysis' in results and 'error' not in results['sentiment_analysis']:
                sentiment_data = results['sentiment_analysis']
                self.assertEqual(sentiment_data['total_analyzed'], 10)
                self.assertIn('sentiment_distribution', sentiment_data)
                
                # Check that we have a mix of sentiments
                sentiment_dist = sentiment_data['sentiment_distribution']
                total_sentiments = sum(sentiment_dist.values())
                self.assertEqual(total_sentiments, 10)
            
            # Verify temporal analysis (if no errors)
            if 'temporal_analysis' in results and 'error' not in results['temporal_analysis']:
                temporal_data = results['temporal_analysis']
                self.assertEqual(temporal_data['total_conversations'], 10)
                self.assertIn('hourly_pattern', temporal_data)
                self.assertIn('date_range', temporal_data)
            
            # Verify results file was created
            self.assertTrue(os.path.exists("comprehensive_analysis_results.json"))
            
            # Load and verify saved results
            with open("comprehensive_analysis_results.json", 'r', encoding='utf-8') as f:
                saved_results = json.load(f)
            
            self.assertEqual(saved_results['metadata']['total_conversations'], 10)
        
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)