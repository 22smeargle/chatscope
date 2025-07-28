# ChatScope Documentation 📚

Welcome to the official documentation for **ChatScope 2.0** - a powerful Python library that transforms your ChatGPT conversation exports into actionable insights through advanced analytics, sentiment analysis, topic modeling, and temporal pattern detection.

## 🚀 Quick Navigation

### Core Documentation
- **[API Reference](API_REFERENCE.md)** - Complete API documentation for all classes and methods
- **[User Guide](USER_GUIDE.md)** - Comprehensive guide for getting started and advanced usage
- **[Examples](EXAMPLES.md)** - Practical code examples and real-world use cases

### Quick Links
- **[GitHub Repository](https://github.com/22smeargle/chatscope)** - Source code and issues
- **[PyPI Package](https://pypi.org/project/chatscope/)** - Install via pip
- **[License](https://github.com/22smeargle/chatscope/blob/main/LICENSE)** - MIT License

## 🎯 What is ChatScope?

ChatScope analyzes your ChatGPT conversation exports to provide:

- **📊 Advanced Analytics** - Conversation categorization and statistics
- **💭 Sentiment Analysis** - Emotion detection and polarity analysis
- **🎯 Topic Modeling** - Discover themes using LDA, BERTopic, and clustering
- **⏰ Temporal Analysis** - Usage patterns and activity trends
- **📈 Interactive Visualizations** - Beautiful charts and graphs
- **🔧 CLI Tools** - Command-line interface for batch processing

## ⚡ Quick Start

### Installation
```bash
pip install chatscope
```

### Basic Usage
```python
from chatscope import ChatGPTAnalyzer

analyzer = ChatGPTAnalyzer()
results = analyzer.analyze('conversations.json')
print(f"Found {results['total_conversations']} conversations")
```

### Advanced Analysis
```python
from chatscope import AdvancedChatGPTAnalyzer

analyzer = AdvancedChatGPTAnalyzer()
results = analyzer.comprehensive_analysis('conversations.json')
print(f"Average sentiment: {results['sentiment_analysis']['average_polarity']:.2f}")
```

## 📋 Features Overview

### Basic Features
- ✅ Conversation categorization
- ✅ Export to JSON/CSV
- ✅ Basic statistics
- ✅ Command-line interface

### Advanced Features (v2.0+)
- ✅ Sentiment analysis with emotion detection
- ✅ Multiple topic modeling algorithms
- ✅ Temporal pattern analysis
- ✅ Interactive Plotly visualizations
- ✅ Word clouds and advanced charts
- ✅ Comprehensive CLI with advanced options

## 🛠️ Requirements

- **Python 3.8+**
- **OpenAI API key** (for categorization)
- **Optional dependencies** for advanced features:
  - `scikit-learn` - Topic modeling and clustering
  - `textblob` - Sentiment analysis
  - `nltk` - Natural language processing
  - `plotly` - Interactive visualizations
  - `wordcloud` - Word cloud generation
  - `bertopic` - Advanced topic modeling

## 📖 Documentation Sections

### For Beginners
1. **[Installation Guide](USER_GUIDE.md#installation)** - Get started quickly
2. **[Basic Examples](EXAMPLES.md#quick-start-examples)** - Simple use cases
3. **[CLI Usage](USER_GUIDE.md#command-line-interface)** - Command-line tools

### For Advanced Users
1. **[Advanced Analysis](USER_GUIDE.md#advanced-analysis)** - Sentiment, topics, temporal
2. **[API Reference](API_REFERENCE.md)** - Complete method documentation
3. **[Integration Examples](EXAMPLES.md#integration-examples)** - Web apps, dashboards

### For Developers
1. **[Contributing](https://github.com/22smeargle/chatscope#contributing)** - How to contribute
2. **[API Design](API_REFERENCE.md#design-principles)** - Architecture overview
3. **[Testing](https://github.com/22smeargle/chatscope#running-tests)** - Test suite information

## 🎨 Visualization Gallery

ChatScope generates beautiful visualizations:

- **Category Distribution** - Pie charts and bar graphs
- **Sentiment Timeline** - Mood changes over time
- **Topic Clusters** - Interactive scatter plots
- **Activity Heatmaps** - Usage patterns
- **Word Clouds** - Most common terms
- **Conversation Flow** - Message patterns

## 🔧 Configuration

ChatScope can be configured via:
- **Environment variables** - API keys and settings
- **Configuration files** - JSON/YAML config
- **Command-line arguments** - Runtime options
- **Python API** - Programmatic configuration

## 🆘 Support

Need help? Here's where to find it:

1. **[User Guide](USER_GUIDE.md#troubleshooting)** - Common issues and solutions
2. **[GitHub Issues](https://github.com/22smeargle/chatscope/issues)** - Bug reports and feature requests
3. **[Examples](EXAMPLES.md)** - Code samples for common tasks
4. **[API Reference](API_REFERENCE.md)** - Detailed method documentation

## 📈 Version History

### Version 2.0.0 (Latest)
- ✨ Advanced sentiment analysis with emotion detection
- 🎯 Multiple topic modeling algorithms (LDA, BERTopic, K-means)
- ⏰ Comprehensive temporal pattern analysis
- 📊 Interactive visualizations with Plotly
- 🎨 Enhanced CLI with advanced options
- 📈 Performance improvements and better error handling

### Version 1.0.0
- 🎯 Basic conversation categorization
- 📊 Category visualization
- 🔧 Command-line interface
- 📁 JSON export functionality

## 🤝 Contributing

We welcome contributions! See our [Contributing Guidelines](https://github.com/22smeargle/chatscope#contributing) for details.

## 📄 License

ChatScope is released under the [MIT License](https://github.com/22smeargle/chatscope/blob/main/LICENSE).

---

*Made with ❤️ for the ChatGPT community*

**Last updated:** December 2024  
**Version:** 2.0.0  
**Maintainer:** [22smeargle](https://github.com/22smeargle)