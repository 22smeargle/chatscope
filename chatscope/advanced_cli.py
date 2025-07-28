"""Advanced command-line interface for ChatScope with extended features."""

import argparse
import sys
import logging
from typing import Optional

from .advanced_analyzer import AdvancedChatGPTAnalyzer
from .exceptions import ChatGPTAnalyzerError


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Advanced ChatGPT Conversation Analyzer with sentiment analysis, topic modeling, and temporal analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comprehensive analysis
  chatscope-advanced conversations.json
  
  # Analysis with specific features
  chatscope-advanced conversations.json --sentiment --topics --temporal
  
  # Topic modeling with BERTopic
  chatscope-advanced conversations.json --topic-method BERTopic --num-topics 15
  
  # Sentiment analysis only
  chatscope-advanced conversations.json --sentiment-only
  
  # Custom output directory
  chatscope-advanced conversations.json --output-dir results/
"""
    )
    
    # Input/Output arguments
    parser.add_argument(
        "input_file",
        help="Path to the ChatGPT conversations JSON file"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Output directory for results and visualizations (default: current directory)"
    )
    
    parser.add_argument(
        "--results-file",
        default="comprehensive_analysis_results.json",
        help="Name of the results JSON file (default: comprehensive_analysis_results.json)"
    )
    
    # API Configuration
    api_group = parser.add_argument_group("API Configuration")
    api_group.add_argument(
        "--api-key",
        help="OpenAI API key (can also be set via OPENAI_API_KEY environment variable)"
    )
    
    api_group.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of titles to process in each API request (default: 20)"
    )
    
    api_group.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API requests in seconds (default: 1.0)"
    )
    
    api_group.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Maximum tokens per API request (default: 4000)"
    )
    
    # Analysis Features
    features_group = parser.add_argument_group("Analysis Features")
    features_group.add_argument(
        "--sentiment",
        action="store_true",
        help="Include sentiment analysis"
    )
    
    features_group.add_argument(
        "--topics",
        action="store_true",
        help="Include topic modeling"
    )
    
    features_group.add_argument(
        "--temporal",
        action="store_true",
        help="Include temporal analysis"
    )
    
    features_group.add_argument(
        "--all-features",
        action="store_true",
        help="Enable all analysis features (sentiment, topics, temporal)"
    )
    
    # Feature-specific options
    sentiment_group = parser.add_argument_group("Sentiment Analysis Options")
    sentiment_group.add_argument(
        "--sentiment-only",
        action="store_true",
        help="Perform only sentiment analysis (skip categorization)"
    )
    
    sentiment_group.add_argument(
        "--include-emotions",
        action="store_true",
        default=True,
        help="Include emotional tone analysis (default: True)"
    )
    
    topic_group = parser.add_argument_group("Topic Modeling Options")
    topic_group.add_argument(
        "--topic-method",
        choices=["LDA", "BERTopic", "clustering"],
        default="LDA",
        help="Topic modeling method (default: LDA)"
    )
    
    topic_group.add_argument(
        "--num-topics",
        type=int,
        default=10,
        help="Number of topics to extract (default: 10)"
    )
    
    temporal_group = parser.add_argument_group("Temporal Analysis Options")
    temporal_group.add_argument(
        "--time-granularity",
        choices=["hourly", "daily", "weekly", "monthly"],
        default="daily",
        help="Time granularity for temporal analysis (default: daily)"
    )
    
    temporal_group.add_argument(
        "--detect-trends",
        action="store_true",
        default=True,
        help="Detect trends in temporal data (default: True)"
    )
    
    # Visualization options
    viz_group = parser.add_argument_group("Visualization Options")
    viz_group.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip creating visualizations"
    )
    
    viz_group.add_argument(
        "--viz-format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Visualization format (default: png)"
    )
    
    # Categories
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Custom categories for classification (space-separated)"
    )
    
    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    return parser


def main():
    """Main entry point for the advanced CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    if args.quiet:
        logging.disable(logging.CRITICAL)
    else:
        setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize analyzer
        analyzer = AdvancedChatGPTAnalyzer(
            api_key=args.api_key,
            categories=args.categories,
            batch_size=args.batch_size,
            delay_between_requests=args.delay,
            max_tokens_per_request=args.max_tokens
        )
        
        # Determine which features to include
        if args.all_features:
            include_sentiment = True
            include_topics = True
            include_temporal = True
        elif args.sentiment_only:
            include_sentiment = True
            include_topics = False
            include_temporal = False
        else:
            include_sentiment = args.sentiment
            include_topics = args.topics
            include_temporal = args.temporal
            
            # If no specific features are requested, enable all by default
            if not any([include_sentiment, include_topics, include_temporal]):
                include_sentiment = True
                include_topics = True
                include_temporal = True
        
        # Perform analysis
        if args.sentiment_only:
            # Sentiment-only analysis
            logger.info("Performing sentiment analysis only...")
            conversations = analyzer.load_conversations(args.input_file)
            results = analyzer.analyze_sentiment(
                conversations, 
                include_emotional_tone=args.include_emotions
            )
            
            # Save results
            import json
            import os
            
            output_file = os.path.join(args.output_dir, "sentiment_analysis_results.json")
            os.makedirs(args.output_dir, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Sentiment analysis results saved to {output_file}")
            
        else:
            # Comprehensive analysis
            logger.info("Performing comprehensive analysis...")
            
            # Change to output directory
            import os
            original_cwd = os.getcwd()
            os.makedirs(args.output_dir, exist_ok=True)
            os.chdir(args.output_dir)
            
            try:
                results = analyzer.comprehensive_analysis(
                    input_file=os.path.join(original_cwd, args.input_file),
                    include_sentiment=include_sentiment,
                    include_topics=include_topics,
                    include_temporal=include_temporal,
                    topic_method=args.topic_method,
                    num_topics=args.num_topics,
                    create_visualizations=not args.no_visualizations
                )
                
                # Save results with custom filename
                if args.results_file != "comprehensive_analysis_results.json":
                    import json
                    with open(args.results_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                    
                    # Remove the default file if it exists
                    if os.path.exists("comprehensive_analysis_results.json"):
                        os.remove("comprehensive_analysis_results.json")
                    
                    results['results_file'] = args.results_file
                
                logger.info(f"Comprehensive analysis completed!")
                logger.info(f"Results saved to {results['results_file']}")
                
                if 'visualizations' in results and results['visualizations']:
                    logger.info("Visualizations created:")
                    for viz_type, path in results['visualizations'].items():
                        logger.info(f"  {viz_type}: {path}")
                
            finally:
                os.chdir(original_cwd)
        
        # Print summary
        if not args.quiet:
            print("\n" + "="*50)
            print("ANALYSIS SUMMARY")
            print("="*50)
            
            if args.sentiment_only:
                print(f"Sentiment Analysis Results:")
                print(f"  Total conversations analyzed: {results['total_analyzed']}")
                print(f"  Average polarity: {results['average_polarity']:.3f}")
                print(f"  Average subjectivity: {results['average_subjectivity']:.3f}")
                print(f"  Sentiment distribution: {results['sentiment_distribution']}")
                if 'emotion_distribution' in results:
                    print(f"  Emotion distribution: {results['emotion_distribution']}")
            else:
                print(f"Total conversations: {results['metadata']['total_conversations']}")
                
                if include_sentiment and 'sentiment_analysis' in results:
                    sentiment_data = results['sentiment_analysis']
                    if 'error' not in sentiment_data:
                        print(f"Sentiment analysis: {sentiment_data['total_analyzed']} conversations")
                        print(f"  Distribution: {sentiment_data['sentiment_distribution']}")
                
                if include_topics and 'topic_analysis' in results:
                    topic_data = results['topic_analysis']
                    if 'error' not in topic_data:
                        print(f"Topic modeling ({topic_data['method']}): {topic_data['num_topics']} topics")
                
                if include_temporal and 'temporal_analysis' in results:
                    temporal_data = results['temporal_analysis']
                    if 'error' not in temporal_data:
                        print(f"Temporal analysis: {temporal_data['total_conversations']} conversations")
                        print(f"  Date range: {temporal_data['date_range']['start']} to {temporal_data['date_range']['end']}")
            
            print("\nAnalysis completed successfully!")
    
    except ChatGPTAnalyzerError as e:
        logger.error(f"Analysis error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()