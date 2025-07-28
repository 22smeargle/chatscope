"""ChatScope - Analyze and categorize ChatGPT conversation exports."""

from .analyzer import ChatGPTAnalyzer
from .advanced_analyzer import AdvancedChatGPTAnalyzer
from .exceptions import ChatGPTAnalyzerError, APIError, DataError, ConfigurationError

__version__ = "2.0.0"
__all__ = [
    "ChatGPTAnalyzer", 
    "AdvancedChatGPTAnalyzer",
    "ChatGPTAnalyzerError", 
    "APIError", 
    "DataError", 
    "ConfigurationError"
]