"""
Context-Aware Query Filter

A service for detecting context-dependent queries that should not be cached.
Context-dependent queries are those whose answers may change based on 
conversation history (e.g., "what is my name?").

This class is designed to be extensible - you can easily add new patterns
or modify detection logic without touching the cache implementation.
"""

import re
from typing import List, Optional


class ContextAwareFilter:
    """
    Detects context-dependent queries that should not be cached.
    
    Context-dependent queries are those where the answer depends on
    what was said earlier in the conversation. These should NOT be cached
    because the answer may change as the conversation progresses.
    
    Examples:
    - "what is my name?" (depends on whether name was mentioned)
    - "what did I say?" (depends on conversation history)
    - "remember that..." (depends on context)
    """
    
    def __init__(self, patterns: Optional[List[str]] = None):
        """
        Initialize the context-aware filter.
        
        Args:
            patterns: Optional list of regex patterns. If None, uses default patterns.
        """
        if patterns is None:
            patterns = self._get_default_patterns()
        
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _get_default_patterns(self) -> List[str]:
        """
        Get default regex patterns for context-dependent queries.
        
        Returns:
            List of regex pattern strings
        """
        return [
            # Personal information queries
            r'\bmy\s+(name|age|location|favorite|email|phone|address|job|work|company|color|food|hobby)\b',
            r'\bwhat\s+is\s+my\b',  # "what is my name?", "what is my age?", etc.
            
            # Self-referential queries
            r'\bwho\s+am\s+I\b[?]?',
            r'\bwhat\s+(am\s+I|are\s+we|is\s+my|was\s+my)\b',
            
            # Conversation history queries
            r'\bwhat\s+(did\s+I|did\s+we|did\s+I\s+say|did\s+I\s+tell|did\s+I\s+mention|do\s+I)\b[?]?',
            r'\bwhat\s+did\s+I\b',  # "what did I say?", "what did I ask?", etc.
            r'\b(what|which|where)\s+(did|do|does)\s+I\b',
            
            # Memory/recall queries
            r'\b(remember|recall|remind\s+me)\b.*\b(that|what|when|where|who)\b',
            
            # About me queries
            r'\b(tell\s+me|what\s+do\s+you\s+know)\s+about\s+(me|myself|us)\b',
            
            # Conversation summary queries
            r'\b(summarize|summarise)\s+(what|our)\s+(conversation|chat|discussion)\b',
        ]
    
    def is_context_dependent(self, query: str) -> bool:
        """
        Check if a query is context-dependent.
        
        Args:
            query: User query text
            
        Returns:
            True if query is likely context-dependent, False otherwise
        """
        query_clean = query.strip()
        
        # Check if query matches any context-dependent pattern
        for pattern in self.patterns:
            if pattern.search(query_clean):
                return True
        
        return False
    
    def add_pattern(self, pattern: str) -> None:
        """
        Add a custom pattern for context-dependent query detection.
        
        Args:
            pattern: Regex pattern string
        """
        self.patterns.append(re.compile(pattern, re.IGNORECASE))
    
    def remove_pattern(self, pattern: str) -> None:
        """
        Remove a pattern from the filter.
        
        Args:
            pattern: Regex pattern string to remove
        """
        pattern_obj = re.compile(pattern, re.IGNORECASE)
        self.patterns = [p for p in self.patterns if p.pattern != pattern_obj.pattern]
    
    def get_patterns(self) -> List[str]:
        """
        Get all current patterns.
        
        Returns:
            List of pattern strings
        """
        return [p.pattern for p in self.patterns]

