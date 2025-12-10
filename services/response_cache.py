"""
Response Cache Service

A provider-agnostic response caching service for LLM interactions.
Caches query-response pairs to reduce costs and improve latency.

Features:
- Conversation-aware caching (per-chat caching, no expensive history hashing)
- Model-aware caching (includes model_id in cache key)
- Configurable TTL (Time To Live)
- LRU eviction when cache is full
- Persistent storage (JSON format)
- Cache statistics and management
- Fast cache key generation (no history hashing overhead)
"""

import hashlib
import json
import os
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    from .context_aware_filter import ContextAwareFilter
except ImportError:
    from context_aware_filter import ContextAwareFilter


class ResponseCache:
    """
    Provider-agnostic response caching service (conversation-aware).
    
    Can be used with any LLM provider (Bedrock, OpenAI, Ollama, etc.)
    Caches query-response pairs to reduce costs and improve latency.
    
    Cache keys include:
    - Query text
    - Conversation ID (per-chat caching, fast!)
    - Model ID
    - Temperature and other parameters
    
    This ensures that:
    - Same query in same conversation = cache hit (fast!)
    - Same query in different conversation = different cache entry
    - Same query with different models = different cache entries
    - No expensive history hashing (much faster than context-aware caching)
    """
    
    def __init__(
        self,
        ttl_hours: int = 24,
        max_size: int = 1000,
        persistent: bool = True,
        cache_file: Optional[str] = None,
        context_filter: Optional[ContextAwareFilter] = None
    ):
        """
        Initialize response cache (conversation-aware, model-agnostic).
        
        Args:
            ttl_hours: Time to live in hours (default: 24)
            max_size: Maximum number of cached entries (default: 1000)
            persistent: Whether to persist cache to disk (default: True)
            cache_file: Path to cache file (auto-generated if None)
            context_filter: Optional ContextAwareFilter instance. If None, creates default.
        """
        self.cache: Dict[str, Dict] = {}
        self.ttl = ttl_hours * 3600  # Convert to seconds
        self.max_size = max_size
        self.persistent = persistent
        self.cache_file = cache_file
        if ContextAwareFilter is None:
            raise ImportError("ContextAwareFilter is required. Please ensure context_aware_filter.py is available.")
        self.context_filter = context_filter or ContextAwareFilter()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
        
        # Load from disk if persistent
        if persistent and cache_file:
            self._load_from_disk()
    
    def _hash_history(self, history: Optional[List[Dict]]) -> str:
        """
        DEPRECATED: No longer used. Kept for backward compatibility.
        Conversation-aware caching uses conversation_id instead of history hashing.
        """
        return "deprecated"
    
    def _generate_key(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        model_id: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate deterministic cache key (conversation-aware, not context-aware).
        
        Args:
            query: User query text
            conversation_id: Conversation/chat identifier (for per-chat caching)
            model_id: Model identifier
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            **kwargs: Other parameters that affect output
            
        Returns:
            Cache key (SHA256 hash)
        """
        # Use conversation_id for per-chat caching (much faster than hashing history)
        # If no conversation_id provided, use "default"
        conv_id = conversation_id or "default"
        
        # Build key data (only include parameters that affect output)
        key_data = {
            'query': query,
            'conversation_id': conv_id,  # Per-conversation caching
            'model': model_id,
            'temperature': round(temperature, 2),  # Round to avoid float precision issues
        }
        
        # Add optional parameters
        if max_tokens is not None:
            key_data['max_tokens'] = max_tokens
        
        # Add any other relevant parameters
        for key, value in kwargs.items():
            if value is not None and key not in ['debug', 'use_history', 'conversation_history']:  # Exclude non-deterministic params
                key_data[key] = value
        
        # Create deterministic JSON string
        key_str = json.dumps(key_data, sort_keys=True)
        
        # Generate hash
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        model_id: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Retrieve cached response (conversation-aware).
        
        Args:
            query: User query text
            conversation_id: Conversation/chat identifier (for per-chat caching)
            model_id: Model identifier (ignored - model-agnostic caching)
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            **kwargs: Other parameters
            
        Returns:
            Cached response if found and not expired, None otherwise
            Returns None for context-dependent queries (they should not be cached)
        """
        # Skip caching for context-dependent queries
        if self.context_filter.is_context_dependent(query):
            return None  # Don't use cache for context-dependent queries
        
        # Use "default" model_id to make it model-agnostic (same query = same cache entry regardless of model)
        cache_key = self._generate_key(
            query=query,
            conversation_id=conversation_id,
            model_id="default",  # Model-agnostic: same query in same chat = same cache entry
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            age = time.time() - entry['timestamp']
            
            if age < self.ttl:
                # Cache hit!
                self.stats['hits'] += 1
                return entry['response']
            else:
                # Expired, remove it
                del self.cache[cache_key]
        
        # Cache miss
        self.stats['misses'] += 1
        return None
    
    def set(
        self,
        query: str,
        response: str,
        conversation_id: Optional[str] = None,
        model_id: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Store response in cache (conversation-aware, model-agnostic).
        
        Args:
            query: User query text
            response: AI response text
            conversation_id: Conversation/chat identifier (for per-chat caching)
            model_id: Model identifier (ignored - model-agnostic caching)
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            **kwargs: Other parameters
        """
        # Skip caching for context-dependent queries
        if self.context_filter.is_context_dependent(query):
            return  # Don't cache context-dependent queries
        
        # Use "default" model_id to make it model-agnostic (same query = same cache entry regardless of model)
        cache_key = self._generate_key(
            query=query,
            conversation_id=conversation_id,
            model_id="default",  # Model-agnostic: same query in same chat = same cache entry
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Evict oldest if cache is full (LRU)
        if len(self.cache) >= self.max_size:
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k]['timestamp']
            )
            del self.cache[oldest_key]
            self.stats['evictions'] += 1
        
        # Store entry
        self.cache[cache_key] = {
            'response': response,
            'timestamp': time.time(),
            'query': query,  # Store for debugging/inspection
            'conversation_id': conversation_id or "default"
        }
        
        self.stats['sets'] += 1
        
        # Persist if enabled
        if self.persistent:
            self._save_to_disk()
    
    def clear(self) -> None:
        """Clear all cached responses."""
        self.cache.clear()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
        if self.persistent:
            self._save_to_disk()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        now = time.time()
        valid_entries = sum(
            1 for entry in self.cache.values()
            if now - entry['timestamp'] < self.ttl
        )
        
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.cache) - valid_entries,
            'max_size': self.max_size,
            'ttl_hours': self.ttl / 3600,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'sets': self.stats['sets'],
            'evictions': self.stats['evictions']
        }
    
    def _save_to_disk(self) -> None:
        """Save cache to disk."""
        if not self.cache_file:
            return
        
        try:
            # Ensure directory exists
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir:  # Only create dir if path includes a directory
                os.makedirs(cache_dir, exist_ok=True)
            
            data = {
                'version': '1.0',
                'metadata': {
                    'ttl': self.ttl,
                    'max_size': self.max_size,
                    'saved_at': datetime.now().isoformat()
                },
                'entries': self.cache,
                'stats': self.stats
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Log error but don't fail (non-critical)
            import logging
            logging.warning(f"Failed to save cache to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.cache_file or not os.path.exists(self.cache_file):
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Load entries
                self.cache = data.get('entries', {})
                
                # Load stats if available
                if 'stats' in data:
                    self.stats = data['stats']
                
                # Validate and clean expired entries
                now = time.time()
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if now - entry['timestamp'] >= self.ttl
                ]
                for key in expired_keys:
                    del self.cache[key]
                
        except Exception as e:
            # Don't fail if cache load fails (start fresh)
            self.cache = {}
            self.stats = {
                'hits': 0,
                'misses': 0,
                'sets': 0,
                'evictions': 0
            }

