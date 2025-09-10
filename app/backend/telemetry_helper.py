import hashlib
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

class SimpleTelemetryHelper:
    """Simple telemetry helper that logs metrics using existing Application Insights setup"""
    
    def __init__(self):
        self.query_count = {}
        self.unique_users = set()
        
    def track_request(self, endpoint: str, user_id: str, query: str = None, tokens: int = None, duration_ms: float = None):
        """Track a single request with minimal overhead"""
        try:
            # Hash user ID for privacy
            hashed_user = hashlib.sha256(user_id.encode()).hexdigest()[:16] if user_id else "anonymous"
            self.unique_users.add(hashed_user)
            
            # Track query frequency if provided
            if query:
                query_key = query[:100].lower().strip()
                self.query_count[query_key] = self.query_count.get(query_key, 0) + 1
            
            # Log structured data that Application Insights will pick up
            logger.info(
                "telemetry_event",
                extra={
                    "custom_dimensions": {
                        "event_type": "request",
                        "endpoint": endpoint,
                        "user_hash": hashed_user,
                        "has_query": bool(query),
                        "tokens": tokens or 0,
                        "duration_ms": duration_ms or 0,
                        "unique_user_count": len(self.unique_users),
                        "top_query": max(self.query_count.items(), key=lambda x: x[1])[0] if self.query_count else None
                    }
                }
            )
        except Exception as e:
            # Never let telemetry break the main flow
            logger.debug(f"Telemetry error (non-critical): {e}")
    
    def get_stats(self):
        """Get current statistics"""
        return {
            "unique_users": len(self.unique_users),
            "total_queries": sum(self.query_count.values()),
            "top_queries": sorted(self.query_count.items(), key=lambda x: x[1], reverse=True)[:5]
        }

# Global instance
_telemetry_instance = None

def get_telemetry():
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = SimpleTelemetryHelper()
    return _telemetry_instance