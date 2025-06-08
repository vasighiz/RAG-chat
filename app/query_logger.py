import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from loguru import logger

class QueryLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "query_history.json"
        self._initialize_log_file()
        
    def _initialize_log_file(self):
        """Initialize log file if it doesn't exist."""
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                json.dump([], f)
                
    def log_query(
        self,
        query: str,
        response: str,
        context: List[Dict],
        metadata: Dict = None
    ):
        """Log a query and its response."""
        try:
            # Load existing logs
            with open(self.log_file, "r") as f:
                logs = json.load(f)
                
            # Create new log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "context": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in context
                ],
                "metadata": metadata or {}
            }
            
            # Append new log
            logs.append(log_entry)
            
            # Save updated logs
            with open(self.log_file, "w") as f:
                json.dump(logs, f, indent=2)
                
            logger.info(f"Logged query: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error logging query: {str(e)}")
            
    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent queries from log file."""
        try:
            with open(self.log_file, "r") as f:
                logs = json.load(f)
            return logs[-limit:]
        except Exception as e:
            logger.error(f"Error reading query logs: {str(e)}")
            return [] 