"""Session management utilities."""

import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading

from ..models.schemas import ChatMessage, SessionInfo
from ..config import config

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages chat sessions and their history."""
    
    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, List[ChatMessage]] = {}
        self.session_metadata: Dict[str, dict] = {}
        self.lock = threading.Lock()
        logger.info("Session Manager initialized")
    
    def create_session(self, session_id: str) -> None:
        """
        Create a new chat session.
        
        Args:
            session_id: Unique session identifier
        """
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
                self.session_metadata[session_id] = {
                    "created_at": datetime.now(),
                    "last_updated": datetime.now()
                }
                logger.info(f"Created new session: {session_id}")
    
    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """
        Add a message to a session.
        
        Args:
            session_id: Session identifier
            message: Chat message to add
        """
        with self.lock:
            # Create session if it doesn't exist
            if session_id not in self.sessions:
                self.create_session(session_id)
            
            # Add timestamp to message if not present
            if not message.timestamp:
                message.timestamp = datetime.now().isoformat()
            
            # Add message to session
            self.sessions[session_id].append(message)
            
            # Update metadata
            self.session_metadata[session_id]["last_updated"] = datetime.now()
            
            logger.debug(f"Added message to session {session_id}: {message.role} - {len(message.content)} chars")
    
    def get_session_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """
        Get chat history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of chat messages
        """
        with self.lock:
            if session_id not in self.sessions:
                return []
            
            messages = self.sessions[session_id]
            
            if limit and limit > 0:
                return messages[-limit:]
            
            return messages.copy()
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages from a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cleared, False if session didn't exist
        """
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].clear()
                self.session_metadata[session_id]["last_updated"] = datetime.now()
                logger.info(f"Cleared session: {session_id}")
                return True
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session completely.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was deleted, False if session didn't exist
        """
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                del self.session_metadata[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
            return False
    
    def get_active_sessions(self) -> List[SessionInfo]:
        """
        Get information about all active sessions.
        
        Returns:
            List of session information
        """
        with self.lock:
            session_infos = []
            
            for session_id, messages in self.sessions.items():
                metadata = self.session_metadata.get(session_id, {})
                
                session_info = SessionInfo(
                    session_id=session_id,
                    message_count=len(messages),
                    created_at=metadata.get("created_at", "").isoformat() if isinstance(metadata.get("created_at"), datetime) else None,
                    last_updated=metadata.get("last_updated", "").isoformat() if isinstance(metadata.get("last_updated"), datetime) else None
                )
                
                session_infos.append(session_info)
            
            # Sort by last updated (most recent first)
            session_infos.sort(
                key=lambda x: x.last_updated if x.last_updated else "",
                reverse=True
            )
            
            return session_infos
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old inactive sessions.
        
        Args:
            max_age_hours: Maximum age in hours for keeping sessions
            
        Returns:
            Number of sessions cleaned up
        """
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            sessions_to_delete = []
            
            for session_id, metadata in self.session_metadata.items():
                last_updated = metadata.get("last_updated")
                if isinstance(last_updated, datetime) and last_updated < cutoff_time:
                    sessions_to_delete.append(session_id)
            
            # Delete old sessions
            for session_id in sessions_to_delete:
                del self.sessions[session_id]
                del self.session_metadata[session_id]
            
            if sessions_to_delete:
                logger.info(f"Cleaned up {len(sessions_to_delete)} old sessions")
            
            return len(sessions_to_delete)
    
    def get_session_stats(self) -> dict:
        """
        Get statistics about sessions.
        
        Returns:
            Dictionary with session statistics
        """
        with self.lock:
            total_sessions = len(self.sessions)
            total_messages = sum(len(messages) for messages in self.sessions.values())
            
            active_sessions = 0
            recent_cutoff = datetime.now() - timedelta(hours=1)
            
            for metadata in self.session_metadata.values():
                last_updated = metadata.get("last_updated")
                if isinstance(last_updated, datetime) and last_updated > recent_cutoff:
                    active_sessions += 1
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "total_messages": total_messages,
                "average_messages_per_session": total_messages / max(total_sessions, 1)
            }
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists, False otherwise
        """
        with self.lock:
            return session_id in self.sessions
    
    def get_recent_messages(self, session_id: str, count: int = 10) -> List[ChatMessage]:
        """
        Get recent messages from a session for context.
        
        Args:
            session_id: Session identifier
            count: Number of recent messages to get
            
        Returns:
            List of recent chat messages
        """
        return self.get_session_history(session_id, limit=count)

# Global session manager instance
session_manager = SessionManager()