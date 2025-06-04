"""
Session Management for Interactive Analysis

Provides persistent analysis sessions with history tracking,
context management, and collaborative features.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class SessionStatus(Enum):
    """Analysis session status types"""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    ERROR = "error"


class InteractionType(Enum):
    """Types of user interactions"""

    QUERY = "query"
    COMMAND = "command"
    VISUALIZATION = "visualization"
    MODEL_UPLOAD = "model_upload"
    RESULT_EXPORT = "result_export"
    SESSION_SAVE = "session_save"


@dataclass
class Interaction:
    """Single user interaction within a session"""

    id: str
    timestamp: datetime
    type: InteractionType
    input_data: str
    output_data: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert interaction to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.type.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Interaction":
        """Create interaction from dictionary"""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            type=InteractionType(data["type"]),
            input_data=data["input_data"],
            output_data=data.get("output_data"),
            metadata=data.get("metadata", {}),
            execution_time=data.get("execution_time", 0.0),
            success=data.get("success", True),
            error_message=data.get("error_message"),
        )


@dataclass
class AnalysisSession:
    """Interactive analysis session with full context tracking"""

    id: str
    name: str
    created_at: datetime
    last_active: datetime
    status: SessionStatus = SessionStatus.ACTIVE
    description: str = ""
    model_files: List[str] = field(default_factory=list)
    interactions: List[Interaction] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Session statistics
    total_queries: int = 0
    total_execution_time: float = 0.0
    success_rate: float = 1.0

    def add_interaction(self, interaction: Interaction) -> None:
        """Add a new interaction to the session"""
        self.interactions.append(interaction)
        self.last_active = datetime.now()
        self.total_queries += 1
        self.total_execution_time += interaction.execution_time

        # Update success rate
        successful_interactions = sum(1 for i in self.interactions if i.success)
        self.success_rate = successful_interactions / len(self.interactions)

        # Update context with latest interaction
        self.context["last_interaction"] = interaction.to_dict()
        self.context["last_query"] = interaction.input_data

    def get_recent_interactions(self, count: int = 5) -> List[Interaction]:
        """Get the most recent interactions"""
        return self.interactions[-count:] if self.interactions else []

    def get_conversation_history(
        self, include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """Get formatted conversation history"""
        history = []
        for interaction in self.interactions:
            if interaction.type in [InteractionType.QUERY, InteractionType.COMMAND]:
                entry = {
                    "timestamp": interaction.timestamp.strftime("%H:%M:%S"),
                    "user": interaction.input_data,
                    "assistant": interaction.output_data or "No response recorded",
                    "success": interaction.success,
                }
                if include_metadata:
                    entry["metadata"] = interaction.metadata
                history.append(entry)
        return history

    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        duration = self.last_active - self.created_at

        # Analyze interaction types
        type_counts = {}
        for interaction in self.interactions:
            type_name = interaction.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "session_info": {
                "id": self.id,
                "name": self.name,
                "status": self.status.value,
                "duration": str(duration),
                "model_files": len(self.model_files),
            },
            "statistics": {
                "total_interactions": len(self.interactions),
                "total_queries": self.total_queries,
                "success_rate": f"{self.success_rate:.1%}",
                "total_execution_time": f"{self.total_execution_time:.2f}s",
                "avg_execution_time": f"{self.total_execution_time / max(1, len(self.interactions)):.2f}s",
            },
            "interaction_breakdown": type_counts,
            "context_size": len(self.context),
            "recent_activity": [
                i.input_data[:50] + "..." if len(i.input_data) > 50 else i.input_data
                for i in self.get_recent_interactions(3)
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "status": self.status.value,
            "description": self.description,
            "model_files": self.model_files,
            "interactions": [i.to_dict() for i in self.interactions],
            "context": self.context,
            "metadata": self.metadata,
            "total_queries": self.total_queries,
            "total_execution_time": self.total_execution_time,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisSession":
        """Create session from dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]),
            status=SessionStatus(data["status"]),
            description=data.get("description", ""),
            model_files=data.get("model_files", []),
            interactions=[
                Interaction.from_dict(i) for i in data.get("interactions", [])
            ],
            context=data.get("context", {}),
            metadata=data.get("metadata", {}),
            total_queries=data.get("total_queries", 0),
            total_execution_time=data.get("total_execution_time", 0.0),
            success_rate=data.get("success_rate", 1.0),
        )


class SessionManager:
    """Manages multiple analysis sessions with persistence and search"""

    def __init__(self, sessions_dir: str = "sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.active_sessions: Dict[str, AnalysisSession] = {}
        self.current_session: Optional[AnalysisSession] = None

    def create_session(self, name: str, description: str = "") -> AnalysisSession:
        """Create a new analysis session"""
        session_id = str(uuid.uuid4())[:8]
        session = AnalysisSession(
            id=session_id,
            name=name,
            created_at=datetime.now(),
            last_active=datetime.now(),
            description=description,
        )

        self.active_sessions[session_id] = session
        self.current_session = session
        self.save_session(session)

        console.print(
            f"✅ Created new session: [bold cyan]{name}[/bold cyan] (ID: {session_id})"
        )
        return session

    def load_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Load a session from disk"""
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            console.print(f"❌ Session {session_id} not found")
            return None

        try:
            with open(session_file, "r") as f:
                data = json.load(f)

            session = AnalysisSession.from_dict(data)
            self.active_sessions[session_id] = session

            console.print(f"✅ Loaded session: [bold cyan]{session.name}[/bold cyan]")
            return session

        except Exception as e:
            console.print(f"❌ Error loading session {session_id}: {e}")
            return None

    def save_session(self, session: AnalysisSession) -> bool:
        """Save a session to disk"""
        session_file = self.sessions_dir / f"{session.id}.json"

        try:
            with open(session_file, "w") as f:
                json.dump(session.to_dict(), f, indent=2, default=str)
            return True

        except Exception as e:
            console.print(f"❌ Error saving session {session.id}: {e}")
            return False

    def set_current_session(self, session_id: str) -> bool:
        """Set the current active session"""
        if session_id in self.active_sessions:
            self.current_session = self.active_sessions[session_id]
            console.print(
                f"✅ Switched to session: [bold cyan]{self.current_session.name}[/bold cyan]"
            )
            return True

        # Try to load from disk
        session = self.load_session(session_id)
        if session:
            self.current_session = session
            return True

        return False

    def list_sessions(
        self, status_filter: Optional[SessionStatus] = None
    ) -> List[AnalysisSession]:
        """List all available sessions"""
        sessions = []

        # Get sessions from memory
        sessions.extend(self.active_sessions.values())

        # Get sessions from disk
        for session_file in self.sessions_dir.glob("*.json"):
            session_id = session_file.stem
            if session_id not in self.active_sessions:
                session = self.load_session(session_id)
                if session:
                    sessions.append(session)

        # Filter by status if specified
        if status_filter:
            sessions = [s for s in sessions if s.status == status_filter]

        # Sort by last active time
        sessions.sort(key=lambda s: s.last_active, reverse=True)
        return sessions

    def search_sessions(self, query: str) -> List[AnalysisSession]:
        """Search sessions by name, description, or content"""
        query_lower = query.lower()
        matching_sessions = []

        for session in self.list_sessions():
            # Search in name and description
            if (
                query_lower in session.name.lower()
                or query_lower in session.description.lower()
            ):
                matching_sessions.append(session)
                continue

            # Search in interaction content
            for interaction in session.interactions:
                if query_lower in interaction.input_data.lower():
                    matching_sessions.append(session)
                    break

        return matching_sessions

    def display_sessions_table(
        self, sessions: Optional[List[AnalysisSession]] = None
    ) -> None:
        """Display sessions in a beautiful table format"""
        if sessions is None:
            sessions = self.list_sessions()

        if not sessions:
            console.print("[yellow]No sessions found.[/yellow]")
            return

        table = Table(title="🔬 Analysis Sessions", box=box.ROUNDED)
        table.add_column("ID", style="bold cyan")
        table.add_column("Name", style="bold white")
        table.add_column("Status", style="bold green")
        table.add_column("Interactions", style="bold yellow")
        table.add_column("Last Active", style="bold blue")
        table.add_column("Success Rate", style="bold magenta")

        for session in sessions:
            # Format status with color
            status_color = {
                SessionStatus.ACTIVE: "green",
                SessionStatus.PAUSED: "yellow",
                SessionStatus.COMPLETED: "blue",
                SessionStatus.ARCHIVED: "dim",
                SessionStatus.ERROR: "red",
            }.get(session.status, "white")

            # Format last active time
            time_diff = datetime.now() - session.last_active
            if time_diff < timedelta(hours=1):
                last_active = f"{int(time_diff.total_seconds() // 60)}m ago"
            elif time_diff < timedelta(days=1):
                last_active = f"{int(time_diff.total_seconds() // 3600)}h ago"
            else:
                last_active = f"{time_diff.days}d ago"

            # Current session indicator
            name = session.name
            if self.current_session and session.id == self.current_session.id:
                name = f"🎯 {name}"

            table.add_row(
                session.id,
                name,
                f"[{status_color}]{session.status.value}[/{status_color}]",
                str(len(session.interactions)),
                last_active,
                f"{session.success_rate:.1%}",
            )

        console.print(table)

    def archive_old_sessions(self, days_old: int = 30) -> int:
        """Archive sessions older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        archived_count = 0

        for session in self.list_sessions():
            if (
                session.last_active < cutoff_date
                and session.status != SessionStatus.ARCHIVED
            ):
                session.status = SessionStatus.ARCHIVED
                self.save_session(session)
                archived_count += 1

        console.print(f"✅ Archived {archived_count} old sessions")
        return archived_count

    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics across all sessions"""
        sessions = self.list_sessions()

        if not sessions:
            return {"total_sessions": 0}

        total_interactions = sum(len(s.interactions) for s in sessions)
        total_time = sum(s.total_execution_time for s in sessions)

        # Status distribution
        status_counts = {}
        for session in sessions:
            status = session.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Average success rate
        avg_success_rate = sum(s.success_rate for s in sessions) / len(sessions)

        return {
            "total_sessions": len(sessions),
            "total_interactions": total_interactions,
            "total_execution_time": total_time,
            "average_session_length": total_interactions / len(sessions),
            "average_success_rate": avg_success_rate,
            "status_distribution": status_counts,
            "most_active_session": max(
                sessions, key=lambda s: len(s.interactions)
            ).name,
            "latest_session": max(sessions, key=lambda s: s.last_active).name,
        }

    def display_analytics(self) -> None:
        """Display comprehensive session analytics"""
        analytics = self.get_session_analytics()

        if analytics["total_sessions"] == 0:
            console.print("[yellow]No session data available.[/yellow]")
            return

        # Create analytics panel
        analytics_table = Table(show_header=False, box=box.SIMPLE)
        analytics_table.add_column("Metric", style="bold cyan")
        analytics_table.add_column("Value", style="bold white")

        analytics_table.add_row("Total Sessions", str(analytics["total_sessions"]))
        analytics_table.add_row(
            "Total Interactions", str(analytics["total_interactions"])
        )
        analytics_table.add_row(
            "Total Execution Time", f"{analytics['total_execution_time']:.1f}s"
        )
        analytics_table.add_row(
            "Avg Session Length",
            f"{analytics['average_session_length']:.1f} interactions",
        )
        analytics_table.add_row(
            "Average Success Rate", f"{analytics['average_success_rate']:.1%}"
        )
        analytics_table.add_row("Most Active Session", analytics["most_active_session"])
        analytics_table.add_row("Latest Session", analytics["latest_session"])

        console.print(
            Panel(
                analytics_table,
                title="[bold blue]📊 Session Analytics[/bold blue]",
                border_style="blue",
            )
        )
