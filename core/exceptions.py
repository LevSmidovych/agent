class AgentError(Exception):
    """Base exception for agent errors."""


class OllamaConnectionError(AgentError):
    """Raised when Ollama is unreachable or returns an error."""


class MigrationError(AgentError):
    """Raised when a SQLite migration fails."""


class SingleInstanceError(AgentError):
    """Raised when another instance of the app is already running."""
