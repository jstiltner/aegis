"""
Authentication delegation for tool access.

This module provides:
- Credential management without exposure
- Scoped token generation
- Auth provider abstraction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, SecretStr


class AuthScope(str, Enum):
    """Predefined authentication scopes."""
    
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    
    # Tool-specific scopes
    FILE_READ = "file:read"
    FILE_WRITE = "file:write"
    WEB_READ = "web:read"
    WEB_WRITE = "web:write"
    CODE_EXECUTE = "code:execute"
    API_CALL = "api:call"


class Credential(BaseModel):
    """
    A stored credential.
    
    Credentials are stored securely and never exposed directly.
    Instead, scoped tokens are generated for tool access.
    """
    
    model_config = ConfigDict(frozen=True)
    
    credential_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this credential",
    )
    name: str = Field(
        ...,
        description="Human-readable name for the credential",
    )
    credential_type: str = Field(
        ...,
        description="Type of credential (api_key, oauth, basic, etc.)",
    )
    
    # The actual secret (stored securely)
    secret: SecretStr = Field(
        ...,
        description="The secret value",
    )
    
    # Metadata
    description: str = Field(
        default="",
        description="Description of what this credential is for",
    )
    
    # Scopes this credential grants
    scopes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Scopes this credential grants",
    )
    
    # Expiration
    expires_at: datetime | None = Field(
        default=None,
        description="When this credential expires",
    )
    
    # Usage tracking
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this credential was created",
    )
    last_used_at: datetime | None = Field(
        default=None,
        description="When this credential was last used",
    )
    
    @property
    def is_expired(self) -> bool:
        """Check if the credential has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def has_scope(self, scope: str) -> bool:
        """Check if the credential has a specific scope."""
        # Check exact match
        if scope in self.scopes:
            return True
        
        # Check wildcard scopes
        for s in self.scopes:
            if s == "*":
                return True
            if s.endswith(":*"):
                prefix = s[:-1]  # Remove the *
                if scope.startswith(prefix):
                    return True
        
        return False
    
    def has_all_scopes(self, scopes: list[str]) -> bool:
        """Check if the credential has all specified scopes."""
        return all(self.has_scope(s) for s in scopes)


class ScopedToken(BaseModel):
    """
    A scoped token derived from a credential.
    
    Tokens are short-lived and have limited scopes.
    They are what gets passed to tools, not the original credentials.
    """
    
    model_config = ConfigDict(frozen=True)
    
    token_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this token",
    )
    credential_id: str = Field(
        ...,
        description="ID of the credential this token was derived from",
    )
    
    # The token value (derived from credential)
    token: SecretStr = Field(
        ...,
        description="The token value",
    )
    
    # Scopes (subset of credential scopes)
    scopes: tuple[str, ...] = Field(
        ...,
        description="Scopes granted by this token",
    )
    
    # Lifetime
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this token was created",
    )
    expires_at: datetime = Field(
        ...,
        description="When this token expires",
    )
    
    # Context
    tool_name: str | None = Field(
        default=None,
        description="Tool this token is scoped to",
    )
    agent_id: str | None = Field(
        default=None,
        description="Agent this token is scoped to",
    )
    
    @property
    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def remaining_seconds(self) -> float:
        """Get remaining lifetime in seconds."""
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, delta.total_seconds())
    
    def has_scope(self, scope: str) -> bool:
        """Check if the token has a specific scope."""
        return scope in self.scopes


class AuthProvider(ABC):
    """
    Abstract base class for authentication providers.
    
    Auth providers handle the actual authentication with external services.
    """
    
    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Get the provider type identifier."""
        ...
    
    @abstractmethod
    async def authenticate(
        self,
        credential: Credential,
        scopes: list[str],
    ) -> ScopedToken:
        """
        Authenticate and get a scoped token.
        
        Args:
            credential: The credential to use
            scopes: Requested scopes
            
        Returns:
            A scoped token
        """
        ...
    
    @abstractmethod
    async def validate_token(self, token: ScopedToken) -> bool:
        """
        Validate a token is still valid.
        
        Args:
            token: The token to validate
            
        Returns:
            True if valid
        """
        ...
    
    @abstractmethod
    async def revoke_token(self, token: ScopedToken) -> bool:
        """
        Revoke a token.
        
        Args:
            token: The token to revoke
            
        Returns:
            True if successfully revoked
        """
        ...


class SimpleAuthProvider(AuthProvider):
    """
    Simple auth provider that passes through credentials.
    
    This is suitable for API keys and similar simple auth schemes.
    """
    
    def __init__(self, default_ttl_seconds: int = 3600) -> None:
        self.default_ttl_seconds = default_ttl_seconds
        self._revoked_tokens: set[str] = set()
    
    @property
    def provider_type(self) -> str:
        return "simple"
    
    async def authenticate(
        self,
        credential: Credential,
        scopes: list[str],
    ) -> ScopedToken:
        """Create a scoped token from the credential."""
        # Validate credential
        if credential.is_expired:
            raise ValueError("Credential has expired")
        
        # Validate scopes
        for scope in scopes:
            if not credential.has_scope(scope):
                raise ValueError(f"Credential does not have scope: {scope}")
        
        # Create token
        return ScopedToken(
            credential_id=credential.credential_id,
            token=credential.secret,  # Pass through the secret
            scopes=tuple(scopes),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.default_ttl_seconds),
        )
    
    async def validate_token(self, token: ScopedToken) -> bool:
        """Check if token is valid."""
        if token.token_id in self._revoked_tokens:
            return False
        return not token.is_expired
    
    async def revoke_token(self, token: ScopedToken) -> bool:
        """Revoke a token."""
        self._revoked_tokens.add(token.token_id)
        return True


class CredentialStore:
    """
    Secure storage for credentials.
    
    The CredentialStore:
    - Stores credentials securely
    - Never exposes raw credentials
    - Generates scoped tokens for tool access
    """
    
    def __init__(self) -> None:
        self._credentials: dict[str, Credential] = {}
        self._credentials_by_name: dict[str, str] = {}  # name -> id
        self._providers: dict[str, AuthProvider] = {}
        self._active_tokens: dict[str, ScopedToken] = {}
        
        # Register default provider
        self._providers["simple"] = SimpleAuthProvider()
    
    def add_credential(self, credential: Credential) -> None:
        """
        Add a credential to the store.
        
        Args:
            credential: The credential to add
        """
        self._credentials[credential.credential_id] = credential
        self._credentials_by_name[credential.name] = credential.credential_id
    
    def remove_credential(self, credential_id: str) -> bool:
        """
        Remove a credential from the store.
        
        Args:
            credential_id: ID of the credential to remove
            
        Returns:
            True if found and removed
        """
        if credential_id in self._credentials:
            credential = self._credentials[credential_id]
            del self._credentials[credential_id]
            del self._credentials_by_name[credential.name]
            return True
        return False
    
    def get_credential(self, credential_id: str) -> Credential | None:
        """Get a credential by ID (without exposing the secret)."""
        return self._credentials.get(credential_id)
    
    def get_credential_by_name(self, name: str) -> Credential | None:
        """Get a credential by name."""
        credential_id = self._credentials_by_name.get(name)
        if credential_id:
            return self._credentials.get(credential_id)
        return None
    
    def list_credentials(self) -> list[str]:
        """List all credential names."""
        return list(self._credentials_by_name.keys())
    
    def register_provider(self, provider: AuthProvider) -> None:
        """Register an auth provider."""
        self._providers[provider.provider_type] = provider
    
    async def get_token(
        self,
        credential_name: str,
        scopes: list[str],
        tool_name: str | None = None,
        agent_id: str | None = None,
    ) -> ScopedToken:
        """
        Get a scoped token for a credential.
        
        This is the main method for tools to get authentication.
        The raw credential is never exposed.
        
        Args:
            credential_name: Name of the credential
            scopes: Requested scopes
            tool_name: Tool requesting the token
            agent_id: Agent requesting the token
            
        Returns:
            A scoped token
            
        Raises:
            ValueError: If credential not found or invalid
        """
        credential = self.get_credential_by_name(credential_name)
        if credential is None:
            raise ValueError(f"Credential not found: {credential_name}")
        
        # Get provider
        provider = self._providers.get(credential.credential_type)
        if provider is None:
            provider = self._providers["simple"]
        
        # Get token
        token = await provider.authenticate(credential, scopes)
        
        # Add context
        token = ScopedToken(
            token_id=token.token_id,
            credential_id=token.credential_id,
            token=token.token,
            scopes=token.scopes,
            created_at=token.created_at,
            expires_at=token.expires_at,
            tool_name=tool_name,
            agent_id=agent_id,
        )
        
        # Track active token
        self._active_tokens[token.token_id] = token
        
        return token
    
    async def validate_token(self, token: ScopedToken) -> bool:
        """Validate a token."""
        if token.token_id not in self._active_tokens:
            return False
        
        credential = self._credentials.get(token.credential_id)
        if credential is None:
            return False
        
        provider = self._providers.get(credential.credential_type)
        if provider is None:
            provider = self._providers["simple"]
        
        return await provider.validate_token(token)
    
    async def revoke_token(self, token_id: str) -> bool:
        """Revoke a token."""
        token = self._active_tokens.get(token_id)
        if token is None:
            return False
        
        credential = self._credentials.get(token.credential_id)
        if credential is None:
            del self._active_tokens[token_id]
            return True
        
        provider = self._providers.get(credential.credential_type)
        if provider is None:
            provider = self._providers["simple"]
        
        result = await provider.revoke_token(token)
        if result:
            del self._active_tokens[token_id]
        
        return result
    
    async def revoke_all_tokens(self, credential_name: str) -> int:
        """
        Revoke all tokens for a credential.
        
        Args:
            credential_name: Name of the credential
            
        Returns:
            Number of tokens revoked
        """
        credential = self.get_credential_by_name(credential_name)
        if credential is None:
            return 0
        
        count = 0
        tokens_to_revoke = [
            token for token in self._active_tokens.values()
            if token.credential_id == credential.credential_id
        ]
        
        for token in tokens_to_revoke:
            if await self.revoke_token(token.token_id):
                count += 1
        
        return count
    
    def get_active_tokens(self, credential_name: str | None = None) -> list[ScopedToken]:
        """
        Get active tokens.
        
        Args:
            credential_name: Optional filter by credential name
            
        Returns:
            List of active tokens
        """
        tokens = list(self._active_tokens.values())
        
        if credential_name:
            credential = self.get_credential_by_name(credential_name)
            if credential:
                tokens = [
                    t for t in tokens
                    if t.credential_id == credential.credential_id
                ]
            else:
                tokens = []
        
        return tokens
    
    def cleanup_expired_tokens(self) -> int:
        """
        Remove expired tokens.
        
        Returns:
            Number of tokens removed
        """
        expired = [
            token_id for token_id, token in self._active_tokens.items()
            if token.is_expired
        ]
        
        for token_id in expired:
            del self._active_tokens[token_id]
        
        return len(expired)


class AuthDelegator:
    """
    Handles authentication delegation for tool invocations.
    
    The AuthDelegator:
    - Maps tools to required credentials
    - Generates scoped tokens for tool access
    - Tracks token usage
    """
    
    def __init__(self, credential_store: CredentialStore) -> None:
        self.credential_store = credential_store
        self._tool_credentials: dict[str, str] = {}  # tool_name -> credential_name
        self._tool_scopes: dict[str, list[str]] = {}  # tool_name -> required scopes
    
    def configure_tool(
        self,
        tool_name: str,
        credential_name: str,
        scopes: list[str],
    ) -> None:
        """
        Configure authentication for a tool.
        
        Args:
            tool_name: Name of the tool
            credential_name: Name of the credential to use
            scopes: Scopes required by the tool
        """
        self._tool_credentials[tool_name] = credential_name
        self._tool_scopes[tool_name] = scopes
    
    def remove_tool_config(self, tool_name: str) -> bool:
        """Remove authentication configuration for a tool."""
        if tool_name in self._tool_credentials:
            del self._tool_credentials[tool_name]
            del self._tool_scopes[tool_name]
            return True
        return False
    
    def get_tool_credential(self, tool_name: str) -> str | None:
        """Get the credential name configured for a tool."""
        return self._tool_credentials.get(tool_name)
    
    def get_tool_scopes(self, tool_name: str) -> list[str]:
        """Get the scopes required by a tool."""
        return self._tool_scopes.get(tool_name, [])
    
    async def get_token_for_tool(
        self,
        tool_name: str,
        agent_id: str,
        additional_scopes: list[str] | None = None,
    ) -> ScopedToken | None:
        """
        Get a scoped token for a tool invocation.
        
        Args:
            tool_name: Name of the tool
            agent_id: ID of the agent
            additional_scopes: Additional scopes to request
            
        Returns:
            Scoped token, or None if no credential configured
        """
        credential_name = self._tool_credentials.get(tool_name)
        if credential_name is None:
            return None
        
        scopes = self._tool_scopes.get(tool_name, [])
        if additional_scopes:
            scopes = list(set(scopes + additional_scopes))
        
        return await self.credential_store.get_token(
            credential_name=credential_name,
            scopes=scopes,
            tool_name=tool_name,
            agent_id=agent_id,
        )
    
    def requires_auth(self, tool_name: str) -> bool:
        """Check if a tool requires authentication."""
        return tool_name in self._tool_credentials
