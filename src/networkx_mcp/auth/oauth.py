#!/usr/bin/env python3
"""OAuth 2.0 authentication support for remote MCP access.

Implements OAuth 2.0 bearer token validation for secure remote access.
"""

import asyncio
import json
import time
import jwt
from dataclasses import dataclass
from typing import Optional, Dict, Any, Set
from aiohttp import ClientSession, ClientTimeout
import aiohttp

from ..logging import get_logger
from ..config.production import production_config

logger = get_logger(__name__)


@dataclass
class UserInfo:
    """Authenticated user information."""
    user_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    scopes: Set[str] = None
    expires_at: Optional[float] = None
    
    def __post_init__(self):
        if self.scopes is None:
            self.scopes = set()
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def has_scope(self, scope: str) -> bool:
        """Check if user has specific scope."""
        return scope in self.scopes


class OAuth2Handler:
    """OAuth 2.0 authentication handler."""
    
    def __init__(self, 
                 provider_url: Optional[str] = None,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 jwks_url: Optional[str] = None):
        
        self.provider_url = provider_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.jwks_url = jwks_url
        
        # Token cache
        self.token_cache: Dict[str, UserInfo] = {}
        self.cache_timeout = 300  # 5 minutes
        
        # JWKS cache for JWT validation
        self.jwks_cache: Optional[Dict] = None
        self.jwks_cache_time: float = 0
        self.jwks_cache_timeout = 3600  # 1 hour
        
        # Required scopes
        self.required_scopes = {'mcp:read', 'mcp:write'}
        
        logger.info("OAuth2Handler initialized")
    
    async def validate_token(self, authorization_header: str) -> Optional[UserInfo]:
        """
        Validate OAuth 2.0 bearer token.
        
        Args:
            authorization_header: Authorization header value
            
        Returns:
            UserInfo if valid, None otherwise
        """
        if not authorization_header or not authorization_header.startswith("Bearer "):
            logger.debug("Missing or invalid authorization header format")
            return None
        
        token = authorization_header[7:]  # Remove "Bearer "
        
        # Check cache first
        if token in self.token_cache:
            user_info = self.token_cache[token]
            if not user_info.is_expired:
                logger.debug(f"Using cached token for user {user_info.user_id}")
                return user_info
            else:
                del self.token_cache[token]
        
        # Validate token
        try:
            user_info = await self._validate_token_with_provider(token)
            if user_info:
                # Cache valid token
                self.token_cache[token] = user_info
                logger.info(f"Token validated for user {user_info.user_id}")
                return user_info
            else:
                logger.warning("Token validation failed")
                return None
                
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
    
    async def _validate_token_with_provider(self, token: str) -> Optional[UserInfo]:
        """Validate token with OAuth provider."""
        
        # Try JWT validation first (faster)
        if self.jwks_url:
            try:
                user_info = await self._validate_jwt_token(token)
                if user_info:
                    return user_info
            except Exception as e:
                logger.debug(f"JWT validation failed, trying introspection: {e}")
        
        # Fall back to token introspection
        if self.provider_url:
            return await self._validate_with_introspection(token)
        
        # Simple validation for development
        return await self._validate_simple_token(token)
    
    async def _validate_jwt_token(self, token: str) -> Optional[UserInfo]:
        """Validate JWT token using JWKS."""
        if not self.jwks_url:
            return None
        
        # Get JWKS
        jwks = await self._get_jwks()
        if not jwks:
            return None
        
        try:
            # Decode JWT header to get key ID
            header = jwt.get_unverified_header(token)
            kid = header.get('kid')
            
            # Find matching key
            key = None
            for jwk in jwks.get('keys', []):
                if jwk.get('kid') == kid:
                    key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
                    break
            
            if not key:
                logger.warning(f"No matching key found for kid: {kid}")
                return None
            
            # Verify and decode token
            payload = jwt.decode(
                token,
                key,
                algorithms=['RS256'],
                audience=self.client_id,
                options={"verify_exp": True}
            )
            
            # Extract user information
            user_id = payload.get('sub')
            email = payload.get('email')
            name = payload.get('name', payload.get('preferred_username'))
            scopes = set(payload.get('scope', '').split())
            expires_at = payload.get('exp')
            
            if not user_id:
                logger.warning("Token missing user ID (sub)")
                return None
            
            # Check required scopes
            if not self.required_scopes.issubset(scopes):
                missing_scopes = self.required_scopes - scopes
                logger.warning(f"Token missing required scopes: {missing_scopes}")
                return None
            
            return UserInfo(
                user_id=user_id,
                email=email,
                name=name,
                scopes=scopes,
                expires_at=expires_at
            )
            
        except jwt.ExpiredSignatureError:
            logger.debug("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    async def _get_jwks(self) -> Optional[Dict]:
        """Get JWKS from provider."""
        current_time = time.time()
        
        # Check cache
        if (self.jwks_cache and 
            current_time - self.jwks_cache_time < self.jwks_cache_timeout):
            return self.jwks_cache
        
        try:
            timeout = ClientTimeout(total=10)
            async with ClientSession(timeout=timeout) as session:
                async with session.get(self.jwks_url) as response:
                    if response.status == 200:
                        jwks = await response.json()
                        self.jwks_cache = jwks
                        self.jwks_cache_time = current_time
                        logger.debug("JWKS cache updated")
                        return jwks
                    else:
                        logger.error(f"Failed to fetch JWKS: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching JWKS: {e}")
            return None
    
    async def _validate_with_introspection(self, token: str) -> Optional[UserInfo]:
        """Validate token using OAuth 2.0 token introspection."""
        if not self.provider_url:
            return None
        
        introspection_url = f"{self.provider_url}/oauth/introspect"
        
        try:
            timeout = ClientTimeout(total=10)
            async with ClientSession(timeout=timeout) as session:
                data = {
                    'token': token,
                    'token_type_hint': 'access_token'
                }
                
                # Basic auth with client credentials
                auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
                
                async with session.post(
                    introspection_url,
                    data=data,
                    auth=auth
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        if not result.get('active', False):
                            logger.debug("Token is not active")
                            return None
                        
                        # Extract user information
                        user_id = result.get('sub')
                        email = result.get('email')
                        name = result.get('name', result.get('username'))
                        scopes = set(result.get('scope', '').split())
                        expires_at = result.get('exp')
                        
                        if not user_id:
                            logger.warning("Introspection response missing user ID")
                            return None
                        
                        # Check required scopes
                        if not self.required_scopes.issubset(scopes):
                            missing_scopes = self.required_scopes - scopes
                            logger.warning(f"Token missing required scopes: {missing_scopes}")
                            return None
                        
                        return UserInfo(
                            user_id=user_id,
                            email=email,
                            name=name,
                            scopes=scopes,
                            expires_at=expires_at
                        )
                    else:
                        logger.error(f"Token introspection failed: {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Token introspection error: {e}")
            return None
    
    async def _validate_simple_token(self, token: str) -> Optional[UserInfo]:
        """Simple token validation for development/testing."""
        # In development, accept configured tokens
        valid_tokens = {
            "dev-token-123": UserInfo(
                user_id="dev-user",
                email="dev@example.com",
                name="Development User",
                scopes=self.required_scopes.union({'admin'})
            ),
            "test-token-456": UserInfo(
                user_id="test-user",
                email="test@example.com",
                name="Test User",
                scopes=self.required_scopes
            )
        }
        
        # Also accept production auth token as admin
        if production_config.AUTH_TOKEN and token == production_config.AUTH_TOKEN:
            return UserInfo(
                user_id="admin",
                email="admin@localhost",
                name="Admin User",
                scopes=self.required_scopes.union({'admin'})
            )
        
        user_info = valid_tokens.get(token)
        if user_info:
            logger.debug(f"Simple token validation for {user_info.user_id}")
            return user_info
        
        logger.debug("Unknown token in simple validation")
        return None
    
    def requires_scope(self, scope: str):
        """Decorator to require specific scope for endpoint."""
        def decorator(func):
            async def wrapper(request):
                # Get user info from request (set by auth middleware)
                user_info = getattr(request, 'user_info', None)
                
                if not user_info or not user_info.has_scope(scope):
                    from aiohttp import web
                    return web.json_response(
                        {"error": f"Insufficient permissions. Required scope: {scope}"},
                        status=403
                    )
                
                return await func(request)
            return wrapper
        return decorator
    
    async def cleanup_expired_tokens(self):
        """Background task to cleanup expired cached tokens."""
        while True:
            try:
                current_time = time.time()
                expired_tokens = []
                
                for token, user_info in self.token_cache.items():
                    if user_info.is_expired:
                        expired_tokens.append(token)
                
                for token in expired_tokens:
                    del self.token_cache[token]
                    logger.debug("Removed expired token from cache")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Token cleanup error: {e}")
                await asyncio.sleep(60)


# Factory function for OAuth handler
def create_oauth_handler(
    provider_url: Optional[str] = None,
    client_id: Optional[str] = None, 
    client_secret: Optional[str] = None,
    jwks_url: Optional[str] = None
) -> OAuth2Handler:
    """Create OAuth2Handler with environment configuration."""
    
    import os
    
    # Use environment variables if not provided
    provider_url = provider_url or os.getenv('OAUTH_PROVIDER_URL')
    client_id = client_id or os.getenv('OAUTH_CLIENT_ID')
    client_secret = client_secret or os.getenv('OAUTH_CLIENT_SECRET')
    jwks_url = jwks_url or os.getenv('OAUTH_JWKS_URL')
    
    return OAuth2Handler(
        provider_url=provider_url,
        client_id=client_id,
        client_secret=client_secret,
        jwks_url=jwks_url
    )