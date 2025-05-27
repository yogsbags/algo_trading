import logging
import asyncio
import time
import functools
from typing import Callable, Any, List, Dict, Optional, Type, Union
from enum import Enum
import traceback
from dataclasses import dataclass

logger = logging.getLogger('error_handling')

@dataclass
class AuthResponse:
    """Response from authentication operations"""
    status: bool
    message: str
    jwt_token: Optional[str] = None
    refresh_token: Optional[str] = None
    feed_token: Optional[str] = None
    state: Optional[str] = None

class ErrorCategory(Enum):
    """Categories of errors for different handling strategies"""
    # Network/API errors that are likely transient
    TRANSIENT = "transient"
    
    # Authentication errors that require reauthorization
    AUTH = "auth"
    
    # Rate limiting errors that require backing off
    RATE_LIMIT = "rate_limit"
    
    # Data errors (missing, invalid, etc.)
    DATA = "data"
    
    # Configuration errors that require human intervention
    CONFIG = "config"
    
    # Unexpected errors that don't fit other categories
    UNKNOWN = "unknown"

class AppError(Exception):
    """Base application error class with categorization"""
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, 
                 original_error: Optional[Exception] = None, retry_after: Optional[float] = None):
        self.message = message
        self.category = category
        self.original_error = original_error
        self.retry_after = retry_after  # Seconds to wait before retry
        super().__init__(self.message)
    
    def __str__(self):
        base_message = f"{self.category.value.upper()} ERROR: {self.message}"
        if self.retry_after is not None:
            base_message += f" (retry after {self.retry_after}s)"
        if self.original_error:
            base_message += f"\nOriginal error: {str(self.original_error)}"
        return base_message

class NetworkError(AppError):
    """Network-related errors"""
    def __init__(self, message: str, original_error: Optional[Exception] = None, retry_after: Optional[float] = None):
        super().__init__(message, ErrorCategory.TRANSIENT, original_error, retry_after)

class AuthError(AppError):
    """Authentication errors"""
    def __init__(self, message: str, original_error: Optional[Exception] = None, retry_after: Optional[float] = None):
        super().__init__(message, ErrorCategory.AUTH, original_error, retry_after)

class RateLimitError(AppError):
    """Rate limiting errors"""
    def __init__(self, message: str, original_error: Optional[Exception] = None, retry_after: Optional[float] = None):
        super().__init__(message, ErrorCategory.RATE_LIMIT, original_error, retry_after)

class DataError(AppError):
    """Data-related errors"""
    def __init__(self, message: str, original_error: Optional[Exception] = None, retry_after: Optional[float] = None):
        super().__init__(message, ErrorCategory.DATA, original_error, retry_after)

class ConfigError(AppError):
    """Configuration errors"""
    def __init__(self, message: str, original_error: Optional[Exception] = None, retry_after: Optional[float] = None):
        super().__init__(message, ErrorCategory.CONFIG, original_error, retry_after)

# Retry decorator for async functions
def async_retry(max_retries: int = 3, 
                retry_exceptions: List[Type[Exception]] = None,
                initial_delay: float = 1.0,
                max_delay: float = 60.0,
                backoff_factor: float = 2.0,
                logger_instance = None):
    """
    Decorator that retries an async function if it raises specified exceptions
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_exceptions: List of exception types to catch and retry (default: AppError w/ transient/rate_limit categories)
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for successive retry delays
        logger_instance: Logger to use (defaults to error_handling logger)
    """
    if retry_exceptions is None:
        # Default to retrying for transient errors and rate limit errors
        retry_exceptions = [NetworkError, RateLimitError]
    
    log = logger_instance or logger
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(retry_exceptions) as e:
                    last_exception = e
                    
                    # Check if we've maxed out retries
                    if attempt >= max_retries:
                        log.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}")
                        raise
                    
                    # Use retry_after from the exception if available
                    if isinstance(e, AppError) and e.retry_after is not None:
                        retry_wait = e.retry_after
                    else:
                        retry_wait = min(delay * (backoff_factor ** attempt), max_delay)
                    
                    log.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__} after {retry_wait:.2f}s: {e}")
                    await asyncio.sleep(retry_wait)
                except Exception as e:
                    # For non-retryable exceptions, log and re-raise
                    log.error(f"Non-retryable error in {func.__name__}: {e}")
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug(traceback.format_exc())
                    raise
            
            # If we get here, we've exhausted retries
            log.error(f"All retries failed for {func.__name__}")
            raise last_exception
            
        return wrapper
    
    return decorator

# Error categorization function
def categorize_error(e: Exception) -> ErrorCategory:
    """Categorize an exception for appropriate handling"""
    
    # Check if it's already an AppError with a category
    if isinstance(e, AppError):
        return e.category
    
    # Look at the exception type and message for clues
    error_text = str(e).lower()
    error_type = type(e).__name__.lower()
    
    # Network errors
    if any(text in error_text or text in error_type for text in [
        'connection', 'timeout', 'network', 'socket', 'connect', 'timed out'
    ]):
        return ErrorCategory.TRANSIENT
    
    # Authentication errors
    if any(text in error_text for text in [
        'auth', 'login', 'password', 'credential', 'token', 'unauthorized', 'permission',
        '401', '403'
    ]):
        return ErrorCategory.AUTH
    
    # Rate limiting
    if any(text in error_text for text in [
        'rate limit', 'too many requests', 'throttle',
        '429', 'slow down'
    ]):
        return ErrorCategory.RATE_LIMIT
    
    # Data errors
    if any(text in error_text for text in [
        'not found', 'missing', 'invalid data', 'null', 'undefined',
        '404', 'data error'
    ]):
        return ErrorCategory.DATA
    
    # Configuration errors
    if any(text in error_text for text in [
        'config', 'configuration', 'setting', 'parameter',
        'environment variable'
    ]):
        return ErrorCategory.CONFIG
    
    # Default to unknown
    return ErrorCategory.UNKNOWN

# Function to convert exceptions to AppError types
def convert_exception(e: Exception) -> AppError:
    """Convert a generic exception to the appropriate AppError type"""
    category = categorize_error(e)
    
    # If it's already an AppError, return it
    if isinstance(e, AppError):
        return e
    
    # Otherwise, create the appropriate AppError type
    if category == ErrorCategory.TRANSIENT:
        return NetworkError(str(e), original_error=e)
    elif category == ErrorCategory.AUTH:
        return AuthError(str(e), original_error=e)
    elif category == ErrorCategory.RATE_LIMIT:
        return RateLimitError(str(e), original_error=e)
    elif category == ErrorCategory.DATA:
        return DataError(str(e), original_error=e)
    elif category == ErrorCategory.CONFIG:
        return ConfigError(str(e), original_error=e)
    else:
        return AppError(str(e), category=category, original_error=e)