import os
import time
import json
import logging
import threading
import asyncio
import aiohttp
import pyotp
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
from live_strategy.error_handling import AuthResponse

# Get the absolute path to the logs directory
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logs_dir = os.path.join(script_dir, 'scripts', 'logs')
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, 'trading.log')

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('auth_service')

@dataclass
class AuthResponse:
    status: bool
    message: str
    jwt_token: Optional[str] = None
    refresh_token: Optional[str] = None
    feed_token: Optional[str] = None
    state: Optional[str] = None

class AuthService:
    """Service for handling authentication with Angel One API"""
    
    BASE_URL = "https://apiconnect.angelone.in"
    
    def __init__(self, api_key: str, client_id: str, totp_key: str):
        self.api_key = api_key
        self.client_id = client_id
        self.totp_key = totp_key
        self._jwt_token = None
        self._refresh_token = None
        self._feed_token = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize rate limiting attributes
        self._last_login_attempt = None
        self._login_rate_limit = timedelta(minutes=1)  # 1 minute between login attempts
        
        # Initialize token validity tracking
        self._token_validity_duration = timedelta(hours=24)  # Tokens valid for 24 hours
        self._token_expiry = None
        self._state = "STATE_VARIABLE"  # Default state variable
        
        # Initialize token refresh attributes
        self._is_refreshing = False
        self._last_token_refresh = None
        self._token_refresh_rate_limit = timedelta(minutes=5)  # 5 minutes between refresh attempts
        self.API_KEY = api_key  # Store API key for refresh operations
        
    def get_headers(self, api_key: str = None) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '172.20.10.2',
            'X-ClientPublicIP': '2409:40c0:61:9381:61cc:2241:1ecf:b89e',
            'X-MACAddress': '82:73:32:c9:20:01',
            'X-PrivateKey': api_key or self.api_key
        }
        
        if self._jwt_token:
            headers['Authorization'] = f'Bearer {self._jwt_token}'
            
        return headers

    async def initialize_auth(self, credentials: Dict[str, str]) -> bool:
        """Initialize authentication with the provided credentials"""
        try:
            logger.info("Initializing authentication...")
            
            # Check if already authenticated
            if self.is_authenticated:
                logger.info("Already authenticated")
                return True
            
            # Check rate limit
            if self._last_login_attempt and (datetime.now() - self._last_login_attempt) < self._login_rate_limit:
                wait_time = (self._login_rate_limit - (datetime.now() - self._last_login_attempt)).total_seconds()
                logger.info(f'Rate limit hit, waiting {wait_time} seconds...')
                await asyncio.sleep(wait_time)
            
            # Attempt login with current credentials
            response = await self.login(
                client_code=credentials['client_code'],
                password=credentials['password'],
                api_key=credentials['api_key']
            )
            
            if not response.status:
                logger.error(f"Failed to initialize authentication: {response.message}")
                return False
            
            # Store tokens
            self._jwt_token = response.jwt_token
            self._refresh_token = response.refresh_token
            self._feed_token = response.feed_token
            self._state = response.state
            
            # Set token expiry
            self._token_expiry = datetime.now() + self._token_validity_duration
            
            logger.info("Authentication initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            return False

    async def _generate_totp(self) -> str:
        """Generate TOTP code"""
        try:
            # Use the same hardcoded secret key as the JavaScript implementation
            secret_base32 = "TRBMNCFYTMXYDQVF7VNW2OVJXU"
            totp = pyotp.TOTP(secret_base32)
            return totp.now()
        except Exception as e:
            logger.error(f'Error generating TOTP code: {e}')
            return "123456"  # Fallback value

    @property
    def is_authenticated(self) -> bool:
        """Check if currently authenticated with valid token"""
        return bool(self._jwt_token and 
                   self._token_expiry and 
                   datetime.now() < self._token_expiry)

    def _get_token_expiry_status(self) -> str:
        """Get human-readable token expiry status"""
        if not self._token_expiry:
            return "No expiry set"
        
        now = datetime.now()
        if now >= self._token_expiry:
            return "Expired"
        
        time_left = self._token_expiry - now
        hours = time_left.total_seconds() / 3600
        
        if hours > 12:
            return "Valid"
        elif hours > 1:
            return f"Expires in {int(hours)} hours"
        else:
            minutes = time_left.total_seconds() / 60
            return f"Expires in {int(minutes)} minutes"

    async def check_and_refresh_token_if_needed(self) -> bool:
        """Check token status and refresh if needed"""
        if not self.is_authenticated:
            logger.info("Token expired or not available, refreshing...")
            response = await self.refresh_token()
            return response.status
        
        if self._token_expiry:
            time_to_expiry = self._token_expiry - datetime.now()
            # Refresh if less than 30 minutes remaining (matching Dart implementation)
            if time_to_expiry.total_seconds() < 1800:
                logger.info(f"Token expires in {time_to_expiry.seconds//60} minutes, refreshing...")
                response = await self.refresh_token()
                return response.status
        
        return True

    async def login(self, client_code: str, password: str, api_key: str, state: str = 'STATE_VARIABLE') -> AuthResponse:
        """Login to Smart API"""
        try:
            # Add delay before authentication attempts
            await asyncio.sleep(3)
            
            # Check rate limit
            if self._last_login_attempt and (datetime.now() - self._last_login_attempt) < self._login_rate_limit:
                wait_time = (self._login_rate_limit - (datetime.now() - self._last_login_attempt)).total_seconds()
                logger.info(f'Rate limit hit, waiting {wait_time} seconds...')
                await asyncio.sleep(wait_time)
            
            url = f"{self.BASE_URL}/rest/auth/angelbroking/user/v1/loginByPassword"
            
            # Generate TOTP code
            totp_code = await self._generate_totp()
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'X-UserType': 'USER',
                'X-SourceID': 'WEB',
                'X-ClientLocalIP': '172.20.10.2',
                'X-ClientPublicIP': '2409:40c0:61:9381:61cc:2241:1ecf:b89e',
                'X-MACAddress': '82:73:32:c9:20:01',
                'X-PrivateKey': api_key
            }
            
            payload = {
                "clientcode": client_code,
                "password": password,
                "totp": totp_code,
                "state": state
            }
            
            logger.info(f'Attempting login for client: {client_code}')
            logger.info(f'Login URL: {url}')
            logger.info(f'Login headers: {headers}')
            logger.info(f'Login payload: {payload}')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    logger.info(f'Login response status: {response.status}')
                    response_text = await response.text()
                    logger.info(f'Login response text: {response_text}')
                    
                    # Update last login attempt time
                    self._last_login_attempt = datetime.now()
                    
                    if response.status == 200:
                        try:
                            response_data = json.loads(response_text)
                            if response_data.get('status', False):
                                data = response_data.get('data', {})
                                return AuthResponse(
                                    status=True,
                                    message="Login successful",
                                    jwt_token=data.get('jwtToken'),
                                    refresh_token=data.get('refreshToken'),
                                    feed_token=data.get('feedToken'),
                                    state=state
                                )
                            else:
                                return AuthResponse(
                                    status=False,
                                    message=response_data.get('message', 'Login failed')
                                )
                        except json.JSONDecodeError as e:
                            logger.error(f'Failed to decode JSON response: {e}')
                            return AuthResponse(
                                status=False,
                                message=f'Invalid response format: {response_text}'
                            )
                    else:
                        return AuthResponse(
                            status=False,
                            message=f'Login failed with status {response.status}: {response_text}'
                        )
                        
        except Exception as e:
            logger.error(f'Login error: {str(e)}')
            return AuthResponse(
                status=False,
                message=f'Login error: {str(e)}'
            )

    async def refresh_token(self, api_key: Optional[str] = None) -> AuthResponse:
        """Refresh the JWT token"""
        if self._is_refreshing:
            return AuthResponse(status=False, message='Refresh already in progress')

        self._is_refreshing = True
        try:
            if not self._refresh_token:
                return AuthResponse(status=False, message='No refresh token available')

            # Rate limiting
            if self._last_token_refresh:
                time_since_last = datetime.now() - self._last_token_refresh
                if time_since_last.total_seconds() < self._token_refresh_rate_limit.total_seconds():
                    await asyncio.sleep((self._token_refresh_rate_limit - time_since_last).total_seconds())

            self._last_token_refresh = datetime.now()
            api_key = api_key or self.API_KEY

            headers = self.get_headers(api_key)
            data = {"refreshToken": self._refresh_token}

            logger.info("Attempting to refresh token...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.BASE_URL}/rest/auth/angelbroking/jwt/v1/generateTokens',
                    headers=headers,
                    json=data
                ) as response:
                    resp_data = await response.json()
                    logger.info(f"Refresh response: {resp_data}")

                    if response.status == 200 and resp_data.get('status', False):
                        # Update tokens from data field
                        jwt_token = resp_data.get('data', {}).get('jwtToken', '')
                        refresh_token = resp_data.get('data', {}).get('refreshToken', '')
                        feed_token = resp_data.get('data', {}).get('feedToken', '')
                        state = resp_data.get('data', {}).get('state', '')

                        if not jwt_token or not refresh_token:
                            logger.error("Refresh successful but no tokens received")
                            return AuthResponse(
                                status=False,
                                message='Refresh successful but no tokens received'
                            )

                        self._jwt_token = jwt_token
                        self._refresh_token = refresh_token
                        self._feed_token = feed_token
                        self._state = state
                        self._token_expiry = datetime.now() + self._token_validity_duration

                        logger.info("Token refresh successful, tokens updated")
                        return AuthResponse(
                            status=True,
                            message='Token refresh successful',
                            jwt_token=self._jwt_token,
                            refresh_token=self._refresh_token,
                            feed_token=self._feed_token,
                            state=self._state
                        )

                    error_message = resp_data.get('message', 'Token refresh failed')
                    logger.error(f"Token refresh failed: {error_message}")
                    return AuthResponse(
                        status=False,
                        message=error_message
                    )

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return AuthResponse(status=False, message=str(e))
        finally:
            self._is_refreshing = False

    def get_token_status(self) -> Dict[str, Any]:
        """Get current token status"""
        return {
            'is_authenticated': self.is_authenticated,
            'expiry_status': self._get_token_expiry_status(),
            'jwt_token': bool(self._jwt_token),
            'refresh_token': bool(self._refresh_token),
            'feed_token': bool(self._feed_token),
            'state': self._state,
            'token_expiry': self._token_expiry.isoformat() if self._token_expiry else None
        } 