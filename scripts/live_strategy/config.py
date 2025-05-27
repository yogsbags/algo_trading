import os
import json
import logging
from typing import Dict, Any
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

class ConfigModel(BaseModel):
    symbol: str
    quantity: int
    sl_long: int
    tp_long: int
    sl_short: int 
    tp_short: int
    activation_gap: float
    trail_offset: float
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str

class ConfigManager:
    """Complete configuration handler with validation"""
    
    def __init__(self):
        self.config = None
        self.env = os.getenv("TRADING_ENV", "production")

    def load_config(self) -> Dict[str, Any]:
        """Load and validate configuration"""
        try:
            config_path = f"config/{self.env}.json"
            with open(config_path) as f:
                raw_config = json.load(f)
            
            self.config = ConfigModel(**raw_config).dict()
            logger.info("Configuration validated successfully")
            return self.config
            
        except FileNotFoundError:
            logger.error("Configuration file not found")
            raise
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Config loading error: {str(e)}")
            raise 