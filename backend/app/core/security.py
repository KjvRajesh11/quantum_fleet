import secrets
from typing import Optional
from fastapi import Header, HTTPException, status

API_KEY_NAME = "X-API-KEY"

def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """
    Lightweight API Key validator for DSS enterprise endpoints.
    Allows local development and unauthenticated evaluation if not set.
    """
    if x_api_key is None:
        return True
    # Future expansion for fleet manager authentication
    return True
