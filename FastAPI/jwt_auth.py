import logging
import os
from typing import Any, Dict, Optional, Union

import jwt
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()


logger = get_logger("JWT AUTH")

# Get configuration from environment variables
SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
ALGORITHM = os.getenv("AUTH_ALGORITHM")

async def verify_jwt_token(token: str) -> Union[Dict[str, Any], None]:
    """
    Verify JWT token and return payload if valid, None if invalid

    Args:
        token: JWT token string to verify

    Returns:
        Dict containing payload if valid, None if invalid
    """

    try:
        # Decode and verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        logger.info(
            f"Token verified successfully for user: {payload.get('sub', 'unknown')}"
        )
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None

    except jwt.InvalidTokenError:
        logger.warning("Invalid token provided")
        return None

    except Exception as e:
        logger.error(f"Unexpected error verifying token: {str(e)}")
        return None


async def get_current_user(authorization) -> Optional[Dict[str, Any]]:
    """
    Get current user from JWT token in request headers
    Returns None if no valid token found
    """
    # print("========== inside get_current_user() function with auth token: ",authorization)
    if not authorization:
        print("No authorization")
        return None

    if authorization.startswith("Bearer "):
        token = authorization[7:]

    else:
        return None

    # Validate configuration
    if not SECRET_KEY:
        logger.error("AUTH_SECRET_KEY environment variable not set")
        return None

    # Validate input
    if not token or not isinstance(token, str):
        logger.warning("Invalid token provided")
        return None

    return await verify_jwt_token(token)
