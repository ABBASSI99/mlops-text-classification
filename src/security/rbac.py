import os
from fastapi import Depends
import jwt

ROLE_PERMISSIONS = {
    "admin": ["predict", "retrain", "monitor"],
    "user": ["predict"]
}

SECRET_KEY = os.environ.get("SECRET_KEY", "default_secret_key")  # À définir dans l’environnement

def get_current_user(token: str = Depends()):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["sub"], payload["role"]
    except jwt.PyJWTError:
        return None

def has_permission(user, permission):
    if not user:
        return False
    username, role = user
    return permission in ROLE_PERMISSIONS.get(role, [])