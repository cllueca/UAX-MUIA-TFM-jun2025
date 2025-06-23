# Builtins
# Installed
from fastapi import (
    Depends,
    HTTPException,
    Cookie
)
from starlette.status import HTTP_401_UNAUTHORIZED
# Local
from src.core.database import DatabaseOperations
# Types


def get_csession_token_from_cookie(
    session_token: str = Cookie(default=None, alias="sessionToken"),
    db_ops: DatabaseOperations = Depends()
):
    if not session_token:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Missing session token in cookie")
    session_info = db_ops.get_session_by_token(session_token)
    if not session_info:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid or expired session")
    return session_info
