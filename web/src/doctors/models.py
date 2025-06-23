# Builtins
# Installed
from pydantic import (
    BaseModel,
    Field
)
# Local
# Types



class LoginRequest(BaseModel):
    email: str = Field(..., description="The user's email address")
    password: str = Field(..., description="The user's password")


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., description="The user's current password")
    new_password: str = Field(..., description="The user's new password")
