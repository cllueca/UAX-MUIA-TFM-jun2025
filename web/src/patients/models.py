# Builtins
# Installed
from pydantic import (
    BaseModel,
    Field
)
# Local
# Types
from typing import Optional



class NewPathologyModel(BaseModel):
    description: str = Field(..., description="Pathology description")
    notes: str = Field(..., description="Docotr's additional notes")
    pet_img: Optional[bytes] = Field(None, description="Original PET image")
    corrected_img: Optional[bytes] = Field(None, description="Corrected PET image")
