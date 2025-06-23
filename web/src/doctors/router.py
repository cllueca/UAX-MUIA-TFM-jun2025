# Builtins
import os
# Installed
from fastapi import (
    APIRouter,
    Request,
    Depends,
    Path
)
from fastapi.responses import JSONResponse
# Local
from src.core.logger import DOCTOR_LOGGER
from src.core.database import DatabaseOperations
from src.doctors import models
from src.utils.auth import get_csession_token_from_cookie
# Types
from typing import Annotated


doctors_bp = APIRouter(
    include_in_schema=True,
    tags=['Doctors']
)


# Auth logic
@doctors_bp.post("/login")
async def login(
    request: Request,
    data: models.LoginRequest,
    dbops: DatabaseOperations = Depends()
) -> JSONResponse:
    
    success = False
    session_token = ""

    doc = dbops.get_doctor_by_email(email=data.email)
    if doc:
        success = dbops.verify_password(doc['id'], data.password)

        if success:
            session_token = dbops.create_session(doc['id'])
            
        message = "Bienvenido!" if success else "Credenciales erroneas"
        DOCTOR_LOGGER.info(f"LOGIN - {data.email} logged in" if success else f"LOGIN - {data.email}: invalid credentials")
    else:
        DOCTOR_LOGGER.info(f"LOGIN - {data.email} not found on the database")
        message = "El correo no se ha encontrado en la base de datos"
    
    response = JSONResponse(
        {
            "success": success,
            "message": message
        }
    )
    response.set_cookie(
        key="sessionToken",
        value=session_token,
        httponly=True,  # prevents access from JS
        secure=False,    # ensures it's only sent via HTTPS
        samesite="strict",
        path="/"
    )
    return response


@doctors_bp.get("/logout")
async def logout(
    request: Request,
    session = Depends(get_csession_token_from_cookie),
    dbops: DatabaseOperations = Depends()
) -> JSONResponse:
    session_ended = dbops.invalidate_session(session['session_token'])
    
    message = "Sesion cerrada" if session_ended else "La sesion no se pudo cerrar"
    DOCTOR_LOGGER.info(
        f"Session with session token {session['session_token']} terminated for {session['email']}"
        if session_ended else 
        f"Could not end session {session['session_token']} for {session['email']}"
    )
    
    response = JSONResponse(
        {
            "success": session_ended,
            "message": message
        }
    )
    response.delete_cookie("sessionToken", path="/")
    return response


@doctors_bp.put("/{email}/reset-password")
async def login(
    request: Request,
    email: Annotated[str, Path(description="Doctor's email")],
    data: models.ChangePasswordRequest,
    dbops: DatabaseOperations = Depends()
) -> JSONResponse:
    
    success = False
    reset_complete = False

    doc = dbops.get_doctor_by_email(email=email)
    if doc:
        success = dbops.verify_password(doc['id'], data.old_password)
        message = "Credenciales erroneas"
        if success:
            reset_complete = dbops.update_password(doc['id'], data.new_password)
            message = "Password actualizada!" if reset_complete else "Error al cambiar password"
        
        DOCTOR_LOGGER.info(f"CHANGEPWD - {email} password changed" if success else f"CHANGEPWD - {email}: error changing password")
    else:
        DOCTOR_LOGGER.info(f"CHANGEPWD - {email} not found on the database")
        message = "El correo no se ha encontrado en la base de datos"

    return JSONResponse(
        {
            "success": success,
            "message": message
        }
    )


# Doctor-Patient logic
@doctors_bp.get("/patient/all")
async def get_all_patients(
    request: Request,
    session = Depends(get_csession_token_from_cookie),
    dbops: DatabaseOperations = Depends()
) -> JSONResponse: 
    patient_list = dbops.get_all_patients_by_doctor(session['id'])
    patients = [dict(row) for row in patient_list]
    return JSONResponse(content=patients)
