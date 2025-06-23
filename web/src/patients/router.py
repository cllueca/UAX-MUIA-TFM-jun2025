# Builtins
# Installed
from fastapi import (
    APIRouter,
    Request,
    Depends,
    Path,
    UploadFile,
    File
)
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR
)
# Local
from src.core.logger import PATIENT_LOGGER
from src.core.database import DatabaseOperations
from src.patients import models
from src.patients import service
from src.utils.auth import get_csession_token_from_cookie
# Types
from typing import Annotated


patients_bp = APIRouter(
    include_in_schema=True,
    tags=['Patients']
)


@patients_bp.get("/{patient_id}")
async def get_one_patient(
    request: Request,
    patient_id: Annotated[int, Path(description="Patient ID")],
    session = Depends(get_csession_token_from_cookie),
    dbops: DatabaseOperations = Depends()
) -> JSONResponse:
    """Get a patient info from database"""
    patient_info = dict(dbops.get_patient_by_id(patient_id))
    if not patient_info:
        PATIENT_LOGGER.info(f"Patient with id {patient_id} not found on the database")
        return JSONResponse({"message": "Paciente no encontrado"}, status_code=HTTP_404_NOT_FOUND)
    pathologies_list = dbops.get_pathologies_by_patient(patient_id)
    patient_info['pathologies'] = [dict(row) for row in pathologies_list]
    patient_info['id'] = patient_id
    patient_info = service.convert_patient_images(patient_info)
    return JSONResponse(content=patient_info, status_code=HTTP_200_OK)


@patients_bp.get("/{patient_id}/pathology/{pathology_id}")
async def get_pathology(
    request: Request,
    patient_id: Annotated[int, Path(description="Patient ID")],
    pathology_id: Annotated[int, Path(description="Pathology ID")],
    session = Depends(get_csession_token_from_cookie),
    dbops: DatabaseOperations = Depends()
) -> JSONResponse:
    """Get info about a pathology"""
    patient_info = dict(dbops.get_patient_by_id(patient_id))
    if not patient_info:
        PATIENT_LOGGER.info(f"Patient with id {patient_id} not found on the database")
        return JSONResponse({"message": "Paciente no encontrado"}, status_code=HTTP_404_NOT_FOUND)
    pathology = dbops.get_pathology_by_id(pathology_id, patient_id)
    if not pathology:
        PATIENT_LOGGER.info(f"Pathology with id {pathology_id} for patient with id {patient_id} not found on the database")
        return JSONResponse({"message": "Patologia no encontrada"}, status_code=HTTP_404_NOT_FOUND)
    pathology = dict(pathology)
    # Convert images to base 64
    if isinstance(pathology['pet_img'], bytes):
        pathology['pet_img'] = service.convert_png_bytes_to_base64(pathology["pet_img"])
    if isinstance(pathology['corrected_img'], bytes):
        pathology['corrected_img'] = service.convert_png_bytes_to_base64(pathology["corrected_img"])
    return JSONResponse(content=pathology, status_code=HTTP_200_OK)


@patients_bp.post("/{patient_id}/pathology")
async def add_new_pathology(
    request: Request,
    patient_id: Annotated[int, Path(description="Patient ID")],
    data: models.NewPathologyModel,
    session = Depends(get_csession_token_from_cookie),
    dbops: DatabaseOperations = Depends()
) -> JSONResponse:
    """Add a new pathology to the database"""
    patient_info = dict(dbops.get_patient_by_id(patient_id))
    if not patient_info:
        PATIENT_LOGGER.info(f"Patient with id {patient_id} not found on the database")
        return JSONResponse({"message": "Paciente no encontrado"}, status_code=HTTP_404_NOT_FOUND)
    new_pathology = dbops.create_pathology(
        patient_id=patient_id,
        description=data.description,
        doctor_notes=data.notes,
        pet_img=data.pet_img,
        corrected_img=data.corrected_img
    )
    if not new_pathology:
        PATIENT_LOGGER.info(f"Error creating new patholgy for patient with id {patient_id}.")
        return JSONResponse({"message": "Error al crear la patologia"}, status_code=HTTP_500_INTERNAL_SERVER_ERROR)
    return JSONResponse({"pathology_id": new_pathology}, status_code=HTTP_200_OK)


@patients_bp.post("/{patient_id}/pathology/{pathology_id}/image")
async def upload_image(
    request: Request,
    patient_id: Annotated[int, Path(description="Patient ID")],
    pathology_id: Annotated[int, Path(description="Pathology ID")],
    image: Annotated[UploadFile, File(description="Niftii image")],
    session = Depends(get_csession_token_from_cookie),
    dbops: DatabaseOperations = Depends(),
) -> JSONResponse:
    """Upload a PET image, process and store both the original and corrected images on db"""
    patient_info = dict(dbops.get_patient_by_id(patient_id))
    if not patient_info:
        PATIENT_LOGGER.info(f"Patient with id {patient_id} not found on the database")
        return JSONResponse({"message": "Paciente no encontrado"}, status_code=HTTP_404_NOT_FOUND)

    # Validate file type
    if not image.filename.lower().endswith(('.nii', '.nii.gz')):
        return JSONResponse(
            content={"message": "Invalid file type. Only .nii and .nii.gz files are allowed."},
            status_code=HTTP_400_BAD_REQUEST
        )
    # Process image
    try:
        both_images = await service.get_one_niftii_slice(image) 
    except Exception as e:
        return JSONResponse(
            content={"message": f"Failed to process and apply correction to PET image: {str(e)}"},
            status_code=HTTP_500_INTERNAL_SERVER_ERROR
        )
    # Update original image on db
    try:        
        success = dbops.update_original_pet_image(pathology_id, both_images['NAC_PET'])
    except Exception as e:
        return JSONResponse(
            content={"message": f"Failed to upload NAC PET: {str(e)}"},
            status_code=HTTP_500_INTERNAL_SERVER_ERROR
        )
    if not success:
        return JSONResponse(
            status_code=500,
            content={"error": "Pathology record not found or NAC update failed"}
        )
    # Update corrected image on db
    try:       
        success = dbops.update_corrected_pet_image(pathology_id, both_images['AC_PET'])
    except Exception as e:
        return JSONResponse(
            content={"message": f"Failed to upload AC PET: {str(e)}"},
            status_code=HTTP_500_INTERNAL_SERVER_ERROR
        )
    if not success:
        return JSONResponse(
            status_code=500,
            content={"error": "Pathology record not found or AC update failed"}
        )
    return JSONResponse(
        content={
            "message": "PET corrected succesfully",
            "pathology_id": pathology_id,
            "NAC_filename": image.filename,
            "NAC_size": len(both_images['NAC_PET']),
            "AC_size": len(both_images['AC_PET'])
        },
        status_code=HTTP_200_OK
    )
