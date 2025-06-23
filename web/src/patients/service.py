# Builtins
import base64
from io import BytesIO
import os
import tempfile
# Installed
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
# Local
from src.config import CONFIG
from src.core.vision_models import VisionModel
# Types


def convert_png_bytes_to_base64(data):
    """Convert PNG bytes to PNG base64."""
    base64_str = base64.b64encode(data).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"


def convert_patient_images(patient_info):
    """Prepare patient info for JSON response."""
    for pathology in patient_info['pathologies']:
        if 'pet_img' in pathology and isinstance(pathology['pet_img'], bytes):
            pathology['pet_img'] = convert_png_bytes_to_base64(pathology['pet_img'])
        if 'corrected_img' in pathology and isinstance(pathology['corrected_img'], bytes):
            pathology['corrected_img'] = convert_png_bytes_to_base64(pathology['corrected_img'])
    return patient_info


async def get_one_niftii_slice(image):
    unet = VisionModel(model_name=CONFIG.MODEL_NAME)
    unet.load()

    # Read the uploaded NIfTI file as bytes
    image_bytes = await image.read()

    # Write the uploaded bytes to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_input_file:
        tmp_input_file.write(image_bytes)
        tmp_input_file.flush()
        input_path = tmp_input_file.name

    try:
        # Load the image using nibabel from the file
        nifti_img = nib.load(input_path)

        # Extract data
        data = nifti_img.get_fdata()

        # Extract slices 68 to 76 (inclusive)
        slices_to_predict = data[:, :, 68:76]  # Assumes slices are along axis 2 (axial view)

        # Take only the 4 slice from that range
        mid_slice = slices_to_predict[:, :, 4]

        predicted_image = unet.make_prediction(slices_to_predict)
        is_3d_data = (len(predicted_image.shape) == 5)

        if is_3d_data:
            # For 3D data, get the first slice
            predicted_mid_slice = predicted_image[0, 0].squeeze()[4]
        else:                    
            # For 2D data
            predicted_mid_slice = predicted_image[0].squeeze()[4]              

        png_images = {
            'NAC_PET': nifti2pngbytes(mid_slice, input_path),
            'AC_PET': nifti2pngbytes(predicted_mid_slice, input_path)
        }
        unet.unload()
        return png_images

    except Exception as e:
        # Clean up in case of error
        if os.path.exists(input_path):
            os.remove(input_path)
        raise e


def nifti2pngbytes(image, input_path):

    # Normalize the slice data to 0-255 range for PNG
    normalized_slice = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Convert to PNG bytes
    plt.figure(figsize=(10, 10))
    plt.imshow(normalized_slice, cmap='gray')
    plt.axis('off')
    
    # Save to bytes buffer as PNG
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    png_bytes = buffer.getvalue()
    buffer.close()
    
    # Clean up temp files
    if os.path.exists(input_path):
        os.remove(input_path)
    
    return png_bytes
