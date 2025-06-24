// Populate patient layout
async function showPatientLayout(patientId, pathologyId) {
    const patientLayout = document.getElementById('patientInfoLayout');
    patientLayout.style.display = 'block';

    try {
      const response = await fetch(`/patient/${patientId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        }
      });

      if (!response.ok) {
        const errorData = await response.json();
        alert(`Error: ${errorData.message || 'Error desconocido'}`);
        return;
      }

      const data = await response.json();

      if (data) {
        await emptyImageDisplay();
        await populatePatientInfo(data);
        await populatePatientPathologies(data);
        if(pathologyId != null)
          selectPathology(pathologyId);
      }
      else{
        alert(data.message);
      }
    } catch (error) {
      alert(`Error de red: ${error.message}`);
      console.error(error);
    }
}


async function populatePatientInfo(data) {
    const patientNameInput = document.getElementById('patientName');
    const patientAgeInput = document.getElementById('patientAge');
    const patientDniInput = document.getElementById('patientDni');
    patientNameInput.value = `${data.name} ${data.last_name}`;
    patientNameInput.setAttribute('data-id', data.id);
    patientAgeInput.value = data.age;
    patientDniInput.value = data.dni;   
}

async function populatePatientPathologies(data) {
    const pathologiesList = document.getElementById('pathologiesList');
    pathologiesList.innerHTML = "";

    if (!data || !data.pathologies.length) {
        pathologiesList.innerHTML = "<li>No hay patologías registradas.</li>";
        return;
    }
    
    data.pathologies.forEach((p) => {
        const li = document.createElement('li');
        li.textContent = p.description;
        li.dataset.pathologyId = p.id;
        li.tabIndex = 0;
        li.setAttribute('role', 'option');
        li.addEventListener('click', () => selectPathology(p.id));
        li.addEventListener('keydown', e => {
            if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            selectPathology(p.id);
            }
        });
        pathologiesList.appendChild(li);
    });
}

async function emptyImageDisplay() {
    const contents = `
      <div class="column" aria-label="First column">
        <div class="picture-frame-wrapper">
          <h4 class="picture-frame-title">Información</h4>
        
          <h3 id="pathologyTitle">Seleccione una patología</h3>
            <div class="pathology-description" id="pathologyDescription">
          </div>

          <div id="uploadPictureSection" class="upload-section" style="display: none;">
            <label for="uploadPicture" class="upload-label" tabindex="0">Subir imagen</label>
            <input type="file" id="uploadPicture" class="upload-input" accept="image/*" />
          </div>
        </div>
      </div>
      

        <div class="column" aria-label="Second column">
          <div class="picture-frame-wrapper">
                <h4 class="picture-frame-title">PET sin corregir</h4>
                <div class="picture-frame" aria-label="Imagen PET sin corregir">
                    <img id="nonCorrectedPet" alt="Imagen PET sin corregir" src="" style="display:none">
                </div>
            </div>
        </div>
        
        
        <div class="column" aria-label="Third column">
          <div class="picture-frame-wrapper">
                <h4 class="picture-frame-title">PET corregido</h4>
                <div class="picture-frame" aria-label="Imagen PET corregido">
                    <img id="correctedPet" alt="Imagen PET corregido" src="" style="display:none">
                </div>
            </div>
        </div>
      `;

    // Select the section using its ID
    const section = document.getElementById('pathologyDetailSection');

    // Clear the section's contents
    section.innerHTML = '';

    // Assign the newly created contents to the section
    section.innerHTML = contents;

    const uploadPictureInput = document.getElementById('uploadPicture');
    uploadPictureInput.addEventListener('change', async () => {
      await uploadImage(uploadPictureInput);
    });

}

// Handle pathologies
async function newPathology() {
    const modalDescriptionInput = document.getElementById('modalDescription');
    const modalNotesInput = document.getElementById('modalNotes');
    const patientNameInput = document.getElementById('patientName');
    const patientId = patientNameInput.getAttribute('data-id')

    const description = modalDescriptionInput.value.trim();
    if (!description) {
        alert('La descripción es obligatoria.');
        modalDescriptionInput.focus();
        return;
    }
    const notes = modalNotesInput.value.trim();
    try {
      const response = await fetch(`/patient/${patientId}/pathology`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ notes, description })
      });

      if (response.ok) {
        const data = await response.json();

        if (data) {
          hideModal();
          // Automatically select the new pathology
          showPatientLayout(patientId, data.pathology_id);
        }
        else {
          alert(data.message)
        }
      } else {
        const error = await response.json();
        alert(error.detail || 'Error en el inicio de sesión. Verifique sus credenciales.');
      }
    } catch (error) {
      console.error('Error en login:', error);
      alert('Error al comunicarse con el servidor.');
    }
}


async function getPathology(pathologyId) {
    const patientNameInput = document.getElementById('patientName');
    const patientId = patientNameInput.getAttribute('data-id')

    try {
      const response = await fetch(`/patient/${patientId}/pathology/${pathologyId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        }
      });

      if (!response.ok) {
        const errorData = await response.json();
        alert(`Error: ${errorData.message || 'Error desconocido'}`);
        return;
      }

      const data = await response.json();

      if (data) {
        return data;
      }
      else{
        alert(data.message);
      }
    } catch (error) {
      alert(`Error de red: ${error.message}`);
      console.error(error);
    }
}


function clearActivePathology() {
    Array.from(pathologiesList.children).forEach((li) => {
        li.classList.remove("active");
    });
}

async function selectPathology(id) {
    const pathologiesList = document.getElementById('pathologiesList');
    const pathologyTitle = document.getElementById('pathologyTitle');
    const pathologyDescription = document.getElementById('pathologyDescription');
    const nonCorrectedPet = document.getElementById('nonCorrectedPet');
    const correctedPet = document.getElementById('correctedPet');
    const uploadPictureInput = document.getElementById('uploadPicture');
    const uploadPetSection = document.getElementById('uploadPictureSection');

    const pathology = await getPathology(id);

    // Highlight active pathology
    clearActivePathology();
    Array.from(pathologiesList.children).forEach((li) => {
        if (li.dataset.pathologyId == id) {
            li.classList.add("active");
        }
    });

    uploadPetSection.style.display = 'block';

    pathologyTitle.textContent = pathology.description;
    pathologyDescription.textContent = pathology.doctor_notes;

    if (pathology.pet_img) {
        nonCorrectedPet.src = pathology.pet_img;
        nonCorrectedPet.style.display = "block";
    } else {
        nonCorrectedPet.src = "";
        nonCorrectedPet.style.display = "none";
    }

    if (pathology.corrected_img) {
        correctedPet.src = pathology.corrected_img;
        correctedPet.style.display = "block";
    } else {
        correctedPet.src = "";
        correctedPet.style.display = "none";
    }
    // uploadPictureInput.value = ""; // reset file input
}


// Modal handle
function showModal() {
    const modalDescriptionInput = document.getElementById('modalDescription');
    const modalNotesInput = document.getElementById('modalNotes');
    const modalBackdrop = document.getElementById('modalBackdrop');

    modalDescriptionInput.value = '';
    modalNotesInput.value = '';
    modalBackdrop.classList.add('show');
    modalDescriptionInput.focus();
}

function hideModal() {
    const modalBackdrop = document.getElementById('modalBackdrop');
    modalBackdrop.classList.remove('show');
}


// Handle image upload and process
async function uploadImage(uploadPictureInput) {

    const pathologiesList = document.getElementById('pathologiesList');
    const patientNameInput = document.getElementById('patientName');
    const patientId = patientNameInput.getAttribute('data-id')
    let pathologyId = '';

    Array.from(pathologiesList.children).forEach((li) => {
        if (li.classList.contains('active')) {
            pathologyId = li.dataset.pathologyId;
        }
    });

    if (!uploadPictureInput.files.length) return;
    
    const file = uploadPictureInput.files[0];
    
    // Validate NIfTI file types (matching your FastAPI validation)
    if (!file.name.toLowerCase().endsWith('.nii') && !file.name.toLowerCase().endsWith('.nii.gz')) {
        alert('Por favor, suba un archivo NIfTI válido (.nii o .nii.gz).');
        return;
    }
    
    // Check if we have the required IDs
    if (!patientId || !pathologyId) {
        alert('Error: No se ha seleccionado un paciente o patología.');
        return;
    }
    
    try {
        // Create FormData for file upload
        const formData = new FormData();
        formData.append('image', file);
        
        // Upload the file
        const response = await fetch(`/patient/${patientId}/pathology/${pathologyId}/image`, {
            method: 'POST',
            credentials: 'include',
            body: formData // Don't set Content-Type header - let browser set it with boundary
        });
        
        if (response.ok) {
            const data = await response.json();
            selectPathology(pathologyId);    
        } else {
            const error = await response.json();
            alert(error.message || 'Error al subir la imagen.');
        }
        
    } catch (error) {
        console.error('Error uploading file:', error);
        alert('Error al comunicarse con el servidor.');
    }
}

// // Handle image download button
// downloadImageBtn.addEventListener('click', () => {
//     if (!pathologyImage.src) return;
//     const a = document.createElement('a');
//     a.href = pathologyImage.src;
//     a.download = currentPathology ? `${currentPathology.name}.png` : 'image.png';
//     document.body.appendChild(a);
//     a.click();
//     document.body.removeChild(a);
// });
