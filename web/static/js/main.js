// Handle patient load and list toggle
function togglePatients(patientMenuOption) {
    const pacientesList = document.getElementById('pacientesList');
    pacientesList.style.display = pacientesList.style.display === 'none' ? 'block' : 'none';
    patientMenuOption.classList.toggle('active');
}

function filterPatients() {
    const input = document.getElementById('patientSearchInput');
    const filter = input.value.toLowerCase();
    const pacientes = document.getElementById('pacientes');
    const li = pacientes.getElementsByTagName('li');

    for (let i = 0; i < li.length; i++) {
        const txtValue = li[i].textContent || li[i].innerText;
        li[i].style.display = txtValue.toLowerCase().indexOf(filter) > -1 ? '' : 'none';
    }
}

async function loadPatients() {
    try {
        const response = await fetch('/doctor/patient/all', {
            method: 'GET',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });

        if (response.ok) {
            const data = await response.json();
            const pacientesList = document.getElementById('pacientes');
            pacientesList.innerHTML = ''; // Clear existing list items
            // Populate the list with patient names
            data.forEach(patient => {
                const li = document.createElement('li');
                li.setAttribute('role', 'option');
                li.setAttribute('tabindex', '-1');
                // li.setAttribute('data-id', patient.id);
                li.textContent = `${patient.name} ${patient.last_name}`; // Adjust this if the property name is different

                li.addEventListener('click', async () => {
                    await showPatientLayout(patient.id);
                });

                pacientesList.appendChild(li);
            });
        } else {
            const error = await response.json();
            alert(error.detail || 'Error al obtener pacientes.');
        }
    } catch (error) {
        console.error('Error en login:', error);
        alert('Error al comunicarse con el servidor.');
    }
}


document.addEventListener('DOMContentLoaded', () => {
    const patientMenuOption = document.getElementById('patientsMenuOption');
    const patientSearch = document.getElementById('patientSearchInput');
    const addPathologyButton = document.getElementById('addPathologyBtn');
    const uploadPictureInput = document.getElementById('uploadPicture');

    // Modal elements
    const modalBackdrop = document.getElementById('modalBackdrop');
    const addPathologyForm = document.getElementById('addPathologyForm');
    const modalCancelBtn = document.getElementById('modalCancelBtn');


    togglePatients(patientMenuOption);
    loadPatients();

    patientMenuOption.addEventListener('click', () => {
        togglePatients(patientMenuOption)
    });

    patientSearch.addEventListener('keyup', () => {
        filterPatients();
    });

    addPathologyButton.addEventListener('click', showModal);
    modalCancelBtn.addEventListener('click', hideModal);
    modalBackdrop.addEventListener('click', (e) => {
        if (e.target === modalBackdrop) {
            hideModal();
        }
    });
    addPathologyForm.addEventListener('submit', (e) => {
        e.preventDefault();
        newPathology();
    });

    uploadPictureInput.addEventListener('change', async () => {
        await uploadImage(uploadPictureInput);
    });
});
