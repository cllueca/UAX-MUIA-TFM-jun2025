//Handle theme color change
const toggleBtn = document.getElementById("themeToggle");
const body = document.body;

toggleBtn.addEventListener("click", () => {
  body.classList.toggle("dark-theme");
  body.classList.toggle("light-theme");
});


// Handle login
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('loginForm');
  if (!form) return;

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    
    if (form.id !== 'loginForm'){
      return;
    }

    const emailInput = form.querySelector('input[name="email"]') || form.querySelector('input[type="text"]');
    const passwordInput = form.querySelector('input[name="password"]') || form.querySelector('input[type="password"]');

    const email = emailInput ? emailInput.value.trim() : '';
    const password = passwordInput ? passwordInput.value : '';

    if (!email || !password) {
      alert('Por favor, complete ambos campos.');
      return;
    }

    try {
      const response = await fetch('/doctor/login', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ email, password })
      });

      if (response.ok) {
        const data = await response.json();

        if (data.success) {
          window.location.assign("/home");
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
  });
});



// Handle logout
async function logout() {
  const sessionToken = sessionStorage.getItem("sessionToken");
  try {
  const response = await fetch("/doctor/logout", {
    method: "GET",
    credentials: "include",
    headers: {
      "Accept": "application/json"
    }
  });

  if (response.ok) {
      const data = await response.json();

      if (data.success) {
        window.location.assign("/");
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


// Handle password change
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('loginForm');
  const toggleLink = document.getElementById('forgotPasswordLink');

  // Store original form HTML to restore later
  const originalFormHTML = form.innerHTML;
  const originalFormId = form.id;
  const originalLinkText = toggleLink.textContent;

  let forgotPasswordMode = false;

  // Initialize eye toggle for login form password input
  function initLoginPasswordToggle() {
    const passwordWrapper = form.querySelector('.password-wrapper');
    if (!passwordWrapper) return;

    const passwordInput = passwordWrapper.querySelector('input[type="password"], input[type="text"]');
    if (!passwordInput) return;

    // Remove old toggle icon if any
    const existingToggle = passwordWrapper.querySelector('.toggle-visibility');
    if (existingToggle) {
      existingToggle.remove();
    }

    // Create toggle icon
    const toggleIcon = document.createElement('span');
    toggleIcon.className = 'material-icons toggle-visibility';
    toggleIcon.setAttribute('role', 'button');
    toggleIcon.setAttribute('tabindex', '0');
    toggleIcon.setAttribute('aria-label', 'Mostrar contraseña');
    toggleIcon.setAttribute('aria-pressed', 'false');
    toggleIcon.title = 'Mostrar contraseña';
    toggleIcon.textContent = 'visibility';

    // Toggle function
    function togglePassword() {
      const isPassword = passwordInput.type === 'password';
      passwordInput.type = isPassword ? 'text' : 'password';
      toggleIcon.textContent = isPassword ? 'visibility_off' : 'visibility';
      toggleIcon.setAttribute('aria-label', isPassword ? 'Ocultar contraseña' : 'Mostrar contraseña');
      toggleIcon.setAttribute('aria-pressed', isPassword ? 'true' : 'false');
      toggleIcon.title = isPassword ? 'Ocultar contraseña' : 'Mostrar contraseña';
    }

    toggleIcon.addEventListener('click', togglePassword);
    toggleIcon.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        togglePassword();
      }
    });

    passwordWrapper.appendChild(toggleIcon);
  }

  // Function to create password input with toggle icon for forgot password form
  function createPasswordInput(name, placeholder) {
    const wrapper = document.createElement('div');
    wrapper.className = 'password-wrapper';

    const passwordInput = document.createElement('input');
    passwordInput.type = 'password';
    passwordInput.name = name;
    passwordInput.placeholder = placeholder;
    passwordInput.required = true;

    const toggleIcon = document.createElement('span');
    toggleIcon.className = 'material-icons toggle-visibility';
    toggleIcon.setAttribute('role', 'button');
    toggleIcon.setAttribute('tabindex', '0');
    toggleIcon.setAttribute('aria-label', 'Mostrar contraseña');
    toggleIcon.setAttribute('aria-pressed', 'false');
    toggleIcon.title = 'Mostrar contraseña';
    toggleIcon.textContent = 'visibility';

    toggleIcon.addEventListener('click', () => {
      const isPassword = passwordInput.type === 'password';
      passwordInput.type = isPassword ? 'text' : 'password';
      toggleIcon.textContent = isPassword ? 'visibility_off' : 'visibility';
      toggleIcon.setAttribute('aria-label', isPassword ? 'Ocultar contraseña' : 'Mostrar contraseña');
      toggleIcon.setAttribute('aria-pressed', isPassword ? 'true' : 'false');
      toggleIcon.title = isPassword ? 'Ocultar contraseña' : 'Mostrar contraseña';
    });

    toggleIcon.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        toggleIcon.click();
      }
    });

    wrapper.appendChild(passwordInput);
    wrapper.appendChild(toggleIcon);
    return wrapper;
  }

  toggleLink.addEventListener('click', function(event) {
    event.preventDefault();

    if (!forgotPasswordMode) {
      // Switch to forgot password form
      form.innerHTML = '';
      form.id = 'changePasswordForm';

      // Create new input fields
      const emailInput = document.createElement('input');
      emailInput.type = 'text';
      emailInput.name = 'email';
      emailInput.placeholder = 'user@fakehospital.com';
      emailInput.required = true;

      const oldPasswordWrapper = createPasswordInput('old_password', 'Contraseña Anterior');
      const newPasswordWrapper = createPasswordInput('new_password', 'Nueva Contraseña');

      const buttonGroupDiv = document.createElement('div');
      buttonGroupDiv.className = 'button-group';

      const confirmButton = document.createElement('button');
      confirmButton.className = 'login_button';
      confirmButton.type = 'submit';
      confirmButton.textContent = 'Confirmar';

      buttonGroupDiv.appendChild(confirmButton);

      form.appendChild(emailInput);
      form.appendChild(oldPasswordWrapper);
      form.appendChild(newPasswordWrapper);
      form.appendChild(buttonGroupDiv);

      // Change link text to allow going back
      toggleLink.textContent = 'Iniciar Sesión';

      // Add submit event handler for confirm
      form.addEventListener('submit', onForgotPasswordSubmit);

      forgotPasswordMode = true;
    } else {
      // Restore original login form
      form.innerHTML = originalFormHTML;
      toggleLink.textContent = originalLinkText;
      form.id = originalFormId;
      forgotPasswordMode = false;

      // Remove submit listener to avoid duplicates in case of toggling multiple times
      form.removeEventListener('submit', onForgotPasswordSubmit);

      // Re-initialize toggle icon and event for original login password input
      initLoginPasswordToggle();
    }
  });

  // Async submit handler for forgot password form
  async function onForgotPasswordSubmit(event) {
    event.preventDefault();

    // Retrieve values
    const email = form.querySelector('input[name="email"]')?.value.trim();
    const oldPassword = form.querySelector('input[name="old_password"]')?.value;
    const newPassword = form.querySelector('input[name="new_password"]')?.value;

    if (!email || !oldPassword || !newPassword) {
      alert('Por favor, completa todos los campos.');
      return;
    }

    // Build request payload, sending email and new password as "password" per your example
    try {
      const response = await fetch(`/doctor/${email}/reset-password`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ email, old_password: oldPassword, new_password: newPassword }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        alert(`Error: ${errorData.message || 'Error desconocido'}`);
        return;
      }

      const data = await response.json();
      
      if (data.success) {
        // TODO: Show a success message
        toggleLink.click();
      }
      else{
        alert(data.message);
      }
    } catch (error) {
      alert(`Error de red: ${error.message}`);
      console.error(error);
    }
  }

  // Initialize on page load in case login form is present initially
  initLoginPasswordToggle();
});

