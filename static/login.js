// // Default credentials
// const defaultUsername = "admin";
// const defaultPassword = "12345";

// // Get the login form element
// const loginForm = document.getElementById('login-form');

// // Add an event listener for form submission
// loginForm.addEventListener('submit', function (event) {
//     event.preventDefault(); // Prevent the default form submission

//     // Retrieve user input
//     const username = document.getElementById('username').value;
//     const password = document.getElementById('password').value;

//     // Validate user input against default credentials
//     if (username === defaultUsername && password === defaultPassword) {
//         alert('Login successful! Redirecting to dataset page...');
//         window.location.href = "/upload"; // Redirect to dataset.html
//     } else {
//         alert('Invalid username or password. Please try again.');
//     }
// });


document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');
    const messageDiv = document.getElementById('message');

    if (!loginForm) return;

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const username = loginForm.username.value.trim();
        const password = loginForm.password.value;

        if (!username || !password) {
            messageDiv.textContent = 'Please enter both username and password.';
            messageDiv.style.color = 'red';
            return;
        }

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, password })
            });

            const result = await response.json();

            if (result.success) {
                messageDiv.textContent = 'Login successful! Redirecting...';
                messageDiv.style.color = 'green';
                setTimeout(() => {
                    window.location.href = '/upload';
                }, 1000);
            } else {
                messageDiv.textContent = result.message || 'Login failed.';
                messageDiv.style.color = 'red';
            }
        } catch (error) {
            messageDiv.textContent = 'An error occurred. Please try again.';
            messageDiv.style.color = 'red';
            console.error('Login error:', error);
        }
    });
});
