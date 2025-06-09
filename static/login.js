

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
