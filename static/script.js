document.addEventListener("DOMContentLoaded", () => {
    const currentPage = window.location.pathname.split("/").pop();
  
    // Add navigation event listeners
    setupNavigation();
  
    // Handle page-specific logic
    switch (currentPage) {
      case "index.html":
        setupIndexPage();  // Logic for index.html
        break;
      case "first.html":
        setupFirstPage();  // Logic for first.html (Login form)
        break;
      case "dataset.html":
        setupDatasetPage();
        break;
      case "preview.html":
        setupPreviewPage();
        break;
      case "upload.html":
        setupUploadPage();
        break;
      case "data.html":
        setupDataPage();  // This part is not needed based on your flow
        break;
    }
  });
  
  // Function to set up navigation across pages
  function setupNavigation() {
    const navLinks = document.querySelectorAll("nav a");
    navLinks.forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const targetPage = link.getAttribute("href");
        window.location.href = targetPage;
      });
    });
  }
  
  // Function for index.html (Homepage with Login)
  function setupIndexPage() {
    console.log("Index page loaded.");
    // The index page just needs to redirect to the first page (login form) when clicked
    const loginButton = document.querySelector("#loginLink");
    if (loginButton) {
      loginButton.addEventListener("click", () => {
        window.location.href = "first.html";  // Redirect to first.html
      });
    }
  }
  
  // Function for first.html (Login Form)
  function setupFirstPage() {
    console.log("First page loaded.");
    
    const loginForm = document.querySelector("form");
    loginForm.addEventListener("submit", (e) => {
      e.preventDefault();
  
      // Collect form data (username and password)
      const username = document.getElementById("username").value.trim();
      const password = document.getElementById("password").value.trim();
  
      // Check the credentials (hardcoded for demo)
      if (username === "admin" && password === "password") {
        alert("Login successful! Redirecting to the next page...");
        // If login is successful, redirect to the next page (first.html)
        window.location.href = "/signup";  // Redirection to first.html after successful login
      } else {
        alert("Invalid username or password. Please try again.");
      }
    });
  }
  
  // Function for dataset.html (Dataset Upload)
  function setupDatasetPage() {
    console.log("Dataset page loaded.");
    const uploadForm = document.querySelector("form");
    uploadForm.addEventListener("submit", (e) => {
      e.preventDefault();
  
      const fileInput = document.getElementById("fileUpload");
      const file = fileInput.files[0];
  
      if (file) {
        // Store the dataset file name in sessionStorage
        sessionStorage.setItem("datasetFile", file.name);
        alert(`File "${file.name}" uploaded successfully!`);
      } else {
        alert("Please select a dataset file to upload.");
      }
    });
  }
  
  // Function for preview.html (Preview Submitted Data)
  function setupPreviewPage() {
    console.log("Preview page loaded.");
    const previewSection = document.querySelector(".hero-content");
  
    // Retrieve and display submitted form data
    const formData = JSON.parse(sessionStorage.getItem("formData"));
    if (formData) {
      previewSection.innerHTML += "<h3>Submitted Data:</h3>";
      const list = document.createElement("ul");
  
      for (const [key, value] of Object.entries(formData)) {
        const listItem = document.createElement("li");
        listItem.textContent = `${key}: ${value}`;
        list.appendChild(listItem);
      }
  
      previewSection.appendChild(list);
    } else {
      previewSection.innerHTML += "<p>No data submitted yet.</p>";
    }
  }
  
  // Function for upload.html (Upload a File)
  function setupUploadPage() {
    console.log("Upload page loaded.");
    const uploadForm = document.querySelector("form");
    uploadForm.addEventListener("submit", (e) => {
      e.preventDefault();
  
      const fileInput = document.getElementById("fileUpload");
      const file = fileInput.files[0];
  
      if (file) {
        alert(`File "${file.name}" uploaded successfully!`);
      } else {
        alert("Please select a file to upload.");
      }
    });
  }
  
  // Function for data.html (This is not required based on your updated flow)
  function setupDataPage() {
    console.log("Data page loaded.");
  }
  