
document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("upload-form");
    const fileInput = document.getElementById("fileUpload");
  
    uploadForm.addEventListener("submit", (event) => {
      event.preventDefault(); // Prevent form submission
  
      const file = fileInput.files[0];
  
      // Ensure a file is selected
      if (!file) {
        alert("Please select a file before uploading.");
        return;
      }
  
      // Store the file name in sessionStorage
      sessionStorage.setItem("uploadedFileName", file.name);
  
      // Simulate successful upload
      alert("File uploaded successfully! Redirecting...");
      window.location.href = "/upload";
    });
  });
