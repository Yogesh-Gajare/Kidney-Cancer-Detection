
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

// document.addEventListener("DOMContentLoaded", () => {
//     const uploadForm = document.getElementById("upload-form");
//     const fileInput = document.getElementById("fileUpload");

//     uploadForm.addEventListener("submit", async (event) => {
//         event.preventDefault(); // Prevent form submission

//         const file = fileInput.files[0];

//         // Ensure a file is selected
//         if (!file) {
//             alert("Please select a file before uploading.");
//             return;
//         }

//         // Store the file name in sessionStorage
//         sessionStorage.setItem("uploadedFileName", file.name);

//         // Create FormData object to send the file
//         const formData = new FormData();
//         formData.append('dataset', file);  // Append the file to FormData

//         try {
//             // Send the file to the backend server using POST
//             const response = await fetch('/upload', {
//                 method: 'POST',
//                 body: formData  // Send the FormData with the file
//             });

//             const result = await response.json();

//             // Handle the response based on prediction result
//             if (response.ok) {
//                 alert("File uploaded and processed successfully!");
//                 console.log(result);  // Print the prediction result to the console
//                 window.location.href = "upload.html"; // Redirect after successful upload
//             } else {
//                 alert("Error uploading file: " + result.error);
//             }
//         } catch (err) {
//             console.error('Error uploading file:', err);
//             alert('Failed to upload file. Please try again.');
//         }
//     });
// });
