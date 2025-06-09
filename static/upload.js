

document.getElementById('uploadForm').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the form from being submitted in the traditional way
  
    const fileInput = document.getElementById('fileUpload');
    const file = fileInput.files[0];
  
    // Check if a file was selected and ensure it is a PNG image
    if (file && file.type === "image/jpeg") {
      const reader = new FileReader();
  
      // Once the file is read, save the image URL in localStorage and redirect to prediction page
      reader.onloadend = function() {
        localStorage.setItem("uploadedImage", reader.result); // Store the image in localStorage
        window.location.href = "/cnnprediction";  // Redirect to prediction page
      };
  
      reader.readAsDataURL(file);  // Read the file as a Data URL (base64 encoded image)
    } else {
      alert('Please upload a valid JPG image.');
    }
  });

  
  