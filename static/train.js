document.addEventListener("DOMContentLoaded", () => {
    // Get the uploaded file name from sessionStorage (or use another storage method)
    const fileName = sessionStorage.getItem("uploadedFileName");

    if (fileName) {
      const pdfViewer = document.getElementById("pdfViewer");

      // Construct the file URL assuming the file is saved in an 'uploads/' directory
      const fileUrl = `uploads/${fileName}`;

      // Check if the file exists (optional: add an alert or logging here)
      fetch(fileUrl).then(response => {
        if (response.ok) {
          // Set the file URL for the iframe tag to display the PDF
          pdfViewer.setAttribute("src", fileUrl);
        } else {
          alert("PDF file not found or couldn't be loaded.");
        }
      }).catch(error => {
        console.error("Error loading PDF:", error);
        alert("An error occurred while loading the PDF.");
      });
    } else {
      // Redirect to upload page if no file is uploaded
      alert("No dataset uploaded. Redirecting to the upload page.");
      window.location.href = "/upload";
    }

    // Train Dataset button click event
    const trainButton = document.getElementById("train-button");
    trainButton.addEventListener("click", () => {
      alert("Training the dataset... (This is a placeholder action)");
      // Add your dataset training logic here
    });
  });