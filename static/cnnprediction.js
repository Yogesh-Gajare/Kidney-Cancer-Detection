
window.onload = function() {
    const base64Image = localStorage.getItem("uploadedImage");

    if (base64Image) {
        // Set the uploaded image as the source of the image tag
        document.getElementById('uploadedImage').src = base64Image;
    } else {
        alert("No image uploaded. Please upload an image first.");
        window.location.href = "/upload"; // Redirect to upload page if no image
    }
};

document.getElementById('predictBtn').addEventListener('click', function() {
    // Retrieve the image from localStorage (stored as a base64 string)
    const base64Image = localStorage.getItem("uploadedImage");

    // If no image is available, show an alert
    if (!base64Image) {
        alert("⚠️ No image uploaded. Please upload an image first.");
        return;
    }

    // Create a FormData object and append the base64 image as a string
    let formData = new FormData();
    formData.append("image", base64Image);

    // Send the FormData to the backend for prediction
    fetch("/predict_cnn", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log("✅ Prediction received:", data.prediction);

        // Display prediction result
        document.getElementById("predictionResult").style.display = "block";
        document.getElementById("predictionText").innerText = "Prediction: " + data.prediction;

        // Display additional database info if available
        if (data.database_info) {
            let infoText = "Additional Info: \n";
            for (let key in data.database_info) {
                infoText += `${key}: ${data.database_info[key]}\n`;
            }
            document.getElementById("databaseInfo").innerText = infoText;
        }
    })
    .catch(error => {
        console.error("❌ Error:", error);
        alert("⚠️ Error processing image. Please try again.");
    });
});

