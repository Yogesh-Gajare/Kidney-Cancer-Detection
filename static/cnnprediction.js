// // When the page loads, get the uploaded image from localStorage
// window.onload = function() {
//     const imagePath = localStorage.getItem("uploadedImage");
  
//     if (imagePath) {
//       // Set the uploaded image as the source of the image tag
//       document.getElementById('uploadedImage').src = imagePath;
//     } else {
//       alert("No image uploaded. Please upload an image first.");
//       window.location.href = "/upload"; // Redirect to upload page if no image
//     }
//   };
  
//   // Simulate prediction when the button is clicked
//   document.getElementById('predictBtn').addEventListener('click', function() {
//     const prediction = "Positive for Kidney Cancer"; // Example prediction result
  
//     // Display the prediction result
//     document.getElementById('predictionResult').style.display = "block";
//     document.getElementById('predictionText').innerText = prediction;
//   });
  



// // Ensure the uploaded image persists on cnnprediction.html
// window.onload = function () {
//     const imagePath = localStorage.getItem("uploadedImage");

//     if (imagePath) {
//         document.getElementById('uploadedImage').src = imagePath;
//     } else {
//         console.warn("No image found in localStorage. Redirecting...");
//         window.location.href = "/cnnprediction"; // Redirect to upload page if no image
//     }
// };

// // Function to handle image upload and preview
// document.getElementById('fileInput').onchange = function (event) {
//     let file = event.target.files[0];

//     if (file) {
//         let reader = new FileReader();
//         reader.onload = function () {
//             let preview = document.getElementById('uploadedImage');
//             preview.src = reader.result;
//             preview.style.display = 'block';

//             // Store image in LocalStorage for use in cnnprediction.html
//             localStorage.setItem("uploadedImage", reader.result);
//         };
//         reader.readAsDataURL(file);
//     }
// };

// // Function to upload image and redirect to prediction page
// function uploadImage() {
//     let fileInput = document.getElementById('fileInput').files[0];

//     if (!fileInput) {
//         alert("⚠️ Please select an image!");
//         return;
//     }

//     let formData = new FormData();
//     formData.append("file", fileInput);

//     // Store image URL before redirection
//     let imageUrl = URL.createObjectURL(fileInput);
//     localStorage.setItem("uploadedImage", imageUrl);

//     window.location.href = "/cnnprediction"; // Redirect to prediction page
// }

// // Function to handle prediction button click
// document.getElementById('predictBtn').addEventListener('click', function () {
//     let fileInput = document.getElementById('fileInput').files[0];

//     if (!fileInput) {
//         alert("⚠️ Please select an image first!");
//         return;
//     }

//     let formData = new FormData();
//     formData.append("file", fileInput);

//     fetch("/predict", {
//         method: "POST",
//         body: formData
//     })
//         .then(response => response.json())
//         .then(data => {
//             console.log("✅ Prediction received:", data.prediction);

//             // Display prediction result on the same page
//             document.getElementById("predictionResult").style.display = "block";
//             document.getElementById("predictionText").innerText = "Prediction: " + data.prediction;
//         })
//         .catch(error => {
//             console.error("❌ Error:", error);
//             alert("⚠️ Error processing image. Please try again.");
//         });
// });




// // When the page loads, get the uploaded image from localStorage
// window.onload = function() {
//     const imagePath = localStorage.getItem("uploadedImage");
  
//     if (imagePath) {
//         // Set the uploaded image as the source of the image tag
//         document.getElementById('uploadedImage').src = imagePath;
//     } else {
//         alert("No image uploaded. Please upload an image first.");
//         window.location.href = "/upload"; // Redirect to upload page if no image
//     }
// };

// // Function to handle prediction request when the button is clicked
// document.getElementById('predictBtn').addEventListener('click', function() {
//     let fileInput = document.getElementById('fileInput') ? document.getElementById('fileInput').files[0] : null;

//     if (!fileInput) {
//         alert("⚠️ Please select an image first!");
//         return;
//     }

//     let formData = new FormData();
//     formData.append("file", fileInput);

//     fetch("/predict", {
//         method: "POST",
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         console.log("✅ Prediction received:", data.prediction);

//         // Display prediction result on the page
//         document.getElementById("predictionResult").style.display = "block";
//         document.getElementById("predictionText").innerText = "Prediction: " + data.prediction;

//         // Show additional database info if available
//         if (data.database_info) {
//             let infoText = "Additional Info: \n";
//             for (let key in data.database_info) {
//                 infoText += `${key}: ${data.database_info[key]}\n`;
//             }
//             document.getElementById("databaseInfo").innerText = infoText;
//         }
//     })
//     .catch(error => {
//         console.error("❌ Error:", error);
//         alert("⚠️ Error processing image. Please try again.");
//     });
// });



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

