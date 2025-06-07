

// document.addEventListener("DOMContentLoaded", () => {
//     const predictionForm = document.getElementById("prediction-form");
//     const predictButton = document.getElementById("predict-btn");
//     const predictionResult = document.getElementById("prediction-result");

//     // Load stored form data if available
//     loadFormData();

//     // Attach event listeners
//     if (predictionForm) {
//         predictionForm.addEventListener("submit", async (event) => {
//             event.preventDefault();
//             await handlePrediction();
//         });
//     }

//     if (predictButton) {
//         predictButton.addEventListener("click", async (event) => {
//             event.preventDefault();
//             await handlePrediction();
//         });
//     }

//     async function handlePrediction() {
//         const formData = collectFormData();

//         // Validate numeric fields
//         const numericFields = ["age", "bp", "bgr", "bu", "sc", "sod", "hemo", "pcv", "wc", "rc"];
//         for (let field of numericFields) {
//             if (formData[field] && isNaN(formData[field])) {
//                 displayResult(`Invalid input: ${field} must be a number`, "error");
//                 return;
//             }
//         }

//         // Convert categorical values (Yes/No to 1/0)
//         // const categoricalFields = ["htn", "dm"];
//         // categoricalFields.forEach(field => {
//         //     if (formData[field] === "Yes") {
//         //         formData[field] = 1;
//         //     } else if (formData[field] === "No") {
//         //         formData[field] = 0;
//         //     }
//         // });

//         // Ensure categorical fields are sent as "Yes"/"No"
//         const categoricalFields = ["htn", "dm"];
//         categoricalFields.forEach(field => {
//             formData[field] = document.getElementById(field).value;  // Keep it as "Yes"/"No"
//         });


//         try {
//             const response = await fetch("/predict", {
//                 method: "POST",
//                 headers: { "Content-Type": "application/json" },
//                 body: JSON.stringify(formData),
//             });

//             const result = await response.json();
//             localStorage.setItem("formData", JSON.stringify(formData));

//             if (response.ok) {
//                 localStorage.setItem("prediction", result.prediction);
//                 displayResult(`Prediction: ${result.prediction}`, "success");
//             } else {
//                 displayResult(`Error: ${result.error}`, "error");
//             }
//         } catch (error) {
//             console.error("Error:", error);
//             displayResult("Server error. Please try again later.", "error");
//         }
//     }

//     function collectFormData() {
//         let formData = {};
//         const inputs = predictionForm.querySelectorAll("input, select");
//         inputs.forEach(input => {
//             formData[input.id] = input.value.trim();
//         });
//         return formData;
//     }

//     function loadFormData() {
//         const storedData = JSON.parse(localStorage.getItem("formData"));
//         if (storedData) {
//             Object.keys(storedData).forEach(key => {
//                 if (document.getElementById(key)) {
//                     document.getElementById(key).value = storedData[key];
//                 }
//             });
//         }
//     }

//     function displayResult(message, type) {
//         predictionResult.innerHTML = `<h3 class="${type}">${message}</h3>`;
//     }
// });



// document.addEventListener("DOMContentLoaded", () => {
//     const predictionForm = document.getElementById("prediction-form");
//     const predictButton = document.getElementById("predict-btn");
//     const predictionResult = document.getElementById("prediction-result");

//     // Load stored form data if available
//     loadFormData();

//     // Attach event listeners
//     if (predictionForm) {
//         predictionForm.addEventListener("submit", async (event) => {
//             event.preventDefault();
//             await handlePrediction();
//         });
//     }

//     if (predictButton) {
//         predictButton.addEventListener("click", async (event) => {
//             event.preventDefault();
//             await handlePrediction();
//         });
//     }

//     async function handlePrediction() {
//         const formData = collectFormData();
//         console.log("🔹 Collected Form Data:", formData);

//         // Validate required fields
//         const requiredFields = ["hemo", "pcv", "rc", "wc", "bu", "sc", "sod", "pot", "dm"];
//         for (let field of requiredFields) {
//             if (!formData[field]) {
//                 displayResult(`⚠️ Missing input: ${field} is required`, "error");
//                 return;
//             }
//         }

//         // Validate numeric fields
//         const numericFields = ["hemo", "pcv", "rc", "wc", "bu", "sc", "sod", "pot"];
//         for (let field of numericFields) {
//             if (isNaN(formData[field])) {
//                 displayResult(`⚠️ Invalid input: ${field} must be a number`, "error");
//                 return;
//             }
//         }

//         try {
//             const response = await fetch("/predict", {
//                 method: "POST",
//                 headers: { "Content-Type": "application/json" },
//                 body: JSON.stringify(formData),
//             });

//             const result = await response.json();
//             console.log("✅ Server Response:", result);

//             if (response.ok) {
//                 localStorage.setItem("formData", JSON.stringify(formData));
//                 localStorage.setItem("prediction", result.prediction);
//                 displayResult(`✅ Prediction: ${result.prediction}`, "success");
//             } else {
//                 displayResult(`❌ Error: ${result.error}`, "error");
//             }
//         } catch (error) {
//             console.error("❌ Server Error:", error);
//             displayResult("⚠️ Server error. Please try again later.", "error");
//         }
//     }

//     function collectFormData() {
//         let formData = {};
//         const inputs = predictionForm.querySelectorAll("input, select");
//         inputs.forEach(input => {
//             formData[input.id] = input.value.trim();
//         });
//         return formData;
//     }

//     function loadFormData() {
//         const storedData = JSON.parse(localStorage.getItem("formData"));
//         if (storedData) {
//             Object.keys(storedData).forEach(key => {
//                 if (document.getElementById(key)) {
//                     document.getElementById(key).value = storedData[key];
//                 }
//             });
//         }
//     }

//     function displayResult(message, type) {
//         predictionResult.innerHTML = `<h3 class="${type}">${message}</h3>`;
//     }
// });



// document.addEventListener("DOMContentLoaded", () => {
//     const predictionForm = document.getElementById("prediction-form");
//     const predictButton = document.getElementById("predict-btn");
//     const predictionResult = document.getElementById("prediction-result");

//     // Attach event listeners
//     if (predictionForm) {
//         predictionForm.addEventListener("submit", async (event) => {
//             event.preventDefault();
//             await handlePrediction();
//         });
//     }

//     if (predictButton) {
//         predictButton.addEventListener("click", async (event) => {
//             event.preventDefault();
//             await handlePrediction();
//         });
//     }

//     async function handlePrediction() {
//         const formData = collectFormData();
//         console.log("🔹 Collected Form Data:", formData);

//         // Validate required fields
//         const requiredFields = ["hemo", "pcv", "rc", "wc", "bu", "sc", "sod", "pot", "dm"];
//         for (let field of requiredFields) {
//             if (!formData[field]) {
//                 displayResult(`⚠️ Missing input: ${field} is required`, "error");
//                 return;
//             }
//         }

//         // Validate numeric fields
//         const numericFields = ["hemo", "pcv", "rc", "wc", "bu", "sc", "sod", "pot"];
//         for (let field of numericFields) {
//             if (isNaN(formData[field])) {
//                 displayResult(`⚠️ Invalid input: ${field} must be a number`, "error");
//                 return;
//             }
//         }

//         try {
//             const response = await fetch("/predict", {
//                 method: "POST",
//                 headers: { "Content-Type": "application/json" },
//                 body: JSON.stringify(formData),
//             });

//             const result = await response.json();
//             console.log("✅ Server Response:", result);

//             if (response.ok) {
//                 displayResult(`✅ Prediction: ${result.prediction}`, "success");
//             } else {
//                 displayResult(`❌ Error: ${result.error}`, "error");
//             }
//         } catch (error) {
//             console.error("❌ Server Error:", error);
//             displayResult("⚠️ Server error. Please try again later.", "error");
//         }
//     }

//     function collectFormData() {
//         let formData = {};
//         const inputs = predictionForm.querySelectorAll("input, select");
//         inputs.forEach(input => {
//             formData[input.id] = input.value.trim();
//         });
//         return formData;
//     }

//     function displayResult(message, type) {
//         predictionResult.innerHTML = `<h3 class="${type}">${message}</h3>`;
//     }
// });




document.addEventListener("DOMContentLoaded", () => {
    const predictionForm = document.getElementById("prediction-form");
    const predictButton = document.getElementById("predict-btn");
    const predictionResult = document.getElementById("prediction-result");

    // Load stored form data if available
    //loadFormData();

    // Attach event listeners
    if (predictionForm) {
        predictionForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            await handlePrediction();
        });
    }

    if (predictButton) {
        predictButton.addEventListener("click", async (event) => {
            event.preventDefault();
            await handlePrediction();
        });
    }

    async function handlePrediction() {
        const formData = collectFormData();
        console.log("🔹 Collected Form Data:", formData);

        // Validate required fields
        const requiredFields = ["hemo", "pcv", "rc", "wc", "bu", "sc", "sod", "pot", "dm"];
        for (let field of requiredFields) {
            if (!formData[field]) {
                displayResult(`⚠️ Missing input: ${field} is required`, "error");
                return;
            }
        }

        // Validate numeric fields
        const numericFields = ["hemo", "pcv", "rc", "wc", "bu", "sc", "sod", "pot"];
        for (let field of numericFields) {
            if (isNaN(formData[field])) {
                displayResult(`⚠️ Invalid input: ${field} must be a number`, "error");
                return;
            }
        }

        try {
            const response = await fetch("/annpredict", {
                method: "POST",
                headers: { 
                    "Accept": "application/json",
                    "Content-Type": "application/json"  // ✅ Ensures JSON format
                },
                body: JSON.stringify(formData)  // ✅ Convert data to JSON
            });

            console.log("🔹 Raw Response:", response);  // Debugging log

            const result = await response.json();
            console.log("✅ Server Response:", result);

            if (response.ok) {
                localStorage.setItem("formData", JSON.stringify(formData));
                localStorage.setItem("prediction", result.prediction);
                displayResult(`✅ Prediction: ${result.prediction}`, "success");
            } else {
                displayResult(`❌ Error: ${result.error}`, "error");
            }
        } catch (error) {
            console.error("❌ Server Error:", error);
            displayResult("⚠️ Server error. Please try again later.", "error");
        }
    }

    function collectFormData() {
        let formData = {};
        const inputs = predictionForm.querySelectorAll("input, select");
        inputs.forEach(input => {
            formData[input.id] = input.value.trim();
        });
        return formData;
    }

    function loadFormData() {
        const storedData = JSON.parse(localStorage.getItem("formData"));
        if (storedData) {
            Object.keys(storedData).forEach(key => {
                if (document.getElementById(key)) {
                    document.getElementById(key).value = storedData[key];
                }
            });
        }
    }

    function displayResult(message, type) {
        predictionResult.innerHTML = `<h3 class="${type}">${message}</h3>`;
    }
});
