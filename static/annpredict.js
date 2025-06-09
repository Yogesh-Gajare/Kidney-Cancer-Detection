



window.onload = function () {
    // Retrieve the form data and prediction result from localStorage
    const formData = JSON.parse(localStorage.getItem("formData"));
    const prediction = localStorage.getItem("prediction");

    // Debugging logs
    console.log("formData from LocalStorage:", formData);
    console.log("Prediction from LocalStorage:", prediction);

    if (formData && prediction) {
        document.getElementById("ageDisplay").textContent = formData.age;
        document.getElementById("bloodPressureDisplay").textContent = formData.blood_pressure;
        document.getElementById("albuminDisplay").textContent = formData.albumin;
        document.getElementById("sugarDisplay").textContent = formData.sugar;
        document.getElementById("redBloodCellsDisplay").textContent = formData.red_blood_cells;
        document.getElementById("pusCellDisplay").textContent = formData.pus_cell;
        document.getElementById("bacteriaDisplay").textContent = formData.bacteria;
        document.getElementById("bloodGlucoseDisplay").textContent = formData.blood_glucose;
        document.getElementById("serumCreatinineDisplay").textContent = formData.serum_creatinine;
        document.getElementById("potassiumDisplay").textContent = formData.potassium;
        document.getElementById("anaemiaDisplay").textContent = formData.anaemia;

        document.getElementById("predictionText").textContent = 
            prediction === "Abnormal" ? "Abnormal (Possible Condition Detected)" : "Normal (No Condition Detected)";
    } else {
        console.error("No form data or prediction found in localStorage.");
        alert("No data found. Please go back and submit the form.");
        window.location.href = "/open";
    }
};
