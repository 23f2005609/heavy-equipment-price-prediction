const form = document.getElementById("predictionForm");

const predictButton =
    document.getElementById("predictButton");

const buttonText =
    document.getElementById("buttonText");

const spinner =
    document.getElementById("spinner");

const resetButton =
    document.getElementById("resetButton");

const loading =
    document.getElementById("loading");

const result =
    document.getElementById("result");

const predictedPrice =
    document.getElementById("predictedPrice");

const errorBox =
    document.getElementById("error");


/* ==================================================
   FORM SUBMISSION
================================================== */

form.addEventListener("submit", async function (event) {

    event.preventDefault();


    /* ----------------------------------------------
       Clear previous state
    ---------------------------------------------- */

    result.classList.add("hidden");

    errorBox.classList.add("hidden");


    /* ----------------------------------------------
       Get manufacture year
    ---------------------------------------------- */

    const manufactureYear =
        document.getElementById(
            "ManufactureYear"
        ).value;


    if (!manufactureYear) {

        showError(
            "Please enter the manufacture year."
        );

        return;
    }


    const year = Number(manufactureYear);


    if (year < 1950 || year > 2026) {

        showError(
            "Please enter a valid manufacture year between 1950 and 2026."
        );

        return;
    }


    /* ----------------------------------------------
       Operational hours
    ---------------------------------------------- */

    const hoursValue =
        document.getElementById(
            "OperationalHoursMeter"
        ).value;


    let operationalHours = null;


    if (hoursValue !== "") {

        operationalHours = Number(hoursValue);


        if (
            Number.isNaN(operationalHours) ||
            operationalHours < 0
        ) {

            showError(
                "Operational hours must be a non-negative number."
            );

            return;
        }
    }


    /* ----------------------------------------------
       Build API request
    ---------------------------------------------- */

    const data = {

        ManufactureYear: year,

        OperationalHoursMeter:
            operationalHours,

        UtilizationTier:
            getValue("UtilizationTier"),

        AssetScaleFactor:
            getValue("AssetScaleFactor"),

        FunctionalClassification:
            getValue("FunctionalClassification"),

        RegionCode:
            getValue("RegionCode"),

        CabinType:
            getValue("CabinType"),

        Forks:
            getValue("Forks"),

        DrivetrainType:
            getValue("DrivetrainType"),

        Spec_FullDescriptor:
            getValue("Spec_FullDescriptor")
    };


    /* ----------------------------------------------
       Loading state
    ---------------------------------------------- */

    setLoading(true);


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        const responseData =
            await response.json();


        if (!response.ok) {

            throw new Error(
                responseData.detail ||
                "Prediction failed."
            );
        }


        /* ------------------------------------------
           Validate response
        ------------------------------------------ */

        if (
            responseData.predicted_price === undefined ||
            responseData.predicted_price === null
        ) {

            throw new Error(
                "The server returned an invalid prediction."
            );
        }


        /* ------------------------------------------
           Format price
        ------------------------------------------ */

        const price =
            Number(
                responseData.predicted_price
            );


        predictedPrice.textContent =
            "₹" +
            price.toLocaleString(
                "en-IN",
                {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                }
            );


        /* ------------------------------------------
           Show result
        ------------------------------------------ */

        result.classList.remove("hidden");

        result.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });

    }

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        showError(
            error.message ||
            "Unable to generate prediction."
        );

    }

    finally {

        setLoading(false);

    }

});


/* ==================================================
   GET INPUT VALUE
================================================== */

function getValue(id) {

    const element =
        document.getElementById(id);

    if (!element) {
        return null;
    }

    const value =
        element.value.trim();

    return value === ""
        ? null
        : value;
}


/* ==================================================
   LOADING STATE
================================================== */

function setLoading(isLoading) {

    predictButton.disabled =
        isLoading;

    loading.classList.toggle(
        "hidden",
        !isLoading
    );

    spinner.classList.toggle(
        "hidden",
        !isLoading
    );


    if (isLoading) {

        buttonText.textContent =
            "Predicting...";

    } else {

        buttonText.textContent =
            "Predict Selling Price";

    }
}


/* ==================================================
   ERROR MESSAGE
================================================== */

function showError(message) {

    errorBox.textContent =
        message;

    errorBox.classList.remove(
        "hidden"
    );

    errorBox.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}


/* ==================================================
   RESET
================================================== */

resetButton.addEventListener(
    "click",
    function () {

        form.reset();

        result.classList.add(
            "hidden"
        );

        errorBox.classList.add(
            "hidden"
        );

        document
            .getElementById(
                "ManufactureYear"
            )
            .focus();

    }
);

const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const themeText = document.getElementById("themeText");

function setTheme(theme) {
    if (theme === "dark") {
        document.body.classList.add("dark-mode");

        themeIcon.textContent = "☀";
        themeText.textContent = "Light mode";
    } else {
        document.body.classList.remove("dark-mode");

        themeIcon.textContent = "🌙";
        themeText.textContent = "Dark mode";
    }

    localStorage.setItem("theme", theme);
}

const savedTheme = localStorage.getItem("theme") || "light";

setTheme(savedTheme);

themeToggle.addEventListener("click", () => {
    const isDark =
        document.body.classList.contains("dark-mode");

    setTheme(isDark ? "light" : "dark");
});