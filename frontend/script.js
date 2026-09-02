const form = document.getElementById("predictionForm");

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    const result = document.getElementById("result");
    const error = document.getElementById("error");
    const price = document.getElementById("price");

    result.classList.add("hidden");
    error.classList.add("hidden");

    const machine = {
        AssetID: Number(document.getElementById("AssetID").value),
        ProductConfigID: Number(document.getElementById("ProductConfigID").value),
        ManufactureYear: Number(document.getElementById("ManufactureYear").value),

        OperationalHoursMeter:
            document.getElementById("OperationalHoursMeter").value === ""
                ? null
                : Number(document.getElementById("OperationalHoursMeter").value),

        UtilizationTier: document.getElementById("UtilizationTier").value,
        RegionCode: document.getElementById("RegionCode").value,
        VendorPartnerID: document.getElementById("VendorPartnerID").value,

        Spec_FullDescriptor:
            document.getElementById("Spec_FullDescriptor").value,

        Spec_BaseClass:
            document.getElementById("Spec_BaseClass").value,

        Spec_SubClass:
            document.getElementById("Spec_SubClass").value,

        CabinType:
            document.getElementById("CabinType").value,

        DrivetrainType:
            document.getElementById("DrivetrainType").value
    };

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(machine)
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        price.textContent =
            "₹" + Number(data.predicted_price).toLocaleString("en-IN");

        result.classList.remove("hidden");

    } catch (err) {

        error.textContent = err.message;
        error.classList.remove("hidden");

    }

});