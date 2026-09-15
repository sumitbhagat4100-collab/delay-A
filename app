import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Delivery Delay Prediction",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Delivery Delay Prediction App")
st.write("Enter the delivery details below to predict whether a delivery delay is likely.")

# ---------------------------------------------------------
# Load trained Logistic Regression model
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "logistic_regression_model.sav"

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please upload/copy "
        "'logistic_regression_model.sav' into the same folder as app.py."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()

# Features used while training the model
EXPECTED_COLUMNS = [
    "Delivery_Distance",
    "Traffic_Congestion",
    "Weather_Condition",
    "Delivery_Slot",
    "Driver_Experience",
    "Num_Stops",
    "Vehicle_Age",
    "Road_Condition_Score",
    "Package_Weight",
    "Fuel_Efficiency",
    "Warehouse_Processing_Time"
]

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
with st.form("prediction_form"):

    st.subheader("Delivery Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        delivery_distance = st.number_input(
            "Delivery Distance (km)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=0.1
        )

        traffic_congestion = st.slider(
            "Traffic Congestion (1 = Low, 5 = High)",
            min_value=1,
            max_value=5,
            value=3
        )

        weather_condition = st.slider(
            "Weather Condition (1 = Good, 5 = Bad)",
            min_value=1,
            max_value=5,
            value=1
        )

        delivery_slot = st.slider(
            "Delivery Slot (1 = Morning, 2 = Afternoon, 3 = Evening)",
            min_value=1,
            max_value=3,
            value=2
        )

    with col2:
        driver_experience = st.number_input(
            "Driver Experience (years)",
            min_value=0,
            max_value=30,
            value=5
        )

        num_stops = st.number_input(
            "Number of Stops",
            min_value=0,
            max_value=20,
            value=2
        )

        vehicle_age = st.number_input(
            "Vehicle Age (years)",
            min_value=0,
            max_value=15,
            value=3
        )

        road_condition_score = st.number_input(
            "Road Condition Score",
            min_value=0,
            max_value=5000,
            value=3120
        )

    with col3:
        package_weight = st.number_input(
            "Package Weight (kg)",
            min_value=0.0,
            max_value=50.0,
            value=12.0,
            step=0.1
        )

        fuel_efficiency = st.number_input(
            "Fuel Efficiency (km/l)",
            min_value=0.0,
            max_value=200.0,
            value=120.0,
            step=0.1
        )

        warehouse_processing_time = st.number_input(
            "Warehouse Processing Time (minutes)",
            min_value=0,
            max_value=300,
            value=120
        )

    submitted = st.form_submit_button(
        "🔮 Predict Delivery Delay",
        use_container_width=True
    )

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:

    input_data = pd.DataFrame([{
        "Delivery_Distance": delivery_distance,
        "Traffic_Congestion": traffic_congestion,
        "Weather_Condition": weather_condition,
        "Delivery_Slot": delivery_slot,
        "Driver_Experience": driver_experience,
        "Num_Stops": num_stops,
        "Vehicle_Age": vehicle_age,
        "Road_Condition_Score": road_condition_score,
        "Package_Weight": package_weight,
        "Fuel_Efficiency": fuel_efficiency,
        "Warehouse_Processing_Time": warehouse_processing_time
    }])

    # Keep exactly the same feature order used during model training
    input_data = input_data[EXPECTED_COLUMNS]

    try:
        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            # Assumes class 0 = No Delay and class 1 = Delay,
            # matching the original model/app setup.
            no_delay_probability = probabilities[0]
            delay_probability = probabilities[1]
        else:
            no_delay_probability = None
            delay_probability = None

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Delivery Delay is LIKELY")
        else:
            st.success("✅ No Delivery Delay Expected")

        if delay_probability is not None:
            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Probability of No Delay",
                    f"{no_delay_probability * 100:.2f}%"
                )

            with c2:
                st.metric(
                    "Probability of Delay",
                    f"{delay_probability * 100:.2f}%"
                )

            st.progress(float(delay_probability))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Powered by a trained Logistic Regression model.")
