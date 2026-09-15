
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(layout='wide')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Load the trained model
model = joblib.load('logistic_regression_model.sav')

# Define the input features based on the x.columns output
# from earlier in the notebook: ['Delivery_Distance', 'Traffic_Congestion', 
# 'Weather_Condition', 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 
# 'Vehicle_Age', 'Road_Condition_Score', 'Package_Weight', 
# 'Fuel_Efficiency', 'Warehouse_Processing_Time']

with st.form('prediction_form'):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delivery_distance = st.number_input('Delivery Distance (km)', min_value=0.0, max_value=100.0, value=20.0, step=0.1)
        traffic_congestion = st.slider('Traffic Congestion (1=Low, 5=High)', min_value=1, max_value=5, value=3)
        weather_condition = st.slider('Weather Condition (1=Good, 5=Bad)', min_value=1, max_value=5, value=1)
        delivery_slot = st.slider('Delivery Slot (1=Morning, 2=Afternoon, 3=Evening)', min_value=1, max_value=3, value=2)
    
    with col2:
        driver_experience = st.number_input('Driver Experience (years)', min_value=0, max_value=30, value=5)
        num_stops = st.number_input('Number of Stops', min_value=0, max_value=20, value=2)
        vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, max_value=15, value=3)
        road_condition_score = st.number_input('Road Condition Score (0-5000)', min_value=0, max_value=5000, value=3120)
    
    with col3:
        package_weight = st.number_input('Package Weight (kg)', min_value=0.0, max_value=50.0, value=12.0, step=0.1)
        fuel_efficiency = st.number_input('Fuel Efficiency (km/l)', min_value=0.0, max_value=200.0, value=120.0, step=0.1)
        warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, max_value=300, value=120)
    
    submit_button = st.form_submit_button('Predict Delivery Delay')

if submit_button:
    # Create a DataFrame from the input data
    input_data = pd.DataFrame([{
        'Delivery_Distance': delivery_distance,
        'Traffic_Congestion': traffic_congestion,
        'Weather_Condition': weather_condition,
        'Delivery_Slot': delivery_slot,
        'Driver_Experience': driver_experience,
        'Num_Stops': num_stops,
        'Vehicle_Age': vehicle_age,
        'Road_Condition_Score': road_condition_score,
        'Package_Weight': package_weight,
        'Fuel_Efficiency': fuel_efficiency,
        'Warehouse_Processing_Time': warehouse_processing_time
    }])

    # Ensure the columns are in the same order as during training
    expected_columns = [
        'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
        'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
        'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
        'Warehouse_Processing_Time'
    ]
    input_data = input_data[expected_columns]

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader('Prediction Results:')
    if prediction == 1:
        st.error('**Prediction: Delivery Delay is LIKELY**')
    else:
        st.success('**Prediction: No Delivery Delay Expected**')
    
    st.write(f"Probability of No Delay: {prediction_proba[0]*100:.2f}%")
    st.write(f"Probability of Delay: {prediction_proba[1]*100:.2f}%")
    st.write('\n---\n')
    st.info('You can run this Streamlit app by executing `!streamlit run streamlit_app.py` in a new cell, then clicking the public URL that appears.')

