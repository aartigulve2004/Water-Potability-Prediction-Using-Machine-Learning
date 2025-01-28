import streamlit as st
import pickle
import pandas as pd

# Load the trained model and label encoder
with open('water_quality_model.pkl', 'rb') as file:
    model, label_encoder = pickle.load(file)

# Load the dataset to extract valid countries
cities_data = pd.read_csv('Cities1.csv')
valid_countries = cities_data['Country'].unique()
valid_countries_sorted = sorted(valid_countries)  # Sort for a cleaner dropdown

# List of cities
cities = [
    "United States of America", "Germany", "United States of America", "Switzerland", 
    "Switzerland", "Switzerland", "United Kingdom", "Egypt", "United States of America", 
    "France", "United States of America", "Canada", "Brazil", "Lithuania", "Monaco", 
    "Belgium", "Poland", "Uzbekistan", "Italy", "France", "Singapore", "Canada", 
    "United Kingdom", "Germany", "North Macedonia", "Poland", "Slovenia", "Bulgaria", 
    "Italy", "Poland", "Norway", "Germany", "Portugal", "United Arab Emirates", "Italy", 
    "Russia", "Germany", "Poland", "Russia", "Russia", "Russia", "People's Republic of China", 
    "Georgia", "Slovenia", "India", "Czech Republic", "Israel", "Uruguay", "Bangladesh", 
    "Pakistan", "Croatia", "Philippines", "Argentina", "Mexico", "Japan", "Bolivia", 
    "Spain", "Cote d'Ivoire", "Venezuela", "Guatemala", "Cuba", "Austria", "Sweden", 
    "Finland", "Russia", "Iceland", "Estonia", "Latvia", "Slovakia", "Hungary", "Poland", 
    "Luxembourg", "Liechtenstein", "San Marino", "Cambodia", "Vietnam", "Thailand", 
    "Andorra", "Malaysia", "Taiwan", "Ukraine", "Chile", "South Sudan", "Armenia", "Tanzania", 
    "Sudan"
]

# Streamlit App Configuration
st.set_page_config(page_title="Water Quality Prediction", page_icon="💧", layout="centered")

# Page Title
st.title("💧 Water Quality Prediction App")
st.write("Easily predict water pollution levels using air quality and PM2.5 metrics!")

# Sidebar for User Inputs
st.sidebar.header("User Inputs")
st.sidebar.write("Provide the necessary inputs below to get predictions.")

# User Inputs
country = st.sidebar.selectbox("🌍 Select Country:", options=valid_countries_sorted)
air_quality = st.sidebar.slider("🌫️ Air Quality Index (0-200):", min_value=0.0, max_value=200.0, value=50.0)
pm25 = st.sidebar.slider("💨 Average PM2.5 Level (0-100):", min_value=0.0, max_value=100.0, value=10.0)

# Prediction Section: Button placed at the top
if st.button("🔮 Predict Water Pollution"):
    try:
        # Encode the country input
        country_encoded = label_encoder.transform([country])[0]
        
        # Prepare the input for the model
        input_data = [[country_encoded, air_quality, pm25]]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.success(f"🌊 **Predicted Water Pollution Level**: {prediction:.2f}")
        
        # Display the list of cities in a column below the prediction button
        st.subheader("🌍 Cities Available for Prediction")
        
        # Create two columns to display the list of cities in a neat format
        col1, col2 = st.columns(2)
        
        # Split cities into two halves for better presentation
        half_length = len(cities) // 2
        col1.write("\n".join(cities[:half_length]))  # First half of the cities
        col2.write("\n".join(cities[half_length:]))  # Second half of the cities
        
    except Exception as e:
        st.error("Error: Unable to process input. Please try again.")

# Main Section: Display Inputs (Moved below the Prediction section)
st.header("Your Inputs")
st.write("Ensure your inputs are correct before clicking **Predict**.")
st.markdown(f"""
- **Country**: {country}
- **Air Quality Index**: {air_quality}
- **Average PM2.5 Level**: {pm25}
""")

# Display the List of Countries
st.header("🌍 List of Countries Available")
st.write("Select a country from the table below to predict water pollution levels.")
st.table(pd.DataFrame({"Country": valid_countries_sorted}))
