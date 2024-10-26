import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from dotenv import load_dotenv
import google.generativeai as gen_ai
import os
import pickle

# Load environment variables
load_dotenv()

model = pickle.load(open("./models/xgb1.pkl", "rb"))
print("Model Loaded")



# Function to fetch temperature and humidity data from the Open-Meteo API
def fetch_weather_data(latitude, longitude):
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        st.error("Failed to fetch data from Open-Meteo API")
        return None

# Function to simulate fetching rainfall prediction from your model
def fetch_rainfall_prediction(latitude, longitude):
    # Placeholder for model output
    return 50  # Example predicted rainfall in mm

def create_gemini_prompt(lat, lon, crop_type, soil_type, area):
    return (
        f"You are an agricultural advice bot here to help farmers optimize their practices. "
        f"Please provide tailored advice based on the following parameters:\n\n"
        f"Location: {lat}, {lon}\n"
        f"Crop Type: {crop_type}\n"
        f"Soil Type: {soil_type}\n"
        f"Area: {area} sq cm\n\n"
        f"Based on these details, what recommendations can you provide for effective "
        f"crop management, irrigation strategies, and any other relevant agricultural practices?"
    )

def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role
    
locations = {
    "Paris": {"latitude": 48.8566, "longitude": 2.3522},
    "Beijing": {"latitude": 39.9042, "longitude": 116.4074},
    "Cairo": {"latitude": 30.0444, "longitude": 31.2357},
    "Buenos Aires": {"latitude": -34.6037, "longitude": -58.3816},
    "Johannesburg": {"latitude": -26.2041, "longitude": 28.0473},
    "Moscow": {"latitude": 55.7558, "longitude": 37.6173},
    "Bangkok": {"latitude": 13.7563, "longitude": 100.5018},
    "Berlin": {"latitude": 52.5200, "longitude": 13.4050},
    "Dubai": {"latitude": 25.276987, "longitude": 55.296249},
    "Seoul": {"latitude": 37.5665, "longitude": 126.9780},
    "Los Angeles": {"latitude": 34.0522, "longitude": -118.2437},
    "Rome": {"latitude": 41.9028, "longitude": 12.4964},
    "Madrid": {"latitude": 40.4168, "longitude": -3.7038},
    "Singapore": {"latitude": 1.3521, "longitude": 103.8198},
    "Istanbul": {"latitude": 41.0082, "longitude": 28.9784},
    "Toronto": {"latitude": 43.6510, "longitude": -79.3470},
    "Mexico City": {"latitude": 19.4326, "longitude": -99.1332},
    "São Paulo": {"latitude": -23.5505, "longitude": -46.6333},
    "Lagos": {"latitude": 6.5244, "longitude": 3.3792},
    "Jakarta": {"latitude": -6.2088, "longitude": 106.8456},
    "Tokyo": {"latitude": 35.6762, "longitude": 139.6503},
    "Sydney": {"latitude": -33.8688, "longitude": 151.2093},
    "New Delhi": {"latitude": 28.6139, "longitude": 77.2090},
    "London": {"latitude": 51.5074, "longitude": -0.1278},
    "New York": {"latitude": 40.7128, "longitude": -74.0060},
    "Hong Kong": {"latitude": 22.3193, "longitude": 114.1694},
    "Melbourne": {"latitude": -37.8136, "longitude": 144.9631},
    "Rio de Janeiro": {"latitude": -22.9068, "longitude": -43.1729},
    "Kuala Lumpur": {"latitude": 3.1390, "longitude": 101.6869},
    "Manila": {"latitude": 14.5995, "longitude": 120.9842},
    "Lima": {"latitude": -12.0464, "longitude": -77.0428},
    "Vienna": {"latitude": 48.2082, "longitude": 16.3738},
    "Nairobi": {"latitude": -1.2921, "longitude": 36.8219},
    "Santiago": {"latitude": -33.4489, "longitude": -70.6693},
    "Riyadh": {"latitude": 24.7136, "longitude": 46.6753},
    "Karachi": {"latitude": 24.8607, "longitude": 67.0011},
    "Tehran": {"latitude": 35.6892, "longitude": 51.3890},
    "Baghdad": {"latitude": 33.3152, "longitude": 44.3661},
    "Athens": {"latitude": 37.9838, "longitude": 23.7275},
    "Havana": {"latitude": 23.1136, "longitude": -82.3666},
    "Nagpur": {"latitude": 21.1458, "longitude": 79.0882},
    "Mumbai": {"latitude": 19.0760, "longitude": 72.8777},
    "Bengaluru": {"latitude": 12.9716, "longitude": 77.5946},
    "Chennai": {"latitude": 13.0827, "longitude": 80.2707},
    "Hyderabad": {"latitude": 17.3850, "longitude": 78.4867}
}

def predict_rainfall(temp, wind_speed, hourly_humidity, rain_today, month, day):
    avg_humidity = sum(hourly_humidity) / len(hourly_humidity) if hourly_humidity else 0
    input_lst = [temp, wind_speed, avg_humidity, rain_today, month, day]
    prediction = model.predict([input_lst])[0]
    
    return "The prediction is sunny." if prediction == 0 else "The prediction is rainy."

Prediction, Helping_Bot = st.tabs(["Rainfall Prediction", "Helping_Bot"])

def main():
    with Prediction:
        st.title("Weather and Rainfall Dashboard")
        nav = st.sidebar.radio("Navigation", ["Map View", "Rainfall Prediction", "Advice"])

        st.sidebar.header("User Input")
        latitude = st.sidebar.number_input("Enter Latitude", value=20.5937)
        longitude = st.sidebar.number_input("Enter Longitude", value=78.9629)
        soil_type = st.sidebar.selectbox("Select Soil Type", ["Clay", "Sandy", "Loamy", "Silty"])
        crop_type = st.sidebar.selectbox("Select Crop Type", ["Wheat", "Rice", "Maize", "Soybean"])
        area = st.sidebar.number_input("Enter Area (sq cm)", value=100.0)
        
        
        current_temperature = None
        hourly_humidity = None

        if nav == "Map View":
            st.title("Map View")
            
            world_data = {
                "City": [],
                "Latitude": [],
                "Longitude": [],
                "Temperature (°C)": [],
                "Humidity (%)": [],
                "Wind Speed (m/s)": []
            }
            
            for city, coords in locations.items():
                data = fetch_weather_data(coords["latitude"], coords["longitude"])
                if data:
                    current_weather = data['current_weather']
                    world_data["City"].append(city)
                    world_data["Latitude"].append(coords["latitude"])
                    world_data["Longitude"].append(coords["longitude"])
                    world_data["Temperature (°C)"].append(current_weather['temperature'])
                    world_data["Humidity (%)"].append(data['hourly']['relative_humidity_2m'][0])  # Get the first hourly humidity
                    world_data["Wind Speed (m/s)"].append(current_weather['windspeed'])
            
            map_df = pd.DataFrame(world_data)
            
            # Plotly Scatter Map with unique design
            fig = px.scatter_geo(
            map_df,
            lat="Latitude",
            lon="Longitude",
            color="Temperature (°C)",  # Color by temperature
            size="Humidity (%)",  # Bubble size by humidity
            hover_name="City",  # Hover info to show city names
            projection="natural earth",  # Set map projection to 'natural earth'
            size_max=15,
            color_continuous_scale=px.colors.sequential.Viridis,  # Use a unique color scale like 'Viridis'
            title="Unique Weather Map with Temperature and Humidity",
            template="plotly_dark"  # Use a dark theme for a distinct visual appeal
            )

        # Customizing the layout of the map
            fig.update_geos(
                showcoastlines=True, coastlinecolor="Blue",  # Customize the coastline color
                showland=True, landcolor="rgb(230, 145, 56)",  # Custom land color
                showocean=True, oceancolor="LightBlue",  # Custom ocean color
                showlakes=True, lakecolor="Blue",  # Add lake color
                projection_scale=1.2  # Zoom in
            )

            # Customizing map layout further
            fig.update_layout(
                title="Weather and Rainfall Predictions with Unique Map Design",
                font=dict(family="Arial", size=18, color="white"),
                geo=dict(bgcolor="rgba(0,0,0,0)",  # Transparent background for a sleek look
                        showframe=False  # Hide map frame
                        )
            )

            # Show the Plotly map in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        elif nav == "Rainfall Prediction":
            st.title("Rainfall Prediction")
            rain_today = st.radio("Did it rain today?", (0, 1), index=0)
            month = st.selectbox("Select month:", list(range(1, 13)))
            day = st.selectbox("Select day:", list(range(1, 32)))
            if st.sidebar.button("Get Weather Data"):
                weather_data = fetch_weather_data(latitude, longitude)
                if weather_data:
                    current_weather = weather_data['current_weather']
                    current_temperature = current_weather['temperature']
                    current_wind_speed = current_weather['windspeed']
                    hourly_data = weather_data['hourly']
                    hourly_humidity = hourly_data['relative_humidity_2m']
                    timestamps = hourly_data['time']
                    
                    result = predict_rainfall(current_temperature, current_wind_speed, hourly_humidity, rain_today, month, day)

                    st.subheader("Current Weather Data")
                    st.write(f"Temperature: {current_temperature} °C")
                    st.write(f"Wind Speed: {current_wind_speed} m/s")

                    hourly_df = pd.DataFrame({
                        'Timestamp': timestamps,
                        'Temperature (°C)': hourly_data['temperature_2m'],
                        'Relative Humidity (%)': hourly_humidity
                    })

                    st.subheader("Hourly Weather Data")
                    st.dataframe(hourly_df)

                    world_data = {
                        "Country": ["India", "Brazil", "China", "United States", "Russia", "Australia", "Nigeria", "Germany", "France"],
                        "Rainfall Prediction (mm)": [result, 60, 75, 50, 65, 85, 90, 45, 70],
                        "Temperature (°C)": [current_temperature, 25, 30, 20, 15, 35, 28, 18, 22],
                        "Humidity (%)": [70, 60, 80, 50, 40, 30, 90, 85, 75]
                    }

                    map_df = pd.DataFrame(world_data)
                    map_df['Rainfall Prediction (mm)'] = pd.to_numeric(map_df['Rainfall Prediction (mm)'], errors='coerce')
                    map_df['Temperature (°C)'] = pd.to_numeric(map_df['Temperature (°C)'], errors='coerce')
                    map_df = map_df.dropna(subset=['Rainfall Prediction (mm)', 'Temperature (°C)'])
                    if not map_df.empty:  # Only create a plot if there's data
                        fig = px.scatter_geo(
                            map_df,
                            locations="Country",
                            locationmode="country names",
                            size="Rainfall Prediction (mm)",
                            color="Temperature (°C)",
                            hover_name="Country",
                            size_max=50,
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title="Weather and Rainfall Predictions Across the World"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Summary")
                    st.write(f"Predicted Rainfall: {result} mm")

        elif nav == "Advice":
            st.title("Advice")
            
    with Helping_Bot:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        gen_ai.configure(api_key=GOOGLE_API_KEY)
        model = gen_ai.GenerativeModel('gemini-1.5-pro')

       
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = model.start_chat(history=[])
        
        for message in st.session_state.chat_session.history:
            with st.chat_message(translate_role_for_streamlit(message.role)):
                st.markdown(message.parts[0].text)

                # Input field for user's message
        user_prompt = st.chat_input("Ask Gemini-Pro...")
        if user_prompt:
                    # Add user's message to chat and display it
                    st.chat_message("user").markdown(user_prompt)

                    # Send user's message to Gemini-Pro and get the response
                    gemini_response = st.session_state.chat_session.send_message(user_prompt)

                    # Display Gemini-Pro's response
                    with st.chat_message("assistant"):
                        st.markdown(gemini_response.text)
if __name__ == "__main__":
    main()