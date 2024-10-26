import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from dotenv import load_dotenv
import google.generativeai as gen_ai
import os

# Load environment variables
load_dotenv()

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
    # Replace this with actual model prediction logic
    return 50  # Example predicted rainfall in mm

def create_gemini_prompt(lat,lon, crop_type, soil_type, area, temp , Wind_speed , humidity, rain_today,month ,day):
    return (
        f"You are an agricultural advice bot here to help farmers optimize their practices. "
        f"Please provide tailored advice based on the following parameters:\n\n"
        f"Location: {lat}{lon}\n"  # e.g., "Latitude: 20.5937, Longitude: 78.9629"
        f"Crop Type: {crop_type}\n"  # e.g., "Rice"
        f"Soil Type: {soil_type}\n"  # e.g., "Loamy"
        f"Area: {area} sq cm\n\n"
        f"Based on these details, what recommendations can you provide for effective "
        f"crop management, irrigation strategies, water management and any other relevant agricultural practices?"
    )

def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role
    
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
    "Jakarta": {"latitude": -6.2088, "longitude": 106.8456}
}

Prediction , Helping_Bot = st.tabs(["Rainfall Prediction", "Helping_Bot"])
def main() :# Streamlit App Title
    with Prediction :
        st.title("Weather and Rainfall Dashboard")

        # Sidebar for user inputs
        nav = st.sidebar.radio("Navigation", ["Map View", "Rainfall Prediction", "Advice"])

        st.sidebar.header("User Input")
        latitude = st.sidebar.number_input("Enter Latitude", value=20.5937)  # Default to India
        longitude = st.sidebar.number_input("Enter Longitude", value=78.9629)  # Default to India

        soil_type = st.sidebar.selectbox("Select Soil Type", ["Clay", "Sandy", "Loamy", "Silty"])
        crop_type = st.sidebar.selectbox("Select Crop Type", ["Wheat", "Rice", "Maize", "Soybean"])
        area = st.sidebar.number_input("Enter Area (sq cm)", value=100.0)
                                    

        # Conditional display based on navigation selection
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
            
            # Plotly Scatter Map
            fig = px.scatter_geo(
                map_df,
                lat="Latitude",
                lon="Longitude",
                color="Temperature (°C)",
                size="Humidity (%)",
                hover_name="City",
                size_max=15,
                color_continuous_scale=px.colors.sequential.Plasma,
                title="Current Weather Data Across the World"
            )
            
            # Show the Plotly map in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        elif nav == "Rainfall Prediction":
            st.title("Rainfall Prediction")
            # Fetch Data on Button Click
        if st.sidebar.button("Get Weather Data"):
            weather_data = fetch_weather_data(latitude, longitude)
            if weather_data:
                # Extracting current weather data
                current_weather = weather_data['current_weather']
                current_temperature = current_weather['temperature']
                current_wind_speed = current_weather['windspeed']
                
                # Extracting hourly weather data
                hourly_data = weather_data['hourly']
                hourly_temperature = hourly_data['temperature_2m']
                hourly_humidity = hourly_data['relative_humidity_2m']
                timestamps = hourly_data['time']
                
                # Simulated rainfall prediction based on user input
                rainfall_prediction = fetch_rainfall_prediction(latitude, longitude)
                
                # Display the current weather data
                st.subheader("Current Weather Data")
                st.write(f"Temperature: {current_temperature} °C")
                st.write(f"Wind Speed: {current_wind_speed} m/s")
                
                # Creating DataFrame for hourly data
                hourly_df = pd.DataFrame({
                    'Timestamp': timestamps,
                    'Temperature (°C)': hourly_temperature,
                    'Relative Humidity (%)': hourly_humidity
                })
                
                # Display the hourly weather data
                st.subheader("Hourly Weather Data")
                st.dataframe(hourly_df)
                
                # Prepare data for the world map
                world_data = {
                    "Country": ["India", "Brazil", "China", "United States", "Russia", "Australia", "Nigeria", "Germany", "France"],
                    "Rainfall Prediction (mm)": [rainfall_prediction, 60, 75, 50, 65, 85, 90, 45, 70],  # Placeholder values
                    "Temperature (°C)": [current_temperature, 25, 30, 20, 15, 35, 28, 18, 22],  # Example temperatures
                    "Humidity (%)": [70, 60, 80, 50, 40, 30, 90, 85, 75]  # Example humidity
                }
                
                # Create DataFrame for map
                map_df = pd.DataFrame(world_data)
                
                # Plotly Scatter Map
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
                
                # Show the Plotly map in Streamlit
                st.plotly_chart(fig, use_container_width=True)
                
                # Show average rainfall prediction as a summary
                st.subheader("Summary")
                st.write(f"Predicted Rainfall: {rainfall_prediction} mm")

        elif nav == "Advice":
            st.title("Advice")
    with Helping_Bot:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        # Set up Google Gemini-Pro AI model
        gen_ai.configure(api_key=GOOGLE_API_KEY)
        model = gen_ai.GenerativeModel('gemini-1.5-pro')

        '''lat = 20.5937
        lon: 78.9629
        crop_type = "Rice"  # Example crop
        soil_type = "Loamy"  # Example soil type
        area = 1000  # Example area in square cm

        prompt = create_gemini_prompt(lat,lon, crop_type, soil_type, area, temp , Wind_speed , humidity, rain_today,month ,day)
'''
        #chat  = model.start_chat(history=[])
        #chat.send_message(prompt)
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