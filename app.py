import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from dotenv import load_dotenv
import google.generativeai as gen_ai
import os
import pickle
import leafmap.foliumap as leafmap

# Load environment variables
load_dotenv()

model1 = pickle.load(open("./models/xgb1.pkl", "rb"))
print("Model Loaded")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key=GOOGLE_API_KEY)


def create_gemini_prompt(latitude, longitude, crop_type, soil_type, area, current_temperature, humidity):
    return (
        f"You are an agricultural advice bot here to help farmers optimize their practices. "
        f"Please provide tailored advice based on the following parameters:\n\n"
        f"Location: Latitude: {latitude}, Longitude: {longitude}\n"
        f"Crop Type: {crop_type}\n"
        f"Soil Type: {soil_type}\n"
        f"Area: {area} sq cm\n"
        f"Temperature: {current_temperature}°C\n"
        #f"Wind Speed: {wind_speed} km/h\n"
        f"Humidity: {humidity}%\n"
        #f"Rain Today: {'Yes' if rain_today else 'No'}\n"
       # f"Date: Month {month}, Day {day}\n\n"
        f"Based on these details, what recommendations can you provide for effective "
        f"crop management, irrigation strategies, water management, and any other relevant agricultural practices?"
    )

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


def chat_bot(user_prompt):
    if user_prompt:
                    # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

                    # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(user_prompt)

                    # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

def predict_rainfall(temp, wind_speed, hourly_humidity, rain_today, month, day):
    avg_humidity = sum(hourly_humidity) / len(hourly_humidity) if hourly_humidity else 0
    input_lst = [temp, wind_speed, avg_humidity, rain_today, month, day]
    prediction = model1.predict([input_lst])[0]
    
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
            
            col1, col2 = st.columns([4, 1])
            options = list(leafmap.basemaps.keys())
            index = options.index("OpenTopoMap")
            with col2:
                basemap = st.selectbox("Select a basemap:", options, index)
            with col1:
                m = leafmap.Map(
                locate_control=True, latlon_control=True, draw_export=True, minimap_control=True
            )
            m.add_basemap(basemap)
            m.to_streamlit(height=500,width=900)

        elif nav == "Rainfall Prediction":
            st.title("Rainfall Prediction")
            rain_today = st.radio("Did it rain today? ('0' = NO , '1' = YES)", (0, 1), index=0)
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
                    st.subheader("Predicted Rainfall :-")
                    st.write(f"Predicted Rainfall for {latitude} latitude , {longitude} longitude : {result} ")
                    st.subheader("Hourly Weather Data")
                    st.dataframe(hourly_df)

                    world_data = {
                        "latitude": [latitude],
                        "longitude": [longitude],
                        "Temperature (°C)": [current_temperature],
                        "Humidity (%)": [hourly_humidity]
                    }

                    map_df = pd.DataFrame(world_data)
                    map_df['Temperature (°C)'] = pd.to_numeric(map_df['Temperature (°C)'], errors='coerce')
                    map_df['Humidity (%)'] = pd.to_numeric(map_df['Humidity (%)'], errors='coerce')

                    # Dropdown to allow the user to select the variable for the color scale
                    color_option = st.selectbox(
                        'Select a parameter to color the map points:',
                        ('Temperature (°C)', 'Humidity (%)')
                    )


                    if not map_df.empty:
                        center_lat = map_df['latitude'].mean()
                        center_lon = map_df['longitude'].mean()

                        # Check if the DataFrame has valid data to plot
                        if 'latitude' in map_df.columns and 'longitude' in map_df.columns:
                            # Create the scatter_geo plot
                            fig = px.scatter_geo(
                                map_df,
                                lat="latitude",
                                lon="longitude",
                                #size=15,  # Dynamically based on user selection
                                color=color_option,  # Dynamically based on user selection
                                size_max=15,
                                color_continuous_scale=px.colors.sequential.Plasma,  # Color scale
                                title="Weather and Rainfall Predictions Across the World"
                            )

                            # Add automatic zoom and centering
                            fig.update_geos(
                                center={'lat': center_lat, 'lon': center_lon},  # Center the map based on data
                                projection_scale=4.5,  # Adjust this value for zoom level (lower = more zoomed out)
                                showland=True,  # Show land on the map
                            )

                            # Display the plot using Streamlit
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("No valid data to plot.")

                                            

        elif nav == "Advice":
            st.title("Advice")
            model = gen_ai.GenerativeModel('gemini-1.5-pro')
            chat = model.start_chat(history=[])
            prompt_text = create_gemini_prompt(latitude, longitude, crop_type, soil_type, area, current_temperature, hourly_humidity)

            # Send the generated prompt
            response = chat.send_message(prompt_text)
            # show the output
            st.markdown(response.text)

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
        user_prompt = st.chat_input("Ask ")
        if user_prompt:
            chat_bot(user_prompt)
        
if __name__ == "__main__":
    main()

