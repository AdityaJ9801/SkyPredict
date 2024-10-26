def create_gemini_prompt(lat, lon, crop_type, soil_type, area, temp, wind_speed, humidity, rain_today, month, day):
    return (
        f"You are an agricultural advice bot here to help farmers optimize their practices. "
        f"Please provide tailored advice based on the following parameters:\n\n"
        f"Location: Latitude: {lat}, Longitude: {lon}\n"
        f"Crop Type: {crop_type}\n"
        f"Soil Type: {soil_type}\n"
        f"Area: {area} sq cm\n"
        f"Temperature: {temp}°C\n"
        f"Wind Speed: {wind_speed} km/h\n"
        f"Humidity: {humidity}%\n"
        f"Rain Today: {'Yes' if rain_today else 'No'}\n"
        f"Date: Month {month}, Day {day}\n\n"
        f"Based on these details, what recommendations can you provide for effective "
        f"crop management, irrigation strategies, water management, and any other relevant agricultural practices?"
    )

# Initialize model
model = gen_ai.GenerativeModel('gemini-1.5-pro')

# Start a chat session
chat = model.start_chat(history=[])

# Define parameters for the prompt
lat, lon = 20.5937, 78.9629  # Example latitude and longitude
crop_type = "Rice"
soil_type = "Loamy"
area = 5000  # Example area in sq cm
temp = 30  # Temperature in Celsius
wind_speed = 15  # Wind speed in km/h
humidity = 70  # Humidity in percentage
rain_today = True  # Rain today (True or False)
month, day = 3, 4  # Month and day

# Create the prompt
prompt_text = create_gemini_prompt(lat, lon, crop_type, soil_type, area, temp, wind_speed, humidity, rain_today, month, day)

# Send the generated prompt
response = chat.send_message(prompt_text)
# show the output
st.markdown(response.text)
