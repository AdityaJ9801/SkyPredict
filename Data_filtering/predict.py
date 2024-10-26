import pandas as pd
import pickle

# Load the pre-trained model
model = pickle.load(open("./models/xgb1.pkl", "rb"))
print("Model Loaded")

def predict_rainfall( temp, wind_speed, humidity, rain_today, month, day):
    # Prepare the input list for prediction
    input_lst = [ temp, wind_speed, humidity, rain_today, month, day]
    # Predict using the loaded model
    prediction = model.predict([input_lst])[0]
    
    # Interpret the prediction
    if prediction == 0:
        return "The prediction is sunny."
    else:
        return "The prediction is rainy."

# Gather inputs from the user
try:
    # location = float(input("Enter location code (numeric): "))
    temp = float(input("Enter temperature: "))
    wind_speed = float(input("Enter wind speed: "))
    humidity = float(input("Enter humidity: "))
    rain_today = float(input("Enter rain today (1 for yes, 0 for no): "))
    month = int(input("Enter month (1-12): "))
    day = int(input("Enter day of the month (1-31): "))

    # Get prediction result
    result = predict_rainfall(temp, wind_speed, humidity, rain_today, month, day)
    print(result)

except ValueError:
    print("Invalid input. Please enter numeric values for each field.")
