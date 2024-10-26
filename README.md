# Satellite Rainfall Data for Agriculture


https://github.com/user-attachments/assets/46c5a80c-8c6e-4d09-98b9-893cb3c70445


## Challenge
Create a tool to better understand and predict rainfall patterns using satellite data, focusing on agriculture and water management.

## Problem Statement
Farmers and water managers need timely and accurate predictions of rainfall to make informed decisions regarding irrigation, crop planning, and water resource management. This project aims to provide a reliable predictive model combined with real-time satellite data and interactive features to assist in these tasks.

# https://github.com/AdityaJ9801/Ai_Chatbot_CPGRAM/assets/124603391/8f7494d3-7dd8-4d42-b762-54a37f08c2df

## Flowchart
![WhatsApp Image 2024-10-26 at 2 23 03 PM](https://github.com/user-attachments/assets/d818b55f-a77c-4545-93cc-423efd69ca0c)

## Installation

To get started, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/AdityaJ9801/Ai_Chatbot_CPGRAM.git
    cd Ai_Chatbot_CPGRAM
    ```
2. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Mac: source venv/bin/activate
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the app:
    ```bash
    streamlit run app.py
    ```

## Solution Overview
We developed a web application that predicts rainfall patterns and provides real-time weather updates using satellite data. The application is designed to assist farmers and stakeholders in agricultural planning and water management.

### Key Features
1. **Rainfall Prediction Model**:
   - Predicts whether the weather will be sunny or rainy based on current temperature, wind speed, humidity, and other factors.
2. **Real-time Weather Data**:
   - Fetches live satellite data for temperature, humidity, and wind speed using the Open Meteo API.
3. **Interactive Chatbot**:
   - Provides farmers with advice and answers questions using the Gemini API. The chatbot offers insights based on farm data, current weather conditions, and rainfall predictions.
4. **Data Visualization**:
   - Displays current weather data on a map based on latitude and longitude inputs.
   - Visualizes weather trends and predictions through graphs.

## Dataset
- **Source**: Kaggle daily weather dataset (2008-2017)
- **Attributes Used**: Temperature, humidity, wind speed, rain_today, month, day
- **Processing**: Cleaned and filtered dataset to include only relevant attributes for the prediction model.

## Model Training & Evaluation
We trained the dataset using several supervised classification algorithms:
- **CatBoostClassifier**: 68% accuracy
- **RandomForest**: 69% accuracy
- **Logistic Regression**: 71% accuracy
- **GaussianNB**: 69% accuracy
- **KNeighborsClassifier**: 66% accuracy
- **XGBoostClassifier**: 70% accuracy

Based on evaluation results, we chose the **XGBoostClassifier** model for its better accuracy and performance.

## Web Application
The web application was built using **Streamlit** and integrates three core components:
1. **Open Meteo API**: Provides real-time satellite data on temperature, humidity, and wind speed.
2. **Gemini API**: Implements a chatbot for interaction and tailored advice based on current weather and farm data.
3. **Prediction Model**: Predicts whether it will be sunny or rainy based on real-time data inputs (temperature, humidity, wind speed, etc.).

### Additional Features
- **Map Visualization**: Displays current weather data on a map using longitude and latitude.
- **Graphical Trends**: Offers data visualizations to show weather patterns over time.

## Setup & Installation

### Prerequisites
- Python 3.7+
- Required Libraries: 
  - Streamlit
  - XGBoost
  - Open Meteo API integration
  - Gemini API integration

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/satellite-rainfall-agriculture.git

## References
1. [Langchain Documentation - Getting Started](https://python.langchain.com/docs/get_started/introduction)
2. [Langchain Documentation - Vector Stores with FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss)
3. [Mistral-7B-Instruct Model Paper](https://mistral.ai/assets/Mistral_7B_paper_v_0_1.pdf)
4. [Anaconda Blog - How to Build a Retrieval-Augmented Generation Chatbot](https://www.anaconda.com/blog/how-to-build-a-retrieval-augmented-generation-chatbot)



