import requests

# API Key for OpenWeather
OPENWEATHER_API_KEY = "7de154493d96503ed06862f051361fb3"

def get_weather(address):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={address}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()

        # Log the full response for debugging
        print(f"API Response: {data}")

        if response.status_code != 200:
            print(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
            return None, None

        # Check if the necessary fields are present in the response
        if 'main' in data and 'temp' in data['main'] and 'weather' in data and len(data['weather']) > 0:
            temperature = data['main']['temp']
            weather_desc = data['weather'][0]['description']
            return temperature, weather_desc
        else:
            print("Weather data format is invalid or missing expected fields")
            return None, None

    except Exception as e:
        print(f"Error with weather API: {e}")
        return None, None
