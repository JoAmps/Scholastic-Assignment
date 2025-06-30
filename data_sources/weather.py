import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPEN_WEATHER_API_KEY = os.getenv("open_weather_api_key")

if not OPEN_WEATHER_API_KEY:
    raise EnvironmentError("API keys for OpenWeather is missing.")


def get_location_data(city_name: str, limit: int = 5) -> List[Dict]:
    """
    Get latitude/longitude and other geolocation data for a city.

    Args:
        city_name (str): Name of the city (e.g. "London").
        limit (int): Max number of results to return (default: 5).

    Returns:
        List[Dict]: A list of geolocation results.
    """
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city_name, "limit": limit, "appid": OPEN_WEATHER_API_KEY}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"[Location Error] Failed to fetch location data: {e}")
        return []


def get_current_weather(city_name, units: str = "metric") -> Dict:
    """
    Fetch current weather conditions for a location by lat/lon.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        units (str): Unit system - 'standard', 'metric', or 'imperial'. Default is 'metric'.

    Returns:
        Dict: Current weather details (temperature, humidity, description, etc.).
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    results = get_location_data(city_name)
    params = {
        "lat": results[0]["lat"],
        "lon": results[0]["lon"],
        "appid": OPEN_WEATHER_API_KEY,
        "units": units,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"[Weather Error] Failed to fetch weather data: {e}")
        return {}
