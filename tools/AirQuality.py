import requests
from langchain_core.tools import tool # type: ignore
from dotenv import dotenv_values
from pathlib import Path
from typing import Dict, Any


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] 
ENV_PATH = ROOT / '.env'
config = dotenv_values(ENV_PATH) 
userToken = config['userToken']

# https://python.langchain.com/docs/how_to/custom_tools/
@tool(parse_docstring=True)
def get_air_quality(location: str) -> Dict[str, Any]:
    """
    This API can be used to search stations by name
    
    Args:
        location: city name (English) if the input is Chineses, you should translate from Chinese  English
    
    Returns:
        A JSON dictionary containing the search results. An example response might look like:
        {
          "status": "ok",
          "data": [
            {
              "uid": 1234,
              "aqi": "53",
              "time": "2023-05-12T13:00:00Z",
              "station": {...}
            },
            ...
          ]
        }
        If the status is "error", the response might look like:
        {
          "status": "error",
          "data": "Invalid Key"
        }
    """
    # Build the base URL for searching stations by name
    base_url = "https://api.waqi.info/search/"

    # Prepare the query parameters
    params = {
        "keyword": location,
        "token": userToken
    }

    # Make the GET request
    response = requests.get(base_url, params=params)
    
    # Convert response to JSON
    data = response.json()
    
    # You may wish to handle potential errors here:
    if data.get("status") != "ok":
        # Handle error, raise exception, or simply return the response
        # (depends on your use case)
        pass
    
    return data

# Example usage:
if __name__ == "__main__":
    print(f"Token loaded from .env: {userToken}")
    result = get_air_quality("台中; Xitun")
    print(result)
