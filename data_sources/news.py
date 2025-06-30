import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("news_api_key")

if not NEWS_API_KEY:
    raise EnvironmentError("API keys for News API is missing.")


def fetch_news_articles(
    query: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    sort_by: str = "publishedAt",
    page_size: int = 5,
) -> List[Dict]:
    """
    Retrieve recent news articles based on a keyword/topic.

    Args:
        query (str): Search term (e.g., city or topic).
        from_date (str, optional): Start date in 'YYYY-MM-DD' format.
        to_date (str, optional): End date in 'YYYY-MM-DD' format.
        sort_by (str): 'relevancy', 'popularity', or 'publishedAt'.
        page_size (int): Number of articles to return (default: 5).

    Returns:
        List[Dict]: List of articles with title, description, URL, and publication date.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "sortBy": sort_by,
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("articles", [])
        return [item["title"] for item in data]
    except requests.RequestException as e:
        print(f"[NewsAPI Error] Failed to fetch articles: {e}")
        return []
