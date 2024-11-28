import wikipediaapi
import pandas as pd
from typing import List, Dict
import logging

class WikiDataCollector:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia('TMS-Prep-Agent/1.0', 'en')
        self.logger = logging.getLogger(__name__)

    def collect_articles(self, topics: List[str]) -> Dict[str, str]:
        """Collect Wikipedia articles for given topics."""
        articles = {}
        for topic in topics:
            page = self.wiki.page(topic)
            if page.exists():
                articles[topic] = page.text
                self.logger.info(f"Collected article: {topic}")
            else:
                self.logger.warning(f"Article not found: {topic}")
        return articles

    def save_articles(self, articles: Dict[str, str], output_path: str):
        """Save collected articles to CSV."""
        df = pd.DataFrame(list(articles.items()), columns=['topic', 'content'])
        df.to_csv(output_path, index=False)