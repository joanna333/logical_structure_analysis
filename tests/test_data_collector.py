import pytest
from src.data.data_collector import WikiDataCollector
from src.data.preprocessor import TextPreprocessor

def test_wiki_collector():
    collector = WikiDataCollector()
    articles = collector.collect_articles(['Biology', 'Physiology'])
    assert len(articles) > 0
    assert isinstance(articles['Biology'], str)

def test_preprocessor():
    preprocessor = TextPreprocessor()
    text = "This is a test. This is another test."
    result = preprocessor.process_article(text)
    assert len(result['sentences']) == 2
    assert result['cleaned_text'] == "this is a test. this is another test."