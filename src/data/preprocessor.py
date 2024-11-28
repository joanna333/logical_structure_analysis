import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from typing import List, Dict, Tuple
import pandas as pd
import spacy
import networkx as nx

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = re.sub(r'\[[0-9]*\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        return text.strip()

    def segment_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        return sent_tokenize(text)

    def process_article(self, text: str) -> Dict:
        """Full preprocessing pipeline."""
        cleaned_text = self.clean_text(text)
        sentences = self.segment_sentences(cleaned_text)
        
        return {
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'num_sentences': len(sentences)
        }

    def extract_relations(self, text: str) -> Tuple[List[Dict], nx.DiGraph]:
        """Extract relations and build initial graph."""
        doc = self.nlp(text)
        relations = []
        graph = nx.DiGraph()
        
        for sent in doc.sents:
            # Extract subject-verb-object triples
            for token in sent:
                if token.dep_ == "nsubj":
                    subject = token.text
                    verb = token.head.text
                    obj = None
                    for child in token.head.children:
                        if child.dep_ == "dobj":
                            obj = child.text
                            relations.append({
                                'subject': subject,
                                'predicate': verb,
                                'object': obj
                            })
                            # Add to graph
                            graph.add_edge(subject, obj, relation=verb)
        
        return relations, graph