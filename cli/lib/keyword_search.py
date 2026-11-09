import json
import string
import pickle
import os
import math

from collections import Counter
from nltk.stem import PorterStemmer

BM25_K1 = 1.5
BM25_B = 0.75

stemmer = PorterStemmer()

translator = str.maketrans("", "", string.punctuation)

data_path = "data/movies.json"
stopwords_path = "data/stopwords.txt"
CACHE_DIR = "cache"

with open(stopwords_path, "r") as file:
    stopwords = set(file.read().splitlines())

def tokenize_text(text: str) -> list[str]:
    tokens = text.lower().translate(translator).split()

    tokens = [token for token in tokens if token not in stopwords]

    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

def load_movies():
    with open(data_path, "r") as file:
        data = json.load(file)

    return data["movies"]

class InvertedIndex:

    
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        
    def __add_document(self,doc_id,text):
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
                
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
            
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def get_document(self, term):
        return sorted(self.index.get(term, set()))
    
    
    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id, {}).get(term, 0)
    
    
    def build(self):
        movies = load_movies()
        
        for movie in movies:
            doc_id = movie["id"]
            
            self.docmap[doc_id] = movie
            
            self.__add_document(doc_id,f"{movie['title']} {movie['description']}")
            
    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)

        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)

        with open(self.term_frequencies_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)

        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)


    def load(self):
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)

        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)

        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)

        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = pickle.load(file)
            
    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_document(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term: str, k1=BM25_K1, b=BM25_B) -> float:
        raw_tf = self.get_tf(doc_id,term)
        avg_dl = self.__get_avg_doc_length()
        doc_length = self.doc_lengths[doc_id]
        
        length_norm = 1 - b + b * (doc_length / avg_dl)

        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        
        return bm25_tf
    
    def bm25(self, doc_id: int, term:str) -> float:
        return (self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term))
         
    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)
        score_dict = {}
        
        for doc in self.docmap:
            bm25_running = 0.0
            for token in tokens:
                token_score = self.bm25(doc,token)
                bm25_running += token_score
            score_dict[doc] = bm25_running
        
        sorted_scores = sorted(
            score_dict.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return sorted_scores[:limit]
        

    
def build_command():
    index = InvertedIndex()
    
    index.build()
    index.save()
    
def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)
    
    if len(tokens) != 1:
        raise ValueError("Expect exactly 1 token")
    
    return tokens[0]

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    
    tokens = tokenize_term(term)
    
    return float(index.get_bm25_idf(tokens))
    
def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
    index = InvertedIndex()
    index.load()
    
    tokens = tokenize_term(term)
    
    return index.get_bm25_tf(doc_id,tokens,k1,b)