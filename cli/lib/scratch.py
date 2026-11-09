from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem("man"))
print(stemmer.stem("men"))
print(stemmer.stem("actor"))
print(stemmer.stem("actors"))
print(stemmer.stem("acting"))