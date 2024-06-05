# All the imports are inside the functions to not have to install them all
def tfidf():
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(TfidfVectorizer(), TruncatedSVD(100))
    return pipe
