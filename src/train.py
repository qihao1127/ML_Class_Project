from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def train_pipeline(texts, labels):
    vectorizer = TfidfVectorizer(max_features=20000)
    X = vectorizer.fit_transform(texts)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test