from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def get_models():
    lr = LogisticRegression(max_iter=1000)
    nb = MultinomialNB()
    return lr, nb