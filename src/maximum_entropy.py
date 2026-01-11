import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from model import Model

class MaximumEntropy(Model):
    def __init__(self, model: LogisticRegression, vectorizer: TfidfVectorizer):
        self.model = model
        self.vectorizer = vectorizer
    
    def load_model(model_path: str,
        vectorizer_path: str
        ):
        print('Loading ME...')
        model = pickle.load(open(model_path, 'rb'))
        vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        return MaximumEntropy(model, vectorizer)
        
    def predict(self, texts: list):
        # Vectorize
        vec = self.vectorizer.transform(texts)
        # Predict
        pred_labels = self.model.predict(vec)
        pred_labels = [int(label) for label in pred_labels] ##turn labels text into int
        pred_probas = np.max(self.model.predict_proba(vec), axis=1)
        return pred_labels, pred_probas