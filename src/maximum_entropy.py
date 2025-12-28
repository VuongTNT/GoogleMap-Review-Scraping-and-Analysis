import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from model import Model

class MaximumEntropy(Model):
    def __init__(self):
        pass
    
    def load_model(self, 
        model: LogisticRegression,
        vectorizer: TfidfVectorizer
        ):
        self.model = model
        self.vectorizer = vectorizer
        
    
    
    def predict(self, texts: list):
        # Vectorize
        vec = self.vectorizer.transform(texts)
        # Predict
        pred_labels = self.model.predict(vec)
        pred_labels = [int(label) for label in pred_labels] ##turn labels text into int
        
        pred_probas = np.max(self.model.predict_proba(vec), axis=1)
        
        return pred_labels, pred_probas