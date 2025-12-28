import pandas as pd
import re
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import pickle

from model import Model

class NaiveBayesClassifier(Model):
    def __init__(self):
        pass
    
    def load_model(self, model: Pipeline):
        self.model = model

    def grid_search(self, 
        train_df: pd.DataFrame,
        output_path: str = None
        ):
        
        # train_df['rating'] = train_df['rating'].astype(int)
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('nb', MultinomialNB())
        ])
        parameters = {
            'tfidf__max_df': (0.5, 0.75, 1.0),
            'tfidf__ngram_range': ((1, 1), (1, 2)),  # Unigrams or Bigrams
            'nb__alpha': (0.1, 0.5, 1.0)              
        }
        
        grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(train_df['cleaned_text'], train_df['rating'])
        
        best_clf = grid_search.best_estimator_
        
        print(f"Best Score: {grid_search.best_score_}")
        print(f"Best Params: {grid_search.best_params_}")
                
        self.model = best_clf
        if output_path is not None:
            pickle.dump(best_clf, open(output_path, 'wb'))
        return best_clf
    
    def predict(self, texts: list) -> list:
        if self.model is None:
            raise Exception("Model is none.")
        predictions = self.model.predict(texts)
        return predictions
    