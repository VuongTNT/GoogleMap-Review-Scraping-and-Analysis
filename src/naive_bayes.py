import pandas as pd
import re
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle

from model import Model

class NaiveBayesClassifier(Model):
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        # pass
    
    def load_model(self, model: Pipeline):
        self.model = model

    def advanced_clean(self, text: str) -> str:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = word_tokenize(text)
        cleaned_tokens = [
            lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in stop_words and len(token) > 2
        ]
        return " ".join(cleaned_tokens)

    def grid_search(self, 
        train_df: pd.DataFrame,
        output_path: str = None
        ):
        
        train_df['cleaned_text'] = train_df['text'].apply(self.advanced_clean)
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
        if output_path:
            pickle.dump(best_clf, open(output_path, 'wb'))
        return best_clf
    
    def predict(self, texts: list) -> list:
        cleaned_texts = [self.advanced_clean(text) for text in texts]
        if self.model is None:
            raise Exception("Model is none.")
        predictions = self.model.predict(cleaned_texts)
        return predictions
    