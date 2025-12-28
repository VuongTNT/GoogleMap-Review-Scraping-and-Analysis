import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from maximum_entropy import MaximumEntropy
from model import Model
from naive_bayes import NaiveBayesClassifier
from scraper import Scraper
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



reviews_train_df = pd.read_csv('data/reviews_train_cleaned.csv')
reviews_val_df = pd.read_csv('data/reviews_val_cleaned.csv')
reviews_test_df = pd.read_csv('data/reviews_test_cleaned.csv')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def evaluate_model(model: Model):
    y_pred = model.predict(reviews_test_df['cleaned_text'].tolist())
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    
    unique_ratings = sorted(reviews_test_df['rating'].unique())
    target_names_5class = [f'Rating {r}' for r in unique_ratings]
    
    accuracy = accuracy_score(reviews_test_df['rating'], y_pred)
    report = classification_report(reviews_test_df['rating'], y_pred, labels=unique_ratings, target_names=target_names_5class, zero_division=0)
    conf_mat = confusion_matrix(reviews_test_df['rating'], y_pred)
    
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_mat)
    
    return accuracy, report, conf_mat

def advanced_clean(text: str) -> str:
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
    
if __name__ == "__main__":
    try:
        # SCRAPER_PATH = "D:\\Programming\\HUST\\google-maps-scraper\\google-maps-scraper" 
        # scraper = Scraper()
        # # scraper.scrape_loc_id(loc_scraper_dir=SCRAPER_PATH, loops=5)
        # API_KEY = "b80ddec496msh388f25a33756ca7p159f7cjsn019f45abd9b1"
        # scraper.scrape_loc_data(API_KEY, max_loc=5)
        # scraper.extract_reviews()
        
        me = MaximumEntropy()
        me.load_model(
            pickle.load(open('models/me_model.sav', 'rb')),
            pickle.load(open('models/me_vectorizer.sav', 'rb'))
            )
        nb = NaiveBayesClassifier()
        nb.load_model(pickle.load(open('models/nb_model.sav', 'rb')))
        
        print("- Enter a review to analyze sentiment")
        print("- type '#exit' to quit")
        print("- type '#eval' to evaluate models on test set")
        while True:
            print("-" * 50)
            user_input = input("> ")
            
            if user_input.lower() == '#exit':
                break
            if user_input.lower() == '#eval':
                print('Evaluate Maximum Entropy Classifier:')
                evaluate_model(me)
                
                print('Evaluate Naive Bayes Classifier:')
                evaluate_model(nb)
                continue
            
            cleaned_input = advanced_clean(user_input)
            
            nb_pred = nb.predict([cleaned_input])[0]
            me_pred, me_proba = me.predict([cleaned_input])
            me_pred = me_pred[0]
            me_proba = me_proba[0]
            
            print(f"Naive Bayes Prediction: Rating {nb_pred}")
            print(f"Maximum Entropy Prediction: Rating {me_pred} ({me_proba:.4f} confidence)")
        
        
        # print('Loaded Maximum Entropy')
        # print('Evaluate Maximum Entropy Classifier:')
        # evaluate_model(me)
        
        # print('Loaded Naive Bayes')
        # print('Evaluate Naive Bayes Classifier:')
        # evaluate_model(nb)
        
        
        # demo_reviews = [
        # "This is absolutely terrible, the worst I have seen.",
        # "The purchase was great! I'm completely satisfied.", 
        # "It works fine, I guess. Nothing exciting.", 
        # "A good choice, worth the money.", 
        # "Below average, slightly disappointed.", 
        # ]
        # demo_reviews_cleaned = [advanced_clean(review) for review in demo_reviews]
        
        # nb_preds = nb.predict(demo_reviews_cleaned)
        # me_preds, me_probas = me.predict(demo_reviews_cleaned)
        
        # for i, review in enumerate(demo_reviews):
        #     print(f"Review: {review}")
        #     print(f"  Naive Bayes Prediction: Rating {nb_preds[i]}")
        #     print(f"  Maximum Entropy Prediction: Rating {me_preds[i]} ({me_probas[i]:.4f} confidence)")
        
    except KeyboardInterrupt:
        pass
