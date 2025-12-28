import pickle
import pandas as pd

from model import Model
from naive_bayes import NaiveBayesClassifier
from scraper import Scraper
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

        
def evaluate_model(model: Model, test_df: pd.DataFrame):
    y_pred = model.predict(test_df['text'].tolist())
    
    unique_ratings = sorted(test_df['rating'].unique())
    target_names_5class = [f'Rating {r}' for r in unique_ratings]
    
    accuracy = accuracy_score(test_df['rating'], y_pred)
    report = classification_report(test_df['rating'], y_pred, labels=unique_ratings, target_names=target_names_5class, zero_division=0)
    conf_mat = confusion_matrix(test_df['rating'], y_pred)
    
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_mat)
    
    return accuracy, report, conf_mat

if __name__ == "__main__":
    try:
        # SCRAPER_PATH = "D:\\Programming\\HUST\\google-maps-scraper\\google-maps-scraper" 
        # scraper = Scraper()
        # # scraper.scrape_loc_id(loc_scraper_dir=SCRAPER_PATH, loops=5)
        # API_KEY = "b80ddec496msh388f25a33756ca7p159f7cjsn019f45abd9b1"
        # scraper.scrape_loc_data(API_KEY, max_loc=5)
        # scraper.extract_reviews()

        reviews_train_df = pd.read_csv('data/reviews_train.csv')
        reviews_val_df = pd.read_csv('data/reviews_val.csv')
        reviews_test_df = pd.read_csv('data/reviews_test.csv')
        
        demo_reviews = [
        "This is absolutely terrible, the worst I have seen.",
        "The purchase was great! I'm completely satisfied.", 
        "It works fine, I guess. Nothing exciting.", 
        "A good choice, worth the money.", 
        "Below average, slightly disappointed.", 
        ]
        
        nb = NaiveBayesClassifier()
        nb.load_model(pickle.load(open('models/nb_model.sav', 'rb')))
        
        print('Evaluate Naive Bayes Classifier:')
        evaluate_model(nb, reviews_test_df)
        nb_preds = nb.predict(demo_reviews)
        for review, pred in zip(demo_reviews, nb_preds):
            print(f"Review: '{review}'\nPredicted Rating: {pred} / 5\n")
    
    except KeyboardInterrupt:
        pass
