import sys
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

from maximum_entropy import MaximumEntropy
from model import Model
from naive_bayes import NaiveBayesClassifier
from scraper import Scraper
from svm import SVMModel

# reviews_train_df = pd.read_csv('data/reviews_train_cleaned.csv')
# reviews_val_df = pd.read_csv('data/reviews_val_cleaned.csv')
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
    precision = precision_score(reviews_test_df['rating'], y_pred, average='weighted', zero_division=0)
    recall = recall_score(reviews_test_df['rating'], y_pred, average='weighted', zero_division=0)
    f1 = f1_score(reviews_test_df['rating'], y_pred, average='weighted', zero_division=0)
    report = classification_report(reviews_test_df['rating'], y_pred, labels=unique_ratings, target_names=target_names_5class, zero_division=0)
    conf_mat = confusion_matrix(reviews_test_df['rating'], y_pred)
    
    print(f"Accuracy: {accuracy:.4f}\n")
    print(f"Precision: {precision:.4f}\n")
    print(f"Recall: {recall:.4f}\n")
    print(f"F1-score: {f1:.4f}\n")
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
        
        try:
            me = MaximumEntropy.load_model(
                'models/me_model.sav', 
                'models/me_vectorizer.sav')

            nb = NaiveBayesClassifier.load_model('models/nb_model.sav')
            
            svm = SVMModel.load_model(
                "models/svm-tfidf-10k.pkl", 
                "models/tfidf-vectorize-10k.pkl")
            
        except FileNotFoundError as e:
            print(f"File not found: {e.filename}.\nTrain model or download from https://drive.google.com/drive/folders/1nxuMonis8rXSIdpVh_eei8Mu6mrW2CTz")
            
        print("- Enter a review to analyze sentiment")
        print("- type '#exit' to quit")
        print("- type '#eval' to evaluate models on test set")
        while True:
            print("-" * 50)
            user_input = input("> ")
            
            if user_input.lower() == '#exit':
                break
            if user_input.lower() == '#eval':
                print('Evaluate Naive Bayes Classifier:')
                evaluate_model(nb)
                
                print("+" * 20)
                print('Evaluate Maximum Entropy Classifier:')
                evaluate_model(me)
                
                print("+" * 20)
                print('Evaluate SVM:')
                evaluate_model(svm)
                continue
            
            cleaned_input = advanced_clean(user_input)
            
            nb_pred = nb.predict([cleaned_input])[0]
            me_pred, me_proba = me.predict([cleaned_input])
            me_pred = me_pred[0]
            me_proba = me_proba[0]
            svm_pred = svm.predict([cleaned_input])[0]
            
            print(f"Naive Bayes Prediction: Rating {nb_pred}")
            print(f"Maximum Entropy Prediction: Rating {me_pred} ({me_proba:.4f} confidence)")
            print(f"SVM Prediction: Rating {svm_pred}")
        
        
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
        sys.exit(0)
