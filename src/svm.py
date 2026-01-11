from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from joblib import load

from model import Model


class SVMModel(Model):
    def __init__(self, svm: LinearSVC, vectorizer: TfidfVectorizer):
        self.svm = svm
        self.vec = vectorizer

    def predict(self, texts: list):
        vec = self.vec.transform(texts)
        pred_labels = self.svm.predict(vec)
        return pred_labels

    def __str__(self):
        res = f"SVM-Model[\n   {str(self.svm)},\n   {str(self.vec)}\n]"
        return res

    def load_model(svm_path: str, vectorizer_path: str):
        print('Loading SVM...')
        with open(svm_path, 'rb') as f:
            svm: LinearSVC = load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer: TfidfVectorizer = load(f)
        
        return SVMModel(svm, vectorizer)