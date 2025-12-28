from abc import abstractmethod

class Model:
    @abstractmethod    
    def predict(self, texts: list) -> list:
        pass
    
    @abstractmethod
    def grid_search(self, train_df):
        pass