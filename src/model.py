from abc import abstractmethod

class Model:
    @abstractmethod    
    def predict(self, texts: list) -> list:
        pass
    