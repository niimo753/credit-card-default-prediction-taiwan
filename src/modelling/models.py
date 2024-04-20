import numpy as np

class DefaultPaymentClassifier:
    def __init__(self, processor, model, balance=None, threshold=0.5):
        self.processor = processor
        self.model = model
        self.balance = balance
        self.threshold = threshold
    
    def fit(self, X, y):
        processed_x = self.processor.fit_transform(X, y)
        if self.balance is not None:
            processed_x, y = self.balance.fit_resample(processed_x, y)
        self.model.fit(processed_x, y)

    def predict(self, X):
        processed_x = self.processor.transform(X)
        if self.threshold == 0.5:
            predited = self.model.predict(processed_x)
        else:
            prob = self.model.predict_proba(processed_x).T[1]
            predited = np.array([1 if i >= self.threshold else 0 for i in prob])
        return predited
    
    def predict_proba(self, X):
        processed_x = self.processor.transform(X)
        predicted_proba = self.model.predict_proba(processed_x)
        return predicted_proba