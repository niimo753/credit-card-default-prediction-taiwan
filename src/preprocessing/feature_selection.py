from .encoding import WoE

class IVSelector(WoE):
    def __init__(self, threshold=0.02, bins=3):
        super().__init__()
        self.threshold = threshold
        self.bins = bins

    def fit(self, X, y):
        super().fit(X, y)
        threshold = self.threshold
        self.drop = [col for col in self.information_value.keys() if self.information_value[col] < threshold]
        self.feature_name = [col for col in self.feature_name if col not in self.drop]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        data = X.drop(self.drop, axis=1)
        return data

    def transform(self, X):
        data = X.drop(self.drop, axis=1)
        return data

    def get_feature_names_out(self):
        return self.feature_name