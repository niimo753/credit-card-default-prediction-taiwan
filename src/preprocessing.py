import pandas as pd
import numpy as np

class WoE:
    def __init__(self, bins=5, handle_numeric=True):
        self.bins = bins
        self.handle_numeric = handle_numeric

    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        target_name = y.name
        target_count = data[target_name].value_counts().to_dict()

        # fit
        self.feature_name = []
        self.encoded = {}
        self.dist_class = {}
        self.information_value = {}
        self.num_fea_bins = {}

        if self.handle_numeric:
            num_fea = data.iloc[:, :-1].select_dtypes(exclude="object")
            bins = self.bins
            for col in num_fea.columns:
                self.num_fea_bins[col] = self.create_bins(data[col], bins=bins)
                data[col] = self.convert_into_bins(data[[col]], col, self.num_fea_bins[col])
            columns = data.columns[:-1]
        else:
            columns = data.select_dtypes(include="object").columns

        for col in columns:
            self.feature_name.append(col)
            col_value = data.groupby(col)[target_name].value_counts().to_dict()
            self.dist_class[col] = {}
            for key, value in col_value.items():
                if key[0] not in self.dist_class[col]:
                    self.dist_class[col][key[0]] = {key[1]: value/target_count[key[1]]}
                else:
                    self.dist_class[col][key[0]][key[1]] = value/target_count[key[1]]
            encoded = {}
            iv = 0
            for value, dist_class in self.dist_class[col].items():
                if 0 not in dist_class:
                    self.dist_class[col][value][0] = 0.00001
                if 1 not in dist_class:
                    self.dist_class[col][value][1] = 0.00001
                dist_0 = self.dist_class[col][value][0]
                dist_1 = self.dist_class[col][value][1]
                woe = np.log(dist_0/dist_1)

                encoded[value] = woe
                iv += (dist_0 - dist_1) * encoded[value]

            self.encoded[col] = encoded
            self.information_value[col] = iv

        self.information_value = dict(sorted(self.information_value.items(), key=lambda x: x[1], reverse=True))

    def fit_transform(self, X, y):
        self.fit(X, y)
        data = self.transform(X)
        return data

    def transform(self, X):
        data = X.copy()
        for col, bins in self.num_fea_bins.items():
            data[col] = self.convert_into_bins(data[[col]], col, bins)

        for col in self.feature_name:
            data[col] = data[col].apply(lambda x: self.encoded[col][x] if x in self.encoded[col].keys() else 0)
        
        return data

    def create_bins(self, feature, bins):
        bins = pd.cut(feature, bins=bins, right=False, retbins=True)[1]
        bins_dict = {}
        for i in range(len(bins)-1):
            if i==0:
                bins_name = f"under {bins[i]}"
                lower = bins[i]-10**10
                upper = bins[i+1]
            elif i==len(bins)-2:
                bins_name = f"over {bins[i]}"
                lower = bins[i]
                upper = bins[i+1]
            else:
                bins_name = f"{bins[i]} - {bins[i+1]}"
                lower = bins[i]
                upper = bins[i+1] + 10**10
            bins_dict[(lower, upper)] = bins_name
        return bins_dict
    
    def convert_into_bins(self, X, feature, bins):
        data = X.copy()
        for bins, bins_name in bins.items():
            data.loc[data[feature].between(bins[0], bins[1], "right"), f"{feature}_bins"] = bins_name
        return data.iloc[:, -1]

    def get_feature_names_out(self):
        return self.feature_name
    
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