import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from feature_engine.encoding import WoEEncoder
from .encoding import WoE


class Preprocessing:
    def __init__(self, scaler=None, encoder=None, numeric_to_object=None, numeric_into_bins=None, specific_encoders=None,
                 feature_selector=None):
        self.scaler = scaler
        self.encoder = encoder
        self.specific_encoders = specific_encoders
        self.num_to_obj = numeric_to_object
        self.numbins = numeric_into_bins
        self.cate_feas = None
        self.num_feas = None
        self.selector = feature_selector
    
    def fit(self, X, y):
        data = X.copy()
        if self.selector is not None:
            self.selector.fit(data, y)

        if self.num_to_obj is not None:
            data[self.num_to_obj] = data[self.num_to_obj].astype("object")
        if self.numbins is not None:
            for feature, bins in self.numbins.items():
                if not isinstance(bins, dict):
                    self.numbins[feature] = self.create_bins(data[feature], bins=bins)
                data[feature] = self.convert_into_bins(data[[feature]], feature, self.numbins[feature])

        self.cate_feas = list(data.select_dtypes(include="object").columns)
        self.num_feas = [col for col in data.columns if col not in self.cate_feas]

        if self.specific_encoders is not None:
            self.cate_feas = [col for col in self.cate_feas if col not in self.specific_encoders.keys()]
            for feature, encoder in self.specific_encoders.items():
                if not isinstance(encoder, OneHotEncoder):
                    if (isinstance(encoder, WoEEncoder)) or (isinstance(encoder, WoE)):
                        self.specific_encoders[feature].fit(X=data[feature], y=y)
                    else:
                        self.specific_encoders[feature].fit(data[feature])
                else:
                    self.specific_encoders[feature].fit(data[[feature]])

        if self.scaler is not None:
            self.scaler.fit(data[self.num_feas])

        if self.encoder is not None:
            if (isinstance(self.encoder, WoEEncoder)) or (isinstance(self.encoder, WoE)):
                self.encoder.fit(X=data[self.cate_feas], y=y)
            else:
                self.encoder.fit(data[self.cate_feas])

    def fit_transform(self, X, y):
        self.fit(X, y)
        data = self.transform(X)
        return data

    def transform(self, X):
        data = X.copy()

        if self.num_to_obj is not None:
            data[self.num_to_obj] = data[self.num_to_obj].astype("object")
        if self.numbins is not None:
            for feature in self.numbins.keys():
                data[feature] = self.convert_into_bins(data[[feature]], feature, self.numbins[feature])

        if self.specific_encoders is not None:
            for feature, encoder in self.specific_encoders.items():
                if isinstance(encoder, OneHotEncoder):
                    encoded = self.specific_encoders[feature].transform(data[[feature]]).toarray()
                    encoded = pd.DataFrame(encoded, columns=self.specific_encoders[feature].get_feature_names_out(), index=data.index)
                    data = pd.concat([data, encoded], axis=1)
                    data = data.drop(feature, axis=1)
                else:
                    data[feature] = self.specific_encoders[feature].transform(data[feature])

        if self.scaler is not None:
            data[self.num_feas] = self.scaler.transform(data[self.num_feas])
        
        if self.encoder is not None:
            if isinstance(self.encoder, OneHotEncoder):
                encoded = self.encoder.transform(data[self.cate_feas]).toarray()
                encoded = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(), index=data.index)
                data = pd.concat([data, encoded], axis=1)
                data = data.drop(self.cate_feas, axis=1)
            else:
                data[self.cate_feas] = self.encoder.transform(data[self.cate_feas])
            
            data = data.fillna(0)
            
        if self.selector is not None:
            data = self.selector.transform(data)

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