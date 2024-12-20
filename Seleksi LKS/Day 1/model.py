import pandas as pd
import numpy as np

class KNN:

    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train.values
        self.y_train = y_train.values

    def single_predict(self, single_X_test):
        distances = np.array([np.linalg.norm(single_X_test-x) for x in self.X_train])
        k_nearest = distances.argsort()[:self.k]
        predicted = pd.Series(self.y_train[k_nearest]).mode().values
        return predicted[0]
        
    
    def predict(self, X_test):
        predicted_list = []
        for test in X_test.values:
            predicted_list.append(self.single_predict(test))
        return predicted_list

    def accuracy(self, y_test, y_pred):
        correct = np.sum(y_test == y_pred)
        return correct/len(y_test)

class Cross_Validation:

    def __init__(self, model, n_folds=5, rand=True, strat=True):
        self.model = model
        self.n_folds = n_folds
        self.rand = rand
        self.strat = strat
    
    def fit(self, df, target_column):
        self.X = df.drop(target_column, axis=1)
        self.y = df[target_column]

        indices = [range(len(df))]
        class_indices = [np.where(self.y == c)[0] for c in np.unique(self.y)]

        if self.rand:
            if self.strat:
                class_indices = [np.random.permutation(cat) for cat in class_indices]
                stratified = []

                for (c1, c2, c3) in zip(*class_indices):
                    stratified.append((c1,c2,c3))

                folds = np.array_split(stratified, self.n_folds)
        
                folds_conc = [np.concatenate(arr) for arr in folds]
        
                self.folds = folds_conc
            
            indices = np.random.permutation(indices[0])
            self.folds = np.array_split(indices, self.n_folds)

        self.folds = np.array_split(indices, self.n_folds)
        # return self.folds

    def run_cv(self):
        
        accuracy = []

        for idx in range(len(self.folds)):
            testing_indices = self.folds[idx]
            training_indices = np.concatenate([self.folds[x] for x in range(len(self.folds)) if x != idx])
        
            X_train, X_test, y_train, y_test = self.X.iloc[training_indices], self.X.iloc[testing_indices], self.y.iloc[training_indices], self.y.iloc[testing_indices]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy.append(self.model.accuracy(y_test, y_pred))

        return np.mean(accuracy)

def train_test_split(df, feature, target, test_ratio):
    df = df.sample(frac=1)
    
    test = df.iloc[:int(test_ratio*len(df))]
    train = df.iloc[int(test_ratio*len(df)):]
    
    X_train, y_train = train[feature], train[target]
    X_test, y_test = test[feature], test[target]

    return X_train, X_test, y_train, y_test

def stratified_split(df, feature, target, test_ratio):
    indices = []
    for (c1,c2,c3) in zip(*[np.where(df[target] == c)[0] for c in np.unique(df[target])]):
        indices.append((c1,c2,c3))
    
    conc_indices = np.concatenate(indices)
    
    testing_indices = conc_indices[:int(len(df)*test_ratio)]
    training_indices = conc_indices[int(len(df)*test_ratio):]

    X_train, y_train  = df[feature].iloc[training_indices], df[target].iloc[training_indices]
    X_test, y_test = df[feature].iloc[testing_indices], df[target].iloc[testing_indices]
    
    return X_train, X_test, y_train, y_test