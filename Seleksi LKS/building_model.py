import numpy as np
import pandas as pd

class KNN:

    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train:pd.core.frame.DataFrame, y_train:pd.core.series.Series):
        self.num_col = X_train.select_dtypes(include=np.number).columns
        self.cat_col = X_train.select_dtypes(exclude=np.number).columns
        self.num_X_train = X_train[self.num_col].values
        self.cat_X_train = X_train[self.cat_col].values

        self.y_train = y_train.values

    def _single_predict(self, single_X_test:pd.core.series.Series):
        self.num_single_X_test = single_X_test[self.num_col].values
        self.cat_single_X_test = single_X_test[self.cat_col].values
        
        if self.num_X_train.shape[1] > 0:
            num_distance = [np.linalg.norm(self.num_X_train[i] - self.num_single_X_test) for i in range(len(self.num_X_train))]
        
        else:
            num_distance = 0

        if self.cat_X_train.shape[1] > 0:
            cat_distance = [np.mean(self.cat_X_train[i] != self.cat_single_X_test) for i in range(len(self.cat_X_train))]

        else:
            cat_distance = 0

        distance = np.array(num_distance) + np.array(cat_distance)

        k_nearest_indices = distance.argsort()[:self.k]

        return pd.Series(self.y_train[k_nearest_indices]).mode().values[0]

    def predict(self, X_test:pd.core.frame.DataFrame):
        
        predicted_value = [self._single_predict(X_test.iloc[i]) for i in range(len(X_test))]
        return predicted_value

    def accuracy(self, y_test:pd.core.series.Series, y_pred):
        return np.sum(y_test == y_pred)/len(y_test)

class CV:

    def __init__(self, model, n_rep=1, n_folds=5, rand=True, strat=True):
        self.model = model
        self.n_rep = n_rep
        self.n_folds = n_folds
        self.rand = rand
        self.strat = strat

    def fit(self, X:pd.core.frame.DataFrame, y:pd.core.series.Series):
        self.X = X
        self.y = y
        indices = np.arange(len(X))
        class_indices = [np.where(y == c)[0] for c in np.unique(y)]
        
        self.folds = []
        
        for _ in range(self.n_rep):
            if self.strat:
                if self.rand:
                    class_indices = [np.random.permutation(idx) for idx in class_indices]

                split = [np.array_split(c, self.n_folds) for c in class_indices]
                folds = []
                for (c1,c2,c3,c4,c5) in zip(*folds):
                    folds.append(np.concatenate((c1,c2,c3,c4,c5)))
            
            if self.rand:
                indices = np.random.permutation(indices)
                folds = np.array_split(indices, self.n_folds)


            self.folds.append(np.array(folds))


    def run_cv(self):

        accuracy = []
        
        for rep in self.folds:
            for i in range(len(rep)):
                test_indices = rep[i]
                train_indices_idx = [idx for idx in range(len(rep)) if idx != i]
                train_indices = np.concatenate(rep[train_indices_idx])
                X_train, y_train, X_test, y_test = self.X.iloc[train_indices], self.y.iloc[train_indices], self.X.iloc[test_indices], self.y.iloc[test_indices]
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                accuracy.append(self.model.accuracy(y_test, y_pred))
            
        return np.mean(accuracy)
            