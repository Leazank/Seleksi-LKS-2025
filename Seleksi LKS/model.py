import pandas as pd
import numpy as np

class KNN:

    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self,X_train:pd.core.frame.DataFrame, y_train:pd.core.series.Series) -> None:
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        class_segmented = [np.where(self.y_train == i)[0] for i in range(len(np.unique(self.y_train)))]
        
        self.dictionary = {key:self.X_train[value] for key,value in zip(np.unique(self.y_train), class_segmented)}

    def predict(self, X_test):
        
        self.X_test = np.array(X_test)

        predicted_arr = []
        for cat in self.X_test:
            predicted_arr.append(self._single_predict(cat))
        
        return predicted_arr
    
    def accuracy(self, y_test:pd.core.series.Series, y_pred:list):
        self.y_test = np.array(y_test)
        self.y_pred = np.array(y_pred)

        correct = np.sum(self.y_test == self.y_pred)
        accuracy = correct/len(y_test)

        return accuracy

    def _single_predict(self, X_test) -> None:
        self.X_test = np.array(X_test)

        predicted = []
        for category in self.dictionary:
            for data in self.dictionary[category]:
                distance = np.sqrt(np.sum((self.X_test-data)**2))
                predicted.append((distance,category))
        
        predicted.sort()

        predicted_class = []
        
        for i in range(self.k):
            predicted_class.append(predicted[i])

        classes = np.unique(predicted_class, return_counts=True)
        majority_indices = classes[1].argmax()

        return classes[0][majority_indices]
    
class Cross_validation:

    def __init__(self, model, n_folds=5, n_rep=1, strat=True, rand=True):
        self.model = model
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.strat = strat
        self.rand = rand


    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        indices = [x for x in range(len(self.X))]
        class_indices = [np.where(self.y == c) for c in range(len(np.unique(self.y)))]

        if self.strat:

            if self.rand:
                class_indices = [np.random.permutation(c[0]) for c in class_indices]
                splitted = [np.array_split(c, self.n_folds) for c in class_indices]

                folds = []
                for (c1, c2, c3) in zip(*splitted):
                    folds.append(np.concatenate((c1,c2,c3)))

                
            
            else:
                splitted = [np.array_split(c[0], self.n_folds) for c in class_indices]

                folds = []
                for (c1, c2, c3) in zip(*splitted):
                    folds.append(np.concatenate((c1,c2,c3)))
            
        
        
        else:

            if self.rand:
                indices = np.random.permutation(indices)

                folds = np.array_split(indices, self.n_folds)

            else:
                folds = np.array_split(indices, self.n_folds)
            

        self.folds = folds
        return folds
        

    def run_cv(self):
        
        rep_accuracy = []

        for _ in range(self.n_rep):

            folds_accuracy = []
            liat_index = []
            for idx in range(len(self.folds)):

                test_indices = self.folds[idx]
                train_indices = [self.folds[x] for x in range(len(self.folds)) if x != idx]

                X_train, X_test, y_train, y_test = self.X[train_indices[0]], self.X[test_indices], self.y[train_indices[0]], self.y[test_indices]

                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                folds_accuracy.append(self.model.accuracy(y_test, y_pred))
            
            rep_accuracy.append(np.mean(folds_accuracy))
        
        return np.mean(rep_accuracy)


