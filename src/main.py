import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
import pickle

MODEL_PATH = "model.pkl"

class KNN:

    def __init__(self, neighbors):
        self.neighbors = neighbors
        self.model = KNeighborsClassifier(n_neighbors=neighbors)


    def fit(self):
        cwd = os.getcwd()
        data = pd.read_csv(os.path.join(cwd, "../", "dataset", "data_with_class_column.csv"))
        X = data[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values
        y = data['Class'].values
        self.model.fit(X, y)

        if not os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "wb") as m:
                pickle.dump(self.model, m)
    
    def getModel():
        with open(MODEL_PATH, "rb") as m:
            return pickle.load(m)
    
    def predict(self, X):
        return self.model.predict(X)
    

def main():
    
    if not os.path.exists(MODEL_PATH):
        model = KNN(5)
        model.fit()

    else:
        model = KNN.getModel()

    print(model.predict([[-40, 9, -70, -100, 100, 100]]))


if __name__ == "__main__":
    main()

    #42.394348	-105.038998	65.494029	0.518491	-0.008071	-0.510420 => 0 1 1 0 => 6