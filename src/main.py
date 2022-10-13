import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier

def main():
    
    cwd = os.getcwd()
    data = pd.read_csv(os.path.join(cwd, "../", "dataset", "data_with_class_column.csv"))

    model = KNeighborsClassifier(n_neighbors=5)
    X = data[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']].values
    y = data['Class'].values
    model.fit(X, y)

    print(model.predict([[40, -90, 70, 10, 10, 10]]))


if __name__ == "__main__":
    main()

    #42.394348	-105.038998	65.494029	0.518491	-0.008071	-0.510420 => 0 1 1 0 => 6