import data_prep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import json
from sklearn.metrics import classification_report

def split_data(df, split=0.3):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    # don't need to scale data for RandomForest

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=64)

    return X_train, X_test, y_train, y_test

def load_model_hyperparams(file):
    with open(file) as json_file:
        hp = json.load(json_file)

    return hp

def train_model(best_params, X, y):
    classifier = RandomForestClassifier(criterion=best_params['criterion'], max_depth=best_params['max_depth'],\
        min_samples_leaf=best_params['min_samples_leaf'], min_samples_split=best_params['min_samples_split'],\
            max_features=best_params['max_features'], n_estimators=best_params['n_estimators'])

    classifier.fit(X, y)

    return classifier

def predict_test(model, X, y):
    prediction = model.predict(X)
    print(classification_report(y, prediction))

    return prediction

def main():
    df = data_prep.main()
    X_train, X_test, y_train, y_test = split_data(df)
    best_params = load_model_hyperparams('hyperparams.txt')
    classifier = train_model(best_params, X_train, y_train)
    prediction = predict_test(classifier, X_test, y_test)

    return classifier, X_train, X_test, y_train, y_test, prediction

if __name__ == '__main__':
    main()