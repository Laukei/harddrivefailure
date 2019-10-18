from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from data import get_data


def main():
    df = get_data()

    print('feature encoding')
    features = ['will_fail','model','manufacturer']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])

    X_train, X_test, y_train, y_test = train_test_split(df.drop(['will_fail'],axis=1), df['will_fail'], test_size=0.1)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()