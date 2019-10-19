from data import get_data, preprocess_data, to_tensor
from model import Net

import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder

def main():
    print('getting data...')
    df = get_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print('training...')
    model = Net()
    target = to_tensor(y_train)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(5):
        y_pred = model(to_tensor(X_train))
        loss = loss_fn(y_pred,target)
        print(f'loss on pass {i}: {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('trained!')
    print(f'accuracy score: {balanced_accuracy_score(y_test,model.predict(to_tensor(X_test)))}')
    return X_train, X_test, y_train, y_test, model


if __name__ == "__main__":
    main()