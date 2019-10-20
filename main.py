from data import get_data, preprocess_data, to_tensor
from model import Net

import torch
from sklearn.metrics import balanced_accuracy_score

def main():
    print('getting data...')
    df = get_data()
    X_train, X_test, y_train, y_test, enc = preprocess_data(df)

    print('training...')
    model = Net()
    target = to_tensor(y_train)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(100):
        y_pred = model(to_tensor(X_train))
        print(y_pred)
        print(target)
        loss = loss_fn(y_pred,target)
        print(f'loss on pass {i}: {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('trained!')
    y_pred = enc.inverse_transform(model.predict(to_tensor(X_test)))
    print('y_test')
    y_test_deenc = enc.inverse_transform((y_test))
    print(y_test_deenc)
    print('y_pred')
    print((y_pred))
    print(f'accuracy score: {balanced_accuracy_score(y_test_deenc,y_pred)}')
    return X_train, X_test, y_train, y_test, model


if __name__ == "__main__":
    main()