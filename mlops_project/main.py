import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import model
import train


# Загрузка и формирование датасета
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 4)
X_train, X_test = torch.from_numpy(X_train), torch.from_numpy(X_test)
y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)


# Создание НС
dense_network = model.DenseNetwork(
    in_features=64, hidden_size=32, n_classes=10, n_layers=3, activation=model.ReLU
)
optimizer = torch.optim.LBFGS(dense_network.parameters(), max_iter=1)
loss_fn = model.CrossEntropyLoss()


# Выбор девайса
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)
dense_network.to(device)
# print('device:', device)


# Обучение
train_losses, test_losses, train_accs, test_accs = train.training_loop(
    n_epochs=200,
    network=dense_network,
    loss_fn=torch.nn.CrossEntropyLoss(),
    optimizer=optimizer,
    ds_train=(X_train, y_train),
    ds_test=(X_test, y_test),
    device=device,
)
