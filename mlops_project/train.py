import torch

import model
from utils import training_loop


def main():
    # Загрузка датасета из DVC
    X_train = torch.load("data/x_train.pt")
    y_train = torch.load("data/y_train.pt")
    X_test = torch.load("data/x_test.pt")
    y_test = torch.load("data/y_test.pt")

    # Создание полносвязной НС
    dense_network = model.DenseNetwork(
        in_features=64, hidden_size=32, n_classes=10, n_layers=3, activation=model.ReLU
    )
    optimizer = torch.optim.LBFGS(dense_network.parameters(), max_iter=1)

    # Выбор девайса
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    dense_network.to(device)

    # Обучение
    train_losses, test_losses, train_accs, test_accs = training_loop(
        n_epochs=200,
        network=dense_network,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        ds_train=(X_train, y_train),
        ds_test=(X_test, y_test),
        device=device,
    )

    # Сохранение параметров обученной модели
    torch.save(dense_network.state_dict(), "models/dense_network.pt")


if __name__ == "__main__":
    main()
