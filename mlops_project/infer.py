import torch

import model
import utils


def main():
    # Загрузка датасета из DVC
    X = torch.load("data/x_test.pt")
    y = torch.load("data/y_test.pt")

    # Загрузка параметров обученной модели
    dense_network = model.DenseNetwork(
        in_features=64, hidden_size=32, n_classes=10, n_layers=3, activation=model.ReLU
    )
    dense_network.load_state_dict(torch.load("models/dense_network.pt"))

    # Выбор девайса
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    dense_network.to(device)

    # Применение модели
    utils.inference(dense_network, torch.nn.CrossEntropyLoss(), (X, y), device)


if __name__ == "__main__":
    main()
