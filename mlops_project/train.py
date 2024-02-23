import os

import hydra
import torch
from omegaconf import DictConfig

import model
from utils import training_loop


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Загрузка датасета из DVC
    os.system("dvc pull --force")
    X_train = torch.load(cfg.data.x_train_path)
    y_train = torch.load(cfg.data.y_train_path)
    X_test = torch.load(cfg.data.x_test_path)
    y_test = torch.load(cfg.data.y_test_path)

    # Создание полносвязной НС
    dense_network = model.DenseNetwork(
        in_features=cfg.model.in_features,
        hidden_size=cfg.model.hidden_size,
        n_classes=cfg.model.n_classes,
        n_layers=cfg.model.n_layers,
        activation=model.ReLU,
    )
    optimizer = torch.optim.LBFGS(dense_network.parameters(), max_iter=1)

    # Выбор девайса
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    dense_network.to(device)

    # Обучение
    train_losses, test_losses, train_accs, test_accs = training_loop(
        n_epochs=cfg.training.n_epochs,
        network=dense_network,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        ds_train=(X_train, y_train),
        ds_test=(X_test, y_test),
        device=device,
    )

    # Сохранение параметров обученной модели
    torch.save(dense_network.state_dict(), cfg.save_params.path)


if __name__ == "__main__":
    main()
