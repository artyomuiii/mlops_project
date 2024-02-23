import os

import hydra
import torch
from omegaconf import DictConfig

import model
import utils


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Загрузка датасета из DVC
    os.system("dvc pull --force")
    X = torch.load(cfg.data.x_infer_path)
    y = torch.load(cfg.data.y_infer_path)

    # Загрузка параметров обученной модели
    dense_network = model.DenseNetwork(
        in_features=cfg.model.in_features,
        hidden_size=cfg.model.hidden_size,
        n_classes=cfg.model.n_classes,
        n_layers=cfg.model.n_layers,
        activation=model.ReLU,
    )
    dense_network.load_state_dict(torch.load(cfg.save_params.path))

    # Выбор девайса
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    dense_network.to(device)

    # Применение модели
    utils.inference(dense_network, torch.nn.CrossEntropyLoss(), (X, y), device)


if __name__ == "__main__":
    main()
