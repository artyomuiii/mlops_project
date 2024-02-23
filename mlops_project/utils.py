import torch
import tqdm.notebook as tqdm


def training_loop(n_epochs, network, loss_fn, optimizer, ds_train, ds_test, device):
    """
    :param int n_epochs: Число итераций оптимизации
    :param torch.nn.Module network: Нейронная сеть
    :param Callable loss_fn: Функция потерь
    :param torch.nn.Optimizer optimizer: Оптимизатор
    :param Tuple[torch.Tensor, torch.Tensor] ds_train: Признаки и метки истинного класса обучающей выборки
    :param Tuple[torch.Tensor, torch.Tensor] ds_test: Признаки и метки истинного класса тестовой выборки
    :param torch.Device device: Устройство на котором будут происходить вычисления
    :returns: Списки значений функции потерь и точности на обучающей и тестовой выборках после каждой итерации
    """
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
    for epoch in tqdm.tqdm(range(n_epochs), total=n_epochs, leave=True):
        # Переводим сеть в режим обучения
        network.train()

        # Итерация обучения сети
        def closure():
            """
            Функция-замыкания для подсчёта градиентов функции потерь по обучающей выборке:
                1. Очистка текущих градиентов
                2. Выполнение прямого прохода по сети в вычисление функции потерь
                3. Вычисление градиентов функции потерь
            :returns: Значение функции потерь
            """
            optimizer.zero_grad()

            pred = network(ds_train[0].float().to(device))
            loss = loss_fn(pred, ds_train[1].to(device))

            loss.backward()
            return loss

        # Шаг оптимизации
        optimizer.step(closure)

        # Переводим сеть в инференс режим
        network.eval()

        # При тестировании сети нет необходимости считать градиенты, поэтому можно отключить автоматическое дифференцирование
        #   для ускорения операций
        with torch.no_grad():
            # Вычисление качества и функции потерь на обучающей выборке
            y_pred = network(ds_train[0].float().to(device))
            loss = loss_fn(y_pred, ds_train[1].to(device))
            acc = (
                torch.sum(torch.argmax(y_pred, 1) == ds_train[1].to(device))
                / ds_train[1].to(device).shape[0]
            )
            train_losses.append(loss)
            train_accuracies.append(acc * 100)

            # Вычисление качества и функции потерь на тестовой выборке
            y_pred = network(ds_test[0].float().to(device))
            loss = loss_fn(y_pred, ds_test[1].to(device))
            acc = (
                torch.sum(torch.argmax(y_pred, 1) == ds_test[1].to(device))
                / ds_test[1].to(device).shape[0]
            )
            test_losses.append(loss)
            test_accuracies.append(acc * 100)

            if epoch % 20 == 0:
                print(
                    "Loss (Train/Test): {0:.3f}/{1:.3f}. Accuracy, % (Train/Test): {2:.2f}/{3:.2f}".format(
                        train_losses[-1],
                        test_losses[-1],
                        train_accuracies[-1],
                        test_accuracies[-1],
                    )
                )

    return train_losses, test_losses, train_accuracies, test_accuracies


def inference(network, loss_fn, ds, device):
    network.eval()

    with torch.no_grad():
        y_pred = network(ds[0].float().to(device))
        torch.save(y_pred, "models/y_pred.pt")

        loss = loss_fn(y_pred, ds[1].to(device))
        acc = (
            torch.sum(torch.argmax(y_pred, 1) == ds[1].to(device))
            / ds[1].to(device).shape[0]
        ) * 100

        print("Loss: {0:.3f}. Accuracy, %: {1:.2f}".format(loss, acc))
