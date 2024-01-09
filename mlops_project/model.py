import torch


class ReLU(torch.nn.Module):
    def __init__(self):
        '''
        Слой ReLU поэлементно применяет Rectified Linear Unit к своему входу
        '''
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Применяет ReLU ко входному тензору
        '''
        return torch.nn.functional.relu(x)
    

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        '''
        Полносвязный слой — это слой выполняющий аффинное преобразование f(x) = x W + b
        '''
        super().__init__()
        
        # Создание необходимых обучаемых параметров
        self.weight = torch.nn.parameter.Parameter(torch.empty(in_features, out_features))
        self.bias = torch.nn.parameter.Parameter(torch.empty(out_features))
        
        # Выполнение инициализации весов
        self.reset_parameters()

        
    def reset_parameters(self):
        '''
        Инициализация весов полносвязного слоя из нормального распределения с 
            нулевым средним и стандартным отклонением 0.01
        Вектор-смещение инициализируется нулями
        '''
        torch.nn.init.normal_(self.weight, mean=0, std=0.01)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Выполнение аффинного преобразования f(x) = x W + b
        
        :param torch.Tensor x: входная матрица размера [batch_size, in_features]
        :returns: матрица размера [batch_size, out_features]
        '''
        return x @ self.weight + self.bias
    
    def __repr__(self):
        '''
        Опциональный метод для красивого вывода
        '''
        return 'Linear({0:d}, {1:d})'.format(*self.weight.shape)

    
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Применение логсофтмакса к каждой строке, а затем выборка элементов в соответствии с метками истинного класса
        :param torch.Tensor x: Матрица логитов размера [batch_size, n_classes]
        :param torch.Tensor labels: Список меток истинного класса. Размер [batch_size]
        :returns: Кросс-энтропийная функция потерь 
        """
        batch_size = labels.shape[0]
        max_x = torch.max(x, 1)[0].view(-1, 1)              # (batch_size)x1
        exp_norm_x = torch.exp(x - max_x)                   # (batch_size)x(n_classes)
        sum_exp_norm = torch.sum(exp_norm_x, 1).view(-1, 1) # (batch_size)x1
        softmax = exp_norm_x / sum_exp_norm                 # (batch_size)x(n_classes)
        return -torch.mean(torch.log(softmax[range(batch_size),labels])) # const


class DenseNetwork(torch.nn.Module):
    def __init__(self, in_features, hidden_size, n_classes, n_layers, activation=ReLU):
        '''
        :param int in_features: Число входных признаков
        :param int hidden_size: Размер скрытых слоёв
        :param int n_classes: Число выходов сети 
        :param int n_layers: Число слоёв в сети
        :param torch.nn.Module activation: Класс функции активации
        '''
        super().__init__()
        
        in_feat = None
        out_feat = None
        self.layers = torch.nn.Sequential()
        for layer in range(1, n_layers + 1):
            if layer == 1:
                in_feat = in_features
                if layer == n_layers:
                    out_feat = n_classes
                else:
                    out_feat = hidden_size
            elif layer == n_layers:
                in_feat = hidden_size
                out_feat = n_classes
            else:
                in_feat = hidden_size
                out_feat = hidden_size
            self.layers.append(Linear(in_feat, out_feat))
            if layer != n_layers:
                self.layers.append(activation())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Прямой проход по сети
        :param torch.Tensor x: Входной тензор размера [batch_size, in_features]
        :returns: Матрица логитов размера [batch_size, n_classes]
        '''
        return self.layers(x)
