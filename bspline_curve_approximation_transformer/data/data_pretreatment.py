import numpy
import torch
from torch.utils.data import Dataset


class DataPretreator:

    def __init__(self, device: torch.device, dataset: Dataset) -> None:
        super().__init__()
        self.device = device
        item = dataset.__getitem__(0)
        shape = item['input_vector'].shape
        self.input_length = shape[0]
        self.input_depth = shape[1] if len(shape) > 1 else 1
        self.output_length = [len(i) for i in item['output_vector']]

    def data_treatment(self, item):
        return self.__prepare_vector__(item['input_vector']), \
            self.__prepare_mask__(item['input_mask']), \
            [self.__prepare_mask__(i) for i in item['output_vector']], \
            [self.__prepare_mask__(i) for i in item['output_mask']], \
            [self.__prepare_mask__(i) for i in item['output_mask_net']]

    def get_input_length(self):
        return self.input_length

    def get_input_depth(self):
        return self.input_depth

    def get_output_length(self):
        return self.output_length

    def __prepare_vector__(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        tensor = tensor if len(shape) > 2 else tensor.view(shape[0], shape[1], 1)
        return tensor.to(self.device).float()

    def __prepare_mask__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device).float()


class DataPretreatorNoTorch:

    def __init__(self, device: torch.device, input_length: int, output_length: list) -> None:
        super().__init__()
        self.device = device
        self.input_length = input_length
        self.output_length = output_length

    def data_treatment(self, item):
        return self.__prepare_vector__(item['input_vector']), \
            self.__prepare_mask__(item['input_mask']), \
            [self.__prepare_mask__(i) for i in item['output_vector']], \
            [self.__prepare_mask__(i) for i in item['output_mask']], \
            [self.__prepare_mask__(i) for i in item['output_mask_net']]

    def get_input_length(self):
        return self.input_length

    def get_output_length(self):
        return self.output_length

    def __prepare_vector__(self, tensor: numpy.ndarray) -> torch.Tensor:
        tensor = torch.tensor(tensor, device=self.device).unsqueeze(0)
        shape = tensor.shape
        tensor = tensor if len(shape) > 2 else tensor.view(shape[0], shape[1], 1)
        return tensor.to(self.device).float()

    def __prepare_mask__(self, tensor: numpy.ndarray) -> torch.Tensor:
        return torch.tensor(tensor, device=self.device).unsqueeze(0).float()
