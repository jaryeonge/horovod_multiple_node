import torch
from collections import UserDict


class BatchTensor(UserDict):
    """
    This class is derived from a python dictionary and can be used as a dictionary.
    In addition, this class converts a dataset into an input type suitable for the pytorch model.
    """

    def __init__(self, data: dict):
        '''
        :param data: dict{key: value(torch.Tensor)}
        :param pad: bool
        '''
        super(BatchTensor, self).__init__()
        self.data = {}
        self.convert_to_tensors(data)

    def convert_to_tensors(self, data: dict):
        for key, value in data.items():
            if torch.is_tensor(value):
                self.data[key] = value
            else:
                self.data[key] = torch.as_tensor(value)

    def split_labels(self):
        try:
            labels = self.pop('labels')
        except KeyError:
            raise
        return self, labels

    def to(self, device):
        self.data = {k: v.to(device=device) for k, v in self.data.items()}
        return self

    def cuda(self):
        self.data = {k: v.cuda() for k, v in self.data.items()}
        return self
