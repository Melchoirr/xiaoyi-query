import torch
from models import get_model


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = get_model(self.args)
        return model

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
        elif self.args.use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Use MPS')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        raise NotImplementedError

    def train(self, setting):
        raise NotImplementedError

    def vali(self, vali_data, vali_loader, criterion):
        raise NotImplementedError

    def test(self, setting, test=0):
        raise NotImplementedError
