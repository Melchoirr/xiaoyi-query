from models import build_model


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.model = self._build_model()

    def _build_model(self):
        return build_model(self.args)
