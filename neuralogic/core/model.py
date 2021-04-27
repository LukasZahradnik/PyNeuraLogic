from neuralogic.core.settings import Settings


class Model:
    def __init__(self, model, settings: Settings):
        self.model = model
        self.settings = settings

    def save(self):
        pass

    def load(self):
        pass
