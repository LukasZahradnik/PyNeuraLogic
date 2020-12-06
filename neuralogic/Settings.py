from . import get_neuralogic
from py4j.java_gateway import get_field


class Settings:
    def __init__(self):
        self.namespace = get_neuralogic().cz.cvut.fel.ida.setup
        # self.settings = self.namespace.Settings()

        self.settings = self.namespace.Settings.forFastTest()
