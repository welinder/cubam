from Model import *
from numpy import sign

class BinaryModel(Model):
    def __init__(self, filename=None, data=None):
        Model.__init__(self, filename, data)
