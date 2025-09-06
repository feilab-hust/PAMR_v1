from enum import Enum, auto

class DataloaderMode(Enum):
    train = auto()
    test = auto()
    inference = auto()