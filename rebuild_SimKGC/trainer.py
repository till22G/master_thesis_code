from model import build_model

class Trainer:
    def __init__(self, args) -> None:
        build_model(args)