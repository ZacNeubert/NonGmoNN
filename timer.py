from time import time


class Timer:
    def __init__(self, handler):
        self.start = None
        self.handler = handler

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.handler(time() - self.start)
