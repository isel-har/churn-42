import numpy as np


class Initializers:

    rng = np.random.default_rng(seed=42)

    @classmethod
    def he_init(cls, shape:tuple):
        std_dev = np.sqrt(2.0 / shape[0])
        return cls.rng.random(size=shape, dtype=np.float32) * std_dev
    

    @classmethod
    def random_init(cls, shape:tuple):
        return cls.rng.random(size=shape, dtype=np.float32)
    

    @classmethod
    def xavier_init(cls, shape:tuple):
        std_dev = np.sqrt(2.0 / (shape[0] + shape[1]))
        return cls.rng.random(size=shape, dtype=np.float32) * std_dev


    @staticmethod
    def constant_init(val, shape:tuple):
        return np.full(shape=shape, fill_value=val, dtype=np.float32)
