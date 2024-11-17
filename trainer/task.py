import os
from afire import Fire

from trainer import experiment


if __name__ == "__main__":
    Fire(experiment.run)
