from afire import Fire  # type: ignore[import-untyped]

from trainer import experiment

if __name__ == "__main__":
    Fire(experiment.run)
