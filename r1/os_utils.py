import os


def n_cores() -> int:
    return os.cpu_count()
