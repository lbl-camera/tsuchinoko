import pathlib


def path(item: str):
    return str(pathlib.Path(pathlib.Path(__file__).parent, item))
