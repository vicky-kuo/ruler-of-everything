import os

def getAbsolutePath(path: str):
    absolutePath = os.path.dirname(__file__)
    return os.path.join(absolutePath, path)