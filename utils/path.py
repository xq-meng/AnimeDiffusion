import os


def mkdir(dir):
    if os.path.exists(dir):
        return os.path.isdir(dir)
    else:
        os.makedirs(dir)
    return True


def create_prefix_dir(path: str):
    slash_pos = max(path.rfind('/'), path.rfind('\\'))
    if slash_pos > 0:
        mkdir(path[:slash_pos])