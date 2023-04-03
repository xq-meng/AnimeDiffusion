import os
import threading


_util_path_lock = threading.Lock()


def mkdir(dir):
    _util_path_lock.acquire()
    if os.path.exists(dir):
        if not os.path.isdir(dir):
            raise ValueError
    else:
        os.makedirs(dir)
    _util_path_lock.release()


def create_prefix_dir(path: str):
    slash_pos = max(path.rfind('/'), path.rfind('\\'))
    if slash_pos > 0:
        mkdir(path[:slash_pos])
