import os
import shutil

def clear_and_remake_directory(a_dir):
    if os.path.exists(a_dir):
        shutil.rmtree(a_dir)
    os.makedirs(a_dir)

def make_dir(dir_path):
    if os.path.exists(dir_path):
        return
    else:
        os.makedirs(dir_path)
