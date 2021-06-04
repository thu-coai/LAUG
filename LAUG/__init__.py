import os

from LAUG.nlu import NLU
from LAUG.nlg import NLG


from os.path import abspath, dirname


def get_root_path():
    return dirname(dirname(abspath(__file__)))


DATA_ROOT = os.path.join(get_root_path(), 'data')
REPO_ROOT = os.path.join(get_root_path())