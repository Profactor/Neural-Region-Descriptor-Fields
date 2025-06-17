import os, os.path as osp


def get_nrdf_src():
    return os.environ['NRDF_SOURCE_DIR']


def get_nrdf_root():
    return os.environ['NRDF_ROOT_DIR']


def get_nrdf_data():
    return osp.join(get_nrdf_root(), 'data')


def get_nrdf_eval_data():
    return osp.join(get_nrdf_src(), 'eval_data')


def get_nrdf_model_weights():
    return osp.join(get_nrdf_src(), 'model_weights')

