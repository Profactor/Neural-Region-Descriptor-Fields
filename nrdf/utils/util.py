import os
import os.path as osp
import collections


def make_unique_path_to_dir(base_path: str) -> str:
    """
    Add index to base_path until the path is unique
    Assumes that base path is leading to a directory

    Args:
        base_path (str): path leading to directory 

    Returns:
        str: path with appropriate index appended to it
    """
    ### Make unique root path name ###
    path_index = 0
    final_base_path = base_path + '_' + str(path_index)
    while osp.isdir(final_base_path):
        path_index += 1
        final_base_path = base_path + '_' + str(path_index)
    return final_base_path

def dict_to_gpu(ob):
    if isinstance(ob, collections.abc.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()
    
def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

