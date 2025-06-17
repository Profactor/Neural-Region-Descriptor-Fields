import os, os.path as osp
import configargparse
import torch
from numpy import loadtxt
from torch.utils.data import DataLoader

from nrdf.model.imdc_pointnet_occupancy_net import IMDCPointOnet
from nrdf.training import losses, training
from nrdf.training import dataset_onet_m
from nrdf.utils import path_util
from nrdf.utils.util import make_unique_path_to_dir


if __name__ == '__main__':
    p = configargparse.ArgumentParser()
    p.add_argument('--logging_root', type=str, default=osp.join(path_util.get_nrdf_model_weights(), 'nrdf'), help='root for logging')
    p.add_argument('--obj_class', type=str, required=True, help='car, chair, plane, all')
    p.add_argument('--data_type', type=str, default='occ', help='training GT data type')
    p.add_argument('--experiment_name', type=str, required=True, help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
    
    # General training options
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--enc_pcd_size', type=int, default=1500)
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
    p.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for.')
    p.add_argument('--epochs_til_ckpt', type=int, default=5, help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=500, help='Time interval in seconds until tensorboard summary is saved.')
    p.add_argument('--iters_til_ckpt', type=int, default=10000, help='Training steps until save checkpoint')
    p.add_argument('--aug', action='store_true', help='whether to use descriptor loss function')
    p.add_argument('--rot_aug', action='store_true', help='whether to use descriptor loss function')
    p.add_argument('--full_obj', action='store_true', help='whether to use descriptor loss function')    
    p.add_argument('--position_encode', action='store_true', help='whether to use position encoding')    
    p.add_argument('--cuda', type=bool, default=True, help='True for GPU, False for CPU')
    p.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123') 
    p.add_argument('--random_sample_pcd', action='store_true', help='whether to use descriptor loss function')
    p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

    opt = p.parse_args()

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = True
    print(opt)    

    # with data loaders
    obj_class = opt.obj_class
    data_path = os.path.join(path_util.get_nrdf_data(), obj_class)
    # train dataset
    train_list_path = os.path.join(data_path, "train_nrdf.txt")
    print('train_list_path: ', train_list_path)    
    data_list = loadtxt(train_list_path, comments="#", delimiter=",", unpack=False, dtype="U")
    train_data = dataset_onet_m.ONetData(data_list=data_list, data_path=data_path, pcd_size=opt.enc_pcd_size, aug=opt.aug, rot_aug=opt.rot_aug, for_sr=True)
    train_loader = DataLoader(train_data, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    print('train len -', len(train_loader))
    # val dataset
    val_list_path = os.path.join(data_path, "val_nrdf.txt")
    print('val_list_path: ', val_list_path)    
    val_data_list = loadtxt(val_list_path, comments="#", delimiter=",", unpack=False, dtype="U")
    val_data = dataset_onet_m.ONetData(data_list=val_data_list, data_path=data_path, pcd_size=opt.enc_pcd_size, aug=opt.aug, rot_aug=opt.rot_aug, for_sr=True)
    val_loader = DataLoader(val_data, num_workers=0, batch_size=opt.batch_size, shuffle=True)
    print('val len -', len(val_loader))                                

    # -- CREATE MODEL -- #
    sigmoid = True if opt.data_type == 'occ' else False
    model = IMDCPointOnet(encode_position=opt.position_encode, sigmoid=sigmoid).cuda()
    print(model)

    # -- LOAD CHECKPOINT --#
    if opt.checkpoint_path is not None:
        model.load_state_dict(torch.load(opt.checkpoint_path))

    # Can use if have multiple gpus.  Currently untested.
    # model_parallel = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5])
    model_parallel = model

    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    root_path = make_unique_path_to_dir(root_path)

    # -- CREATE CONFIG -- #
    config = {}
    config['argparse_args'] = vars(opt)

    # -- RUN TRAIN FUNCTION -- #
    # Set loss function here
    loss_fn = val_loss_fn = losses.occupancy_mse_loss

    training.train_imdc_point_onetdata(model=model_parallel, train_dataloader=train_loader,
        val_dataloader=val_loader, epochs=opt.num_epochs, lr=opt.lr,
        steps_til_summary=opt.steps_til_summary,
        epochs_til_checkpoint=opt.epochs_til_ckpt,
        model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt,
        clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True,
        config_dict=config)

