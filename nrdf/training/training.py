import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import os.path as osp
import yaml
import shutil
from collections import defaultdict
import torch.distributed as dist
from nrdf.utils import util


def average_gradients(model):
    """Averages gradients across workers"""
    size = float(dist.get_world_size())

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

def train_imdc_point_onetdata(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None,
          config_dict={}):

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters(), betas=[0.5, 0.999])]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)
        summaries_dir_train = os.path.join(model_dir, 'summaries/train')
        util.cond_mkdir(summaries_dir_train)
        summaries_dir_val = os.path.join(model_dir, 'summaries/val')
        util.cond_mkdir(summaries_dir_train)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer_train = SummaryWriter(summaries_dir_train)
        writer_val = SummaryWriter(summaries_dir_val)

        config_fn = osp.join(model_dir, 'config.yml')
        with open(config_fn, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    total_steps = 0
    best_loss = float('inf')
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt) # gt -> Ground truth

                start_time = time.time()
                
                # occupancy prediction
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if rank == 0:
                        writer_train.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                if rank == 0:
                    writer_train.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))

                if train_loss<best_loss and total_steps>5000:
                    best_loss = train_loss
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d_best.pth' % (epoch, total_steps)))

                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_summary and rank == 0 or total_steps ==(len(train_dataloader) * epochs -1):
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = util.dict_to_gpu(model_input)
                                gt = util.dict_to_gpu(gt)
                                model_output = model(model_input)
                                val_loss = val_loss_fn(model_output, gt)

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)
                            writer_val.add_scalar(loss_name, single_loss, total_steps)
                            print(loss_name + " loss %0.6f" % (single_loss))
                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))

        return model, optimizers

def train_imdc_point_onetdata_inv_desc(model, inv_model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          sr_loss_fn=None, val_sr_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None,
          config_dict={}):

    if optimizers is None:
        params = list(model.parameters()) + list(inv_model.parameters())
        optimizers = [torch.optim.Adam(params=params, lr=lr, betas=[0.5, 0.999])]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)
        summaries_dir_train = os.path.join(model_dir, 'summaries/train')
        util.cond_mkdir(summaries_dir_train)
        summaries_dir_val = os.path.join(model_dir, 'summaries/val')
        util.cond_mkdir(summaries_dir_train)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer_train = SummaryWriter(summaries_dir_train)
        writer_val = SummaryWriter(summaries_dir_val)

        config_fn = osp.join(model_dir, 'config.yml')
        with open(config_fn, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    total_steps = 0
    best_loss = float('inf')
    best_shape_loss = float('inf')

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                torch.save(inv_model.state_dict(),
                           os.path.join(checkpoints_dir, 'inv_model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))                           
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.txt' % (epoch, total_steps)),
                           np.array(train_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'sr_train_losses_%04d_iter_%06d.txt' % (epoch, total_steps)),
                           np.array([loss_sr.item()]))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt) # gt -> Ground truth

                start_time = time.time()

                # occupancy prediction
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))

                # Descriptor-Level Self-Object Reconstruction
                model_output = model(model_input, for_im_corr=True)
                pred_pcd = inv_model(model_output['z'], model_output['features']) # features with all layer actns 2050
                loss_sr = sr_loss_fn(pred_pcd, model_input['obj_pcd'])  
                writer_train.add_scalar('loss_sr', loss_sr, total_steps)

                occ_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    if rank == 0:
                        writer_train.add_scalar(loss_name, single_loss, total_steps)
                    occ_loss += single_loss
                
                train_loss = (10*occ_loss) + loss_sr
                train_losses.append(train_loss.item())
                if rank == 0:
                    writer_train.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    torch.save(inv_model.state_dict(),
                               os.path.join(checkpoints_dir, 'inv_model_current.pth')) 
                    
                # update best model info
                if train_loss<best_loss and total_steps>5000:
                    best_loss = train_loss.item()
                    writer_train.add_scalar('loss_best', best_loss, total_steps)
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d_best.pth' % (epoch, total_steps)))
                    torch.save(inv_model.state_dict(),
                               os.path.join(checkpoints_dir, 'inv_model_epoch_%04d_iter_%06d_best.pth' % (epoch, total_steps))) 
                
                if loss_sr<best_shape_loss and total_steps>5000:
                    best_shape_loss = loss_sr.item()
                    writer_train.add_scalar('loss_sr_best', best_shape_loss, total_steps)
                    torch.save(model.state_dict(),
                            os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d_best_sr_shape.pth' % (epoch, total_steps)))
                    torch.save(inv_model.state_dict(),
                            os.path.join(checkpoints_dir, 'inv_model_epoch_%04d_iter_%06d_best_sr_shape.pth' % (epoch, total_steps)))  

                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_summary and rank == 0 or total_steps ==(len(train_dataloader) * epochs -1):
                    print("Epoch %d, Total loss %0.6f, occ loss %0.6f, sr loss %0.6f, iteration time %0.6f" % (epoch, train_loss.item(), losses['occ'].item(), loss_sr.item(), time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            inv_model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = util.dict_to_gpu(model_input)
                                gt = util.dict_to_gpu(gt)

                                model_output = model(model_input)
                                val_loss = val_loss_fn(model_output, gt)

                                model_output_sr = model(model_input, for_im_corr=True)
                                pred_pcd = inv_model(model_output_sr['z'], model_output_sr['features']) # features with all layer actns 2050
                                val_loss_sr = val_sr_loss_fn(pred_pcd, model_input['obj_pcd'])  
                                val_loss['loss_sr'] = val_loss_sr

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)
                            writer_val.add_scalar(loss_name, single_loss, total_steps)
                            print(loss_name + " loss %0.6f" % (single_loss))

                        model.train()
                        inv_model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                    torch.save(inv_model.state_dict(),
                               os.path.join(checkpoints_dir, 'inv_model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))   
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))
                    np.savetxt(os.path.join(checkpoints_dir, 'sr_train_losses_%04d_iter_%06d.txt' % (epoch, total_steps)),
                               np.array([loss_sr.item()]))

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        torch.save(inv_model.state_dict(),
                   os.path.join(checkpoints_dir, 'inv_model_final.pth'))  
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'sr_train_losses_final.txt'),
                   np.array([loss_sr.item()]))    
        return model, optimizers

def train_imdc_point_onetdata_inv_desc_cr(model, inv_model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          sr_loss_fn=None, val_sr_loss_fn=None,
          cr_loss_fn=None, val_cr_loss_fn=None, d_loss_fn=None, val_d_loss_fn=None, sm_radius=0.1,
          overwrite=True, optimizers=None, batches_per_validation=10, gpus=1, rank=0, max_steps=None,
          config_dict={}):

    if optimizers is None:
        params = list(model.parameters()) + list(inv_model.parameters())
        optimizers = [torch.optim.Adam(params=params, lr=lr, betas=[0.5, 0.999])]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    if rank == 0:
        if os.path.exists(model_dir):
            if overwrite:
                shutil.rmtree(model_dir)
            else:
                val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
                if val == 'y' or overwrite:
                    shutil.rmtree(model_dir)

        os.makedirs(model_dir)
        summaries_dir_train = os.path.join(model_dir, 'summaries/train')
        util.cond_mkdir(summaries_dir_train)
        summaries_dir_val = os.path.join(model_dir, 'summaries/val')
        util.cond_mkdir(summaries_dir_train)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        util.cond_mkdir(checkpoints_dir)

        writer_train = SummaryWriter(summaries_dir_train)
        writer_val = SummaryWriter(summaries_dir_val)

        config_fn = osp.join(model_dir, 'config.yml')
        with open(config_fn, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    total_steps = 0
    best_loss = float('inf')
    best_shape_loss = float('inf')

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch and rank == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                torch.save(inv_model.state_dict(),
                           os.path.join(checkpoints_dir, 'inv_model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))                           
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.txt' % (epoch, total_steps)),
                           np.array(train_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'sr_train_losses_%04d_iter_%06d.txt' % (epoch, total_steps)),
                           np.array([loss_sr.item()]))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = util.dict_to_gpu(model_input)
                gt = util.dict_to_gpu(gt) # gt -> Ground truth

                start_time = time.time()

                # occupancy prediction
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    
                # Descriptor-Level Self-Object Reconstruction
                model_output = model(model_input, for_im_corr=True)
                pred_pcd = inv_model(model_output['z'], model_output['features'])
                loss_sr = sr_loss_fn(pred_pcd, model_input['obj_pcd'])  
                writer_train.add_scalar('loss_sr', loss_sr, total_steps)

                # Descriptor-Level Cross-Object Reconstruction
                pred_cr_pcd = inv_model(model_output['z'], torch.cat((model_output['features'][1:,:,:], model_output['features'][:1,:,:]), dim=0))
                loss_cd, loss_normal, loss_emd, loss_smooth = cr_loss_fn(pred_cr_pcd, model_input['obj_pcd'], sm_radius=sm_radius)
                if d_loss_fn is not None:
                    loss_desc = d_loss_fn(pred_cr_pcd, model_input['obj_pcd'], model_output['features'])
                    writer_train.add_scalar('loss_desc', loss_desc, total_steps)

                loss_cr = (10*loss_cd) + (1*loss_emd) + (0.1*loss_smooth) + (0.01*loss_normal) 
                writer_train.add_scalar('loss_cr', loss_cr, total_steps)
                writer_train.add_scalar('loss_cd', loss_cd, total_steps)
                writer_train.add_scalar('loss_emd', loss_emd, total_steps)
                writer_train.add_scalar('loss_normal', loss_normal, total_steps)
                writer_train.add_scalar('loss_smooth', loss_smooth, total_steps)                    

                occ_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    if rank == 0:
                        writer_train.add_scalar(loss_name, single_loss, total_steps)
                    occ_loss += single_loss
                
                if d_loss_fn is not None:
                    train_loss = (10*occ_loss) + loss_sr + loss_cr + (0.1*loss_desc)
                else:
                    train_loss = occ_loss + loss_sr + loss_cr
                    
                train_losses.append(train_loss.item())
                if rank == 0:
                    writer_train.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    torch.save(inv_model.state_dict(),
                               os.path.join(checkpoints_dir, 'inv_model_current.pth')) 

                # update best model info
                if train_loss<best_loss and total_steps>5000:
                    best_loss = train_loss.item()
                    writer_train.add_scalar('loss_best', best_loss, total_steps)
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d_best.pth' % (epoch, total_steps)))
                    torch.save(inv_model.state_dict(),
                               os.path.join(checkpoints_dir, 'inv_model_epoch_%04d_iter_%06d_best.pth' % (epoch, total_steps))) 
                    
                if loss_sr<best_shape_loss and total_steps>5000:
                    best_shape_loss = loss_sr.item()
                    writer_train.add_scalar('loss_sr_best', best_shape_loss, total_steps)
                    torch.save(model.state_dict(),
                            os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d_best_sr_shape.pth' % (epoch, total_steps)))
                    torch.save(inv_model.state_dict(),
                            os.path.join(checkpoints_dir, 'inv_model_epoch_%04d_iter_%06d_best_sr_shape.pth' % (epoch, total_steps)))  

                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()

                if gpus > 1:
                    average_gradients(model)

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_summary and rank == 0 or total_steps ==(len(train_dataloader) * epochs -1):
                    if d_loss_fn is not None:
                        print("Epoch %d, Total loss %0.6f, occ loss %0.6f, sr loss %0.6f, cr loss %0.6f, cd loss %0.6f, emd loss %0.6f, normal loss %0.6f, smooth loss %0.6f, desc loss %0.6f, iteration time %0.6f" \
                         % (epoch, train_loss, losses['occ'].item(), loss_sr.item(), loss_cr.item(), loss_cd.item(), loss_emd.item(), loss_normal.item(), loss_smooth.item(), loss_desc.item(), time.time() - start_time))
                    else:
                        print("Epoch %d, Total loss %0.6f, occ loss %0.6f, sr loss %0.6f, cr loss %0.6f, cd loss %0.6f, emd loss %0.6f, normal loss %0.6f, smooth loss %0.6f, iteration time %0.6f" \
                         % (epoch, train_loss, losses['occ'].item(), loss_sr.item(), loss_cr.item(), loss_cd.item(), loss_emd.item(), loss_normal.item(), loss_smooth.item(), time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        with torch.no_grad():
                            model.eval()
                            inv_model.eval()
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in enumerate(val_dataloader):
                                model_input = util.dict_to_gpu(model_input)
                                gt = util.dict_to_gpu(gt)
                
                                model_output = model(model_input)
                                val_loss = val_loss_fn(model_output, gt)

                                model_output_sr = model(model_input, for_im_corr=True)
                                pred_pcd = inv_model(model_output_sr['z'], model_output_sr['features']) # features with all layer actns 2050
                                val_loss_sr = val_sr_loss_fn(pred_pcd, model_input['obj_pcd'])  
                                val_loss['loss_sr'] = val_loss_sr

                                pred_cr_pcd = inv_model(model_output_sr['z'], torch.cat((model_output_sr['features'][1:,:,:], model_output_sr['features'][:1,:,:]), dim=0))
                                loss_cd, loss_normal, loss_emd, loss_smooth = val_cr_loss_fn(pred_cr_pcd, model_input['obj_pcd'], sm_radius=sm_radius)
                                if d_loss_fn is not None:
                                    loss_desc = val_d_loss_fn(pred_cr_pcd, model_input['obj_pcd'], model_output_sr['features'])
                                    val_loss['loss_desc'] = loss_desc
                                
                                loss_cr = (10*loss_cd) + (1*loss_emd) + (0.1*loss_smooth) + (0.01*loss_normal)
                                val_loss['loss_cr'] = loss_cr
                                val_loss['loss_cd'] = loss_cd
                                val_loss['loss_emd'] = loss_emd
                                val_loss['loss_smooth'] = loss_smooth
                                val_loss['loss_normal'] = loss_normal

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())

                                if val_i == batches_per_validation:
                                    break

                        for loss_name, loss in val_losses.items():
                            single_loss = np.mean(loss)
                            writer_val.add_scalar(loss_name, single_loss, total_steps)
                            print(loss_name + " loss %0.6f" % (single_loss))

                        model.train()
                        inv_model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                    torch.save(inv_model.state_dict(),
                               os.path.join(checkpoints_dir, 'inv_model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))   
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                               np.array(train_losses))
                    np.savetxt(os.path.join(checkpoints_dir, 'sr_train_losses_%04d_iter_%06d.txt' % (epoch, total_steps)),
                               np.array([loss_sr.item()]))

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        torch.save(inv_model.state_dict(),
                   os.path.join(checkpoints_dir, 'inv_model_final.pth'))  
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'sr_train_losses_final.txt'),
                   np.array([loss_sr.item()]))    
        return model, optimizers

