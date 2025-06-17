import torch
from nrdf.training.chamfer_new import chamfer_distance as chamfer_new
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from nrdf.training.loss_topo import smooth_loss
from pytorch3d.ops.knn import knn_gather, knn_points
from emd_module.emd_module import emdModule


emd_dist = emdModule()
criterion = torch.nn.MSELoss()

def occupancy_mse_loss(model_outputs, ground_truth,  val=False):
    loss_dict = dict()
    loss_dict['occ'] = criterion(model_outputs['occ'], ground_truth['occ'].squeeze())
    return loss_dict

def selfrec_loss(esti_shapes, shapes):
    return  criterion(esti_shapes, shapes)

def selfrec_emd_chamf_normal_smooth_loss(esti_shapes, shapes, esti_shapes_normals=None, shapes_normals=None, abs_cosine=False, sm_radius=0.1, smooth=True):
    esti_normals = estimate_pointcloud_normals(esti_shapes) if esti_shapes_normals is None else esti_shapes_normals 
    shape_normals = estimate_pointcloud_normals(shapes) if shapes_normals is None else shapes_normals 
    chamf_loss, normal_loss, _, _, _, _ = chamfer_new(x=esti_shapes, y=shapes, x_normals=esti_normals, y_normals=shape_normals, abs_cosine=abs_cosine)
    
    dist, _ = emd_dist(esti_shapes, shapes, 0.005, 50)
    loss_emd = torch.sqrt(dist).mean(1).mean()

    if smooth:
        loss_smooth = smooth_loss(esti_shapes, shapes, radius=sm_radius)
        return  chamf_loss, normal_loss, loss_emd, loss_smooth
    return  chamf_loss, normal_loss, loss_emd

def desc_loss_fn(e_shape, shape, shape_features):
    # e_shape to shape
    e_shape_features = torch.cat((shape_features[1:,:,:], shape_features[:1,:,:]), dim=0)
    e_nn = knn_points(e_shape, shape, K=1)
    e_features_near = knn_gather(shape_features, e_nn.idx)[..., 0, :]
    feat_dis_e_s = torch.sqrt(torch.sum((e_shape_features - e_features_near)**2, dim=-1))
    feat_dis_e_s = feat_dis_e_s.mean(1).mean()
    
    # shape to e_shape
    s_nn = knn_points(shape, e_shape, K=1)
    s_features_near = knn_gather(e_shape_features, s_nn.idx)[..., 0, :]
    feat_dis_s_e = torch.sqrt(torch.sum((shape_features - s_features_near)**2, dim=-1))
    feat_dis_s_e = feat_dis_s_e.mean(1).mean()

    return feat_dis_e_s + feat_dis_s_e

