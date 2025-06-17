import numpy as np
import os
import torch


random_state = 0
seed = 0
np.random.seed(seed)
rng = np.random.default_rng()


def angle_axis(angle, axis):
    # REF: https://github.com/NolenChen/3DStructurePoints/blob/master/dataset/data_utils.py
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    return R.float()


class PointcloudRandomRotate(object):
    # REF: https://github.com/NolenChen/3DStructurePoints/blob/master/dataset/data_utils.py
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self):
        rotation_angle = np.random.uniform() * 2 * np.pi
        axis = np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()])
        axis = axis / np.sqrt(np.sum(axis * axis))
        rotation_matrix = angle_axis(rotation_angle, axis)
        return rotation_matrix

pcd_rot = PointcloudRandomRotate()    


class ONetData(object):
    def __init__(self, data_list, data_path, pcd_size=1500, sr_pcd_size=8192, coord_size=10240, aug=False, importance_sample=False, rot_aug=False, for_sr=False):
        self.data_list = data_list
        self.data_path = data_path
        self.sr_pcd_size = sr_pcd_size
        self.aug = aug
        self.pcd_size = pcd_size
        self.rot_aug = rot_aug
        self.coord_size = coord_size
        self.for_sr = for_sr
        self.importance_sample = importance_sample
            
    def load_data(self, obj_id):
        pcd_data = np.load(os.path.join(self.data_path, obj_id, "pointcloud.npz"))
        coord_data = np.load(os.path.join(self.data_path, obj_id, "bbx_points.npz"))

        pcd = torch.from_numpy(pcd_data['points'])
        rix = torch.randperm(pcd.size(0))
        point_cloud = pcd[rix[:self.pcd_size]]   
        obj_pcd = pcd[rix[:self.sr_pcd_size]] 

        # Break symmetry if given in float16:
        points = coord_data['points']
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)
        coord = torch.from_numpy(points)
        occupancies = coord_data['occupancies']
        if occupancies.max()>1:
            occupancies = torch.from_numpy(np.unpackbits(occupancies)[:coord.shape[0]])
        else:
            occupancies = torch.from_numpy(occupancies[:coord.shape[0]])

        coord_l = torch.cat((coord, occupancies.unsqueeze(1)), dim=1)
        if self.importance_sample:
            occupied = coord_l[occupancies.bool()][:int(self.coord_size/2)] 
            unoccupied = coord_l[~occupancies.bool()][:int(self.coord_size-len(occupied))] 
            coord_l = torch.cat((occupied, unoccupied), dim=0)
        rix = torch.randperm(coord_l.shape[0])
        coord_l = coord_l[rix[:int(self.coord_size)]]
        coord = coord_l[:,:3]
        labels = coord_l[:,3:4]
              
        # mean center
        center = obj_pcd.mean(dim=0)
        coord = coord - center[None, :]
        point_cloud = point_cloud - center[None, :]
        if self.for_sr:
            obj_pcd = obj_pcd - center[None, :]

        if self.rot_aug:
            rot_mat = pcd_rot()
            point_cloud = torch.matmul(point_cloud.float(), rot_mat.t())
            obj_pcd = torch.matmul(obj_pcd.float(), rot_mat.t())
            coord = torch.matmul(coord, rot_mat.t())            

        if self.aug:
            diameter = torch.sqrt(torch.sum((torch.max(point_cloud,dim=0).values-torch.min(point_cloud,dim=0).values)**2).float())

            scale_rand = rng.random((1,3))+0.5
            scale_shape = point_cloud*scale_rand
            scale_obj_pcd = obj_pcd*scale_rand
            scale_points = coord*scale_rand

            scale_diameter = torch.sqrt(torch.sum((torch.max(scale_shape,dim=0).values-torch.min(scale_shape,dim=0).values)**2).float())
            
            point_cloud = scale_shape/scale_diameter*diameter
            obj_pcd = scale_obj_pcd/scale_diameter*diameter
            coord = scale_points/scale_diameter*diameter

        if self.for_sr:
            res = {'point_cloud': point_cloud.float(),
                'coords': coord.float(),
                'obj_pcd': obj_pcd.float()}
        else:
            res = {'point_cloud': point_cloud.float(),
                'coords': coord.float()}
        return res, {'occ': labels.float()} 

    def __getitem__(self, index):
        obj_id = str(self.data_list[index])
        model_input, gt = self.load_data(obj_id=obj_id)
        return model_input, gt

    def __len__(self):
        return len(self.data_list)

