# Neural-Region-Descriptor-Fields

Official Pytorch implementation of the IROS 2023 conference paper: NRDF - Neural Region Descriptor Fields as Implicit ROI Representation for Robotic 3D Surface Processing.

Paper: [NRDF](https://ieeexplore.ieee.org/document/10802862)

Real World Demonstration Results: [Video](https://www.youtube.com/watch?v=YiEGInDQT-o)


# Installation


### Clone this repo

```
git clone https://github.com/Profactor/Neural-Region-Descriptor-Fields.git
cd Neural-Region-Descriptor-Fields 
```

### Setup and install with poetry
Poetry Info [Link](https://python-poetry.org/docs/)
```
poetry install
```

### Install additional dependencies
Considering torch and pytorch3d dependencies compatible with CUDA 11.7 version.

#### torch
```
python -m pip install torch==1.13.1 torchvision --extra-index-url https://download.pytorch.org/whl/cu117
```

#### pytorch3d
```
python -m pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt1131/download.html
```
#### emd (Earth Mover's Distance)
```
# clone EMD repo from https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd
cd emd
python setup.py install
```


# Training
The Occuppancy ground truths can be generated using the [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks?tab=readme-ov-file#dataset) repo which uses ShapeNet objects.

Please first select for a given class/category, objects with similar part constituency and then list them out in text files, e.g., 'train_nrdf.txt', 'val_nrdf.txt'

Generate Occupancy ground truths for the selected objects and place them in the folder `data/<category_name>`. Replace the category name accordingly. Also keep the selected object list text files ('train_nrdf.txt', and 'val_nrdf.txt') in the same folder.

Then to train run:
```
source nrdf_env.sh # source environment variables
python3 scripts/run_train_nrdf_subprocess.py 

```
replace `<category_name>` with the actual category name.


# Citing
```
@inproceedings{pratheepkumar2024nrdf,
  title={NRDF-Neural Region Descriptor Fields as Implicit ROI Representation for Robotic 3D Surface Processing},
  author={Pratheepkumar, Anish and Ikeda, Markus and Hofmann, Michael and Widmoser, Fabian and Pichler, Andreas and Vincze, Markus},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={12955--12962},
  year={2024},
  organization={IEEE}
}
```

# Acknowledgements
We build on the setup found in [Neural Descriptor Fields](https://github.com/anthonysimeonov/ndf_robot). We also utilize the following repos [occupancy networks](https://github.com/autonomousvision/occupancy_networks), [MSN Point Cloud Completion](https://github.com/Colin97/MSN-Point-Cloud-Completion.git), and the [Implicit Dense Correspondence](https://github.com/liufeng2915/Implicit_Dense_Correspondence). Thanks to these great contributions!
