import subprocess
import time

# Occupancy Prediction
print("nrdf_<category_name>_exp_1")
cmd_1 = "python3 scripts/train_nrdf_occ.py --obj_class <category_name> --experiment_name nrdf_<category_name>_exp_1 --steps_til_summary 500 --num_epochs 500 --batch_size 8  --data_type occ --enc_pcd_size 8192"
std_out_cmd_1 = subprocess.call(cmd_1, shell=True)
time.sleep(5)

# Descriptor-Level Self-Object Reconstruction
print("nrdf_<category_name>_self_recon_exp_1")
cmd_2 = "python3 scripts/train_nrdf_occ_self_recon.py --obj_class <category_name> --batch_size 4 --experiment_name nrdf_<category_name>_self_recon_exp_1 --steps_til_summary 500 --num_epochs 500 --stage1_folder nrdf_<category_name>_exp_1_0 --data_type occ --enc_pcd_size 8192"
std_out_cmd_2 = subprocess.call(cmd_2, shell=True)
time.sleep(5)

# Descriptor-Level Cross-Object Reconstruction
print("nrdf_<category_name>_self_recon_cross_recon_exp_1")
cmd_3 = "python3 scripts/train_nrdf_occ_self_recon_cross_recon.py --obj_class <category_name> --batch_size 2 --desc_loss --sm_radius 0.1 --experiment_name nrdf_<category_name>_self_recon_cross_recon_exp_1 --steps_til_summary 500 --num_epochs 500 --stage2_folder nrdf_<category_name>_self_recon_exp_1_0  --data_type occ --enc_pcd_size 8192"
std_out_cmd_3 = subprocess.call(cmd_3, shell=True)
time.sleep(5)

