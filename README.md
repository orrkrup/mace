# Fine-Tuning Generative Models as an Inference Method for Robotic Tasks
Code for the MACE (Model Adaptation with the Cross Entropy method) algorithm presented in the CoRL 2023 paper "Fine-Tuning Generative Models as an Inference Method for Robotic Tasks". See <a href="https://www.orrkrup.com/mace">project page</a> for more details. 

## Installation and Requirements

Tested in a python `virtualenv`, install dependencies using `pip install -r requirements.txt`. 

### For IsaacGym environments: 

Follow the installation instructions in https://developer.nvidia.com/isaac-gym to download IsaacGym, and once downloaded, use the instructions in the attached documnetation to install. This code was tested with IsaacGym Preview Release 4 only. 

### For VAE environments: 
VAE environments require the 3D SoftIntroVAE code, available <a href="https://github.com/orrkrup/3d-soft-intro-vae-pytorch">here</a>. Clone the repository, and in its main directory, run `pip install -e .`.  

### ShapeNet 
Running the ShapeNet Airplane environment also requires the ShapeNet dataset, or at least its Airplane class, to be located in `./shapenet_root/` (or at any other location specified by the `data_dir` argument in the `config_file`). The dataset can be obtained from the <a href="https://shapenet.org/">ShapeNet website</a>. 

## Training a Model
To train a VAE model, follow the instructions in the 3D SoftIntroVAE repository. 
At this time, code for training the IK model is unavailable; however, trained models are supplied in the `trained_models` directory. 


## Fine-tuning a Model
`train.py <config_file>` loads a pre-trained model using the given config file and fine-tunes it a ground truth object or observation.
Use `-h` for a list of additional options.

Sample command for tuning VAE model for grasping ShapeNet Airplanes: 
``` 
python train.py config/vae/config.json --sample_size 128 --bsz 128 --chamfer_k 5 --lr 0.002 --iterations 1000 --grad_steps 8 --quantile 0.0625
``` 

Example command for tuning the IK model:
```
python train.py config/ik/model_old.pt --iterations 512 --lr 0.0001 --grad_steps 4 --bsz 8192 --sample_size 8192 --quantile 0.00390625 --ik_object_type inv_bin --render_mode high_q
```

Use the `-d` or `--suppress_wandb` flags to disable logging to Weights and Biases. 

### Supported environments:
(Should be reflected in the `dataset` option in the given `config_file`)

| Task / Simulator   | Model              | Dataset (`dataset` argument)                |
| ------------------ | ------------------ | ------------------------------------------- |
| Grasping Geometric | VAE                | ShapeNet Airplanes (`shapenet`)             |
| Inverse Kinematics | IK Autoregressive  | Panda 1M IK (`ik`)                         |


### Modes of Operation:
The MACE code supports multiple modes of operation, controlled by the `--opt_mode` option. 

* `mace`: the default mode, runs the MACE algorithm as described in the paper.
* `is`: Use the importance sampling term baseline, as described in the paper.
* `get_best`: Get the best sample (and best score) of the prior-model, export to output file and quit. Used to collcet statistics. 


## Citation
If you use this code (or parts of it) in your work, please consider citing the paper:

```
@inproceedings{krupnik2023fine,
 	title={Fine-Tuning Generative Models as an Inference Method for Robotic Tasks},
  	author={Krupnik, Orr and Shafer, Elisei and Jurgenson, Tom and Tamar, Aviv},
  	booktitle={7th Annual Conference on Robot Learning},
  	year={2023}
}
```
