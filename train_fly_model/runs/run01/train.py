#%%
import yaml
import torch
import logging
from fly_organelles.run import run

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)

log_dir = "/nrs/cellmap/aurrecoecheas/tensorboard/panc_lsds"

voxel_size = (16, 16, 16)
l_rate = 0.5e-5
batch_size = 14

affinities_map = [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
    ]


N_AFFINITIES = len(affinities_map)

#%%
#%%
labels = ['axon']


#%%


yaml_file = "/groups/espinosamedina/home/aurrecoecheas/jrc-muss-pancreas-innervation/train_fly_model/yamls/datasets_generated_all.yaml"
iterations = 1000000



with open(yaml_file, "r") as data_yaml:
    datasets = yaml.safe_load(data_yaml)


from isolated_unet import UNetLSD, load_checkpoint_from_path

model = UNetLSD()
status = load_checkpoint_from_path(model, "/groups/cellmap/cellmap/aurrecoecheas/lsds_work/398151")
print(status)  # Should print "<All keys matched successfully>"

x = torch.rand((1, 1, 196, 196, 196))
y = model(x)
print(y.shape)

import gunpowder as gp
input_size = gp.Coordinate( 196, 196, 196)
output_size = gp.Coordinate(92, 92, 92)


model = model.to(device)
run(model,
iterations, 
labels, 
None, 
datasets,
voxel_size = voxel_size,
batch_size = batch_size, 
l_rate=l_rate,
log_dir=log_dir,
affinities = True, 
affinities_map = affinities_map,
)
