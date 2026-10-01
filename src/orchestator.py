import yaml
from src.network import get_dataloaders

#TODO Not Implemented yet, Just a placeholder for the orchestrator module. 
# 1. Upload the configuration from the YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# 2. This batch size is used for the PSO and DBO algorithms, 
# and it is also used to create the data loaders for training and testing. 
# The batch size can be adjusted in the YAML configuration file.
#TODO This should not be hardcoded, but rather read from the configuration file.
batch_size_particles = 64 

# 3. Inject the batch size and data directory from the configuration 
# into the get_dataloaders function to create the data loaders for training and testing.
train_loader, test_loader = get_dataloaders(
    batch_size=batch_size_particles,
    data_dir=config['dataset']['data_dir'],
    train_subset_ratio=config['dataset']['train_subset_ratio']
)