import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import yaml
import copy

# Prunes a model given a configuration provided to the function. Config will define in yml format the sparsity for each layer type
# Example config file is in "example.yml"

class Pruner():
    def __init__(self, model, config_path, global_pruning):
        self.model = copy.deepcopy(model)
        self.config_path = config_path
        self.global_pruning = global_pruning
        self.load_config()
    
    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def prune_model(self):
        
        if self.global_pruning:
            method = get_global_method(self.config)
            parameters_to_prune = get_global_params(self.model, self.config)
            
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=method,
                amount=self.config['sparsity'],
            )
        else:
            method = get_local_method(self.config)
            pruning_dict = get_pruning_dict(self.config)
            
            for name, module in self.model.named_modules():
                for layer_inst, params in pruning_dict.items():
                    if isinstance(module, layer_inst):
                        method(module, name=params[0], amount=params[1])
        return self.model


# Get the parameters to prune in the form of a tuple of tuples with (module, 'parameter') such as (model.conv1, 'weight')
def get_global_params(model, config):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if name in config['global_pruning']['layers']:
            parameters_to_prune.append((module, 'weight'))
    return tuple(parameters_to_prune)

def get_global_method(config):
    method_dict = {
        'L1Unstructured': prune.L1Unstructured
    }
    
    return method_dict[config['method']]

def get_local_method(config):
    method_dict = {
        'L1Unstructured': prune.l1_unstructured
    }
    
    return method_dict[config['method']]

def get_pruning_dict(config):
    pruning_dict = {}
    for layer_type, params in config['local_pruning'].items():
            layer_instance = get_layer_instance(layer_type)
            pruning_dict[layer_instance] = params
    return pruning_dict

def get_layer_instance(layer_str):
    layer_dict = {
        'Conv2d': torch.nn.Conv2d,
        'Linear': torch.nn.Linear
    }
    
    return layer_dict[layer_str]




class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
"""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device=device)
pruner = Pruner(model, "configs/example.yml", False)
pruner.prune_model()
print(dict(model.named_buffers()).keys())
print(dict(pruner.model.named_buffers()).keys())"""