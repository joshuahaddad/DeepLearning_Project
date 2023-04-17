import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import yaml
import copy
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    def prune_model(self, retrain=False, **kwargs):
        
        # print("Validating Pre-Pruning")
        # self.validate_model()
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
                        
                        if 'kwargs' in self.config.keys():
                            method(module, name=params[0], amount=params[1], **self.config['kwargs'])
                        else:
                            method(module, name=params[0], amount=params[1])
        
        if retrain:
            self.train_model(**kwargs)
        
        return self.model

    
    # The training and validation procedures here draw heavily from these two resources:
    # https://medium.com/@buiminhhien2k/solving-cifar10-dataset-with-vgg16-pre-trained-architect-using-pytorch-validation-accuracy-over-3f9596942861
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def validate_model(self, testset=None):
        if not testset:
            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
                                        ])
            testset = torchvision.datasets.CIFAR100(root="./data", download=False, transform=transform, train=False)
            
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=256,
                                                 shuffle=True)
        
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            loop = tqdm.tqdm(testloader)
            for data in loop:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                predicted = torch.argmax(outputs.data, 1)
                loss = criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loop.set_description(f"")
                loop.set_postfix(loss=loss, acc=correct/total)
                
        print(f'Accuracy of the network on the CIFAR100 test images: {100 * correct // total} %')
    
    def train_model(self, opt_args=None, trainset=None, criterion=nn.CrossEntropyLoss(), **kwargs):
        """print("Validating Pre Retrain")
        self.validate_model()"""
        if not trainset:
            transform = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                                            transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
                                        ])
            trainset = torchvision.datasets.CIFAR100(root="./data", download=False, transform=transform, train=True)
            testset = torchvision.datasets.CIFAR100(root="./data", download=False, transform=transform, train=False)
        
        if not opt_args:
            opt_args={'lr': 0.001, 'momentum': 0.9, 'epochs': 5, 'batch_size': 128}
            
        trainloader = torch.utils.data.DataLoader(trainset,
                                                 batch_size=opt_args['batch_size'],
                                                 shuffle=True)
        
        
        
        
        optim = torch.optim.Adam(self.model.parameters(), lr = opt_args['lr'])
        running_loss = 0.0
        n_total_step = len(trainloader)
        NUM_EPOCHS = opt_args['epochs']
        
        for epoch in range(NUM_EPOCHS):
            loop = tqdm.tqdm(trainloader)
            for i, data in enumerate(loop):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optim.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optim.step()
                running_loss += loss.item()
                loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
                loop.set_postfix(loss=loss)
        self.validate_model()
        
            

# Get the parameters to prune in the form of a tuple of tuples with (module, 'parameter') such as (model.conv1, 'weight')
def get_global_params(model, config):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if name in config['global_pruning']['layers']:
            parameters_to_prune.append((module, 'weight'))
    return tuple(parameters_to_prune)

def get_global_method(config):
    method_dict = {
        'L1Unstructured': prune.L1Unstructured,
        'LnStructured': prune.LnStructured
    }
    
    return method_dict[config['method']]

def get_local_method(config):
    method_dict = {
        'L1Unstructured': prune.l1_unstructured,
        'LnStructured': prune.ln_structured
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





def validate_CIFAR100_models():
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
    model = model.to(device)
    model.load_state_dict(torch.load("models/cifar100_vgg16_bn-7d8c4031.pt"))
    pruner = Pruner(model, "configs/example.yml", False)
    pruner.validate_model()
    
# validate_CIFAR100_models()
"""pruner.prune_model(retrain=True)
print(dict(model.named_buffers()).keys())
print(dict(pruner.model.named_buffers()).keys())"""