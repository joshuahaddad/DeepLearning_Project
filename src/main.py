import yaml
import torch
import prune
from gradcam_visualization import GradCAM
import saliency_visualization as sv
import class_visualization as cv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_yamls():
    with open("./configs/base.yml") as f:
        config = yaml.safe_load(f)
    
    config['model'] = "VGG16"
    for i in range(101):
        for k, v in config['local_pruning'].items():
            config['local_pruning'][k] = [v[0], i/100]

        with open(f"./configs/local/l1_unstructured/all_layers_{i}.yaml", 'w') as f:
            documents = yaml.dump(config, f)

# Generates 100 images showing the gradcam progression as the layers are pruned
def run_gradcams():
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
    model = model.to(device)
    for i in range(101):  
        gcv = GradCAM()
        gcv.run_program(model, prune_config=f"./configs/local/l1_unstructured/all_layers_{i}.yaml", suffix=i)

def run_saliency():
    return

def run_class_viz():
    return

if __name__ == '__main__':
    generate_yamls()
    run_gradcams()