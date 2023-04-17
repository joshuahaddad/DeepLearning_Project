import yaml
import torch
import prune
from gradcam_visualization import GradCAM
import saliency_visualization as sv
import class_visualization as cv
import imageio
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_yamls(config_prefix, folder):
    with open("./configs/base.yml") as f:
        config = yaml.safe_load(f)
    
    config['model'] = "VGG16"
    for i in range(101):
        for k, v in config['local_pruning'].items():
            config['local_pruning'][k] = [v[0], i/100]

        with open(f"{config_prefix}/{i}.yaml", 'w') as f:
            documents = yaml.dump(config, f)

# Generates 100 images showing the gradcam progression as the layers are pruned
def run_gradcams(config_prefix, folder, retrain=False):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
    model = model.to(device)
    data = []
    for i in range(11):  
        gcv = GradCAM()
        gcv.run_program(model, prune_config=f"{config_prefix}/{i*10}.yaml", suffix=i*10, folder=folder, retrain=retrain)
        

def run_saliency():
    return

def run_class_viz():
    return

def generate_gifs(folder):
    filenames = [f"{i}.png" for i in range(11)]
    ims = [imageio.imread(f"./visualization/gradcam_viz/{folder}/{f}") for f in filenames]
    imageio.mimwrite(f"./visualization/gifs/{folder}/gradcam.gif", ims, duration=0.5)
    
if __name__ == '__main__':
    prefix = "./configs/local/l1_unstructured"
    folder = "l1_unstructured/retrained"
    # generate_yamls(prefix, folder)
    run_gradcams(prefix, folder, retrain=True)
    generate_gifs(folder)