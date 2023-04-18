import yaml
import torch
import prune
from gradcam_visualization import GradCAM
from  saliency_visualization import SaliencyMap
from class_visualization import ClassVisualization
import imageio
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_yamls(config_prefix):
    with open("./configs/base.yml") as f:
        config = yaml.safe_load(f)
    
    config['model'] = "VGG16"
    for i in range(101):
        for k, v in config['local_pruning'].items():
            config['local_pruning'][k] = [v[0], i/100]

        with open(f"{config_prefix}/{i}.yaml", 'w') as f:
            documents = yaml.dump(config, f)

# Generates 10 images showing the gradcam progression as the layers are pruned
def run_gradcams(model, config_prefix, folder, retrain=False):
    for i in range(11):  
        gcv = GradCAM()
        gcv.run_program(model, prune_config=f"{config_prefix}/{i*10}.yaml", suffix=i*10, folder=folder, retrain=retrain)
    
    generate_gifs(folder, "gradcam")


def run_saliency(model, config_prefix, folder, retrain=False):
    for i in range(11):  
        sv = SaliencyMap()
        sv.run_program(model, prune_config=f"{config_prefix}/{i*10}.yaml", suffix=i*10, folder=folder, retrain=retrain)
    generate_gifs(folder, "saliency")

def run_class_viz(model, config_prefix, folder, retrain=False):
    for i in range(11):  
        cv = ClassVisualization()
        cv.run_program(model, prune_config=f"{config_prefix}/{i*10}.yaml", suffix=i*10, folder=folder, retrain=retrain)
        return
    return

def generate_gifs(folder, viz_name):
    filenames = [f"{10*i}.png" for i in range(11)]
    ims = [imageio.imread(f"./visualization/{viz_name}_viz/{folder}/{f}") for f in filenames]
    imageio.mimwrite(f"./visualization/gifs/{folder}/{viz_name}.gif", ims, duration=0.5)
    
if __name__ == '__main__':
    prefix = "./configs/local/l1_unstructured"
    folder = "l1_unstructured/non-retrained"
    # generate_yamls(prefix)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True)
    model = model.to(device)
    
    #run_gradcams(model, prefix, folder, retrain=False)
    # run_saliency(model, prefix, folder)
    # generate_gifs(folder, "saliency")
    run_class_viz(model, prefix, folder, retrain=False)