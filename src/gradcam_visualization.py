import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from helpers.data_utils import *
from helpers.image_utils import *
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from prune import *

class GradCAM():
    
    def __init__(self):
        pass

    def gradcam(self, model, prune_config=None, global_prune=False, retrain=False):

        # Define a preprocessing function to resize the image and normalize its pixels
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))
        ])

        # Check for GPU support
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not model:
            model = models.squeezenet1_1(weights='SqueezeNet1_1_Weights.DEFAULT').to(device)
        
        if prune_config:
            pruner = Pruner(model, prune_config, global_prune)
            model = pruner.prune_model(retrain=retrain, opt_args={'lr': 0.0005, 'momentum': 0.9, 'epochs': 5, 'batch_size': 40})

        ##############################################################################
        # TODO: Define a hook function to get the feature maps from the last         #
        # convolutional layer. Then register the hook to get the feature maps        #
        ##############################################################################
        
        last_conv = model.features[-2]
        def fw_hook(module, input, output):
            self.out_conv = output
        
        last_conv.register_forward_hook(fw_hook)
        
        def bw_hook(module, grad_in, grad_out):
            self.grad_conv = grad_out[0]
        last_conv.register_backward_hook(bw_hook)
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
        gradcams = []
        for x in self.X:
            # Load an input image and perform pre-processing
            image = Image.fromarray(x)
            x = preprocess(image).unsqueeze(0).to(device)

            # Make a forward pass through the model
            logits = model(x)
            

            # Get the class with the highest probability
            class_idx = torch.argmax(logits)

            ###############################################################################
            # TODO: To generate a Grad-CAM heatmap, first compute the gradients of the    #
            # output class with respect to the feature maps. Then, calculate the weights  #
            # of the feature maps. Using these weights, compute the Grad-CAM heatmap.     #
            # Use the cv2 for resizing the heatmap if necessary.                          #
            ###############################################################################
            
            logits[0,class_idx].backward()
            alpha = torch.mean(self.grad_conv, dim=(2,3))

            L = self.out_conv * alpha[:,:, None, None]
            with torch.no_grad():
                heatmap = torch.sum(L, dim=(0,1))
                heatmap = cv2.resize(heatmap.cpu().numpy(), (32,32))


            ##############################################################################
            #                             END OF YOUR CODE                               #
            ##############################################################################

            # store gradcams
            gradcams.append([x.squeeze(0).permute(1,2,0).cpu().numpy(), heatmap])

        return gradcams

    def run_program(self, model, prune_config=None, global_prune=False, retrain=False, suffix="Full", verbose=False, folder="l1_unstructured"):
        # Retrieve images
        np.random.seed(40)
        
        X, y, labels, class_names = load_images(num=5, deterministic=True)
        self.X = X
        gradcams = self.gradcam(model, prune_config=prune_config, global_prune=global_prune, retrain=retrain)
        
        # Create a figure and a subplot with 2 rows and 4 columns
        fig, ax = plt.subplots(2, 5, figsize=(12, 6))
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.2, hspace=0.2)
        fig.suptitle(f"Sparsity Level: {suffix}%")

        # Loop over the subplots and plot an image in each one
        for i in tqdm.tqdm(range(2), desc="Creating plots", leave=True):
            for j in tqdm.tqdm(range(5), desc="Processing image", leave=True):
                # Load image
                if i == 0:
                    item = gradcams[j]
                    image = item[0].clip(0,1)
                    ax[i, j].imshow(image, alpha=.87, vmin=.5)
                    ax[i, j].axis('off')
                elif i == 1:
                    item = gradcams[j]
                    image = item[0].clip(0,1)
                    overlay = item[1]

                    # Plot the image in the current subplot
                    ax[i, j].imshow(image, alpha=1, vmin=100.5, cmap='twilight_shifted')
                    ax[i, j].imshow(overlay, cmap='viridis', alpha=0.779)
                    ax[i, j].axis('off')

                # Add a label above each image in the bottom row
                if i == 1:
                    ax[i, j].set_title(labels[j].title(), fontsize=12, y=1.2)

        # Save and display the subplots
        plt.savefig(f"./visualization/gradcam_viz/{folder}/{suffix}.png")
        plt.close()
        if verbose:
            plt.show()
    
if __name__ == '__main__':

    # Retrieve images
    X, y, labels, class_names = load_images(num=5, deterministic=True)
    gc = GradCAM(X)
    gradcams = gc.gradcam()
    # Create a figure and a subplot with 2 rows and 4 columns
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.2, hspace=0.2)

    # Loop over the subplots and plot an image in each one
    for i in tqdm(range(2), desc="Creating plots", leave=True):
        for j in tqdm(range(5), desc="Processing image", leave=True):
            # Load image
            if i == 0:
                item = gradcams[j]
                image = item[0].clip(0,1)
                ax[i, j].imshow(image, alpha=.87, vmin=.5)
                ax[i, j].axis('off')
            elif i == 1:
                item = gradcams[j]
                image = item[0].clip(0,1)
                overlay = item[1]

                # Plot the image in the current subplot
                ax[i, j].imshow(image, alpha=1, vmin=100.5, cmap='twilight_shifted')
                ax[i, j].imshow(overlay, cmap='viridis', alpha=0.779)
                ax[i, j].axis('off')

            # Add a label above each image in the bottom row
            if i == 1:
                ax[i, j].set_title(labels[j].title(), fontsize=12, y=1.2)

    # Save and display the subplots
    plt.savefig("./visualization/gradcam_visualization.png")
    plt.show()