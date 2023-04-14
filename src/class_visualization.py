import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from helpers.image_utils import preprocess, deprocess, SQUEEZENET_MEAN, SQUEEZENET_STD
from scipy.ndimage import gaussian_filter1d
import random
import numpy as np
import torch
from helpers.data_utils import load_images
import torchvision.models as models
from tqdm import tqdm

class ClassVisualization:

    @staticmethod
    def jitter(X, ox, oy):
        """
        Helper function to randomly jitter an image.

        Inputs
        - X: PyTorch Tensor of shape (N, C, H, W)
        - ox, oy: Integers giving number of pixels to jitter along W and H axes

        Returns: A new PyTorch Tensor of shape (N, C, H, W)
        """
        if ox != 0:
            left = X[:, :, :, :-ox]
            right = X[:, :, :, -ox:]
            X = torch.cat([right, left], dim=3)
        if oy != 0:
            top = X[:, :, :-oy]
            bottom = X[:, :, -oy:]
            X = torch.cat([bottom, top], dim=2)
        return X

    @staticmethod
    def blur_image(X, sigma=1.0):
        X_np = X.cpu().clone().numpy()
        X_np = gaussian_filter1d(X_np, sigma, axis=2)
        X_np = gaussian_filter1d(X_np, sigma, axis=3)
        X.copy_(torch.Tensor(X_np).type_as(X))
        return X

    def create_class_visualization(self, target_y, class_names, model, **kwargs):
        """
        Generate an image to maximize the score of target_y under a pretrained model.

        Inputs:
        - target_y: Integer in the range [0, 25) giving the index of the class
        - model: A pretrained CNN that will be used to generate the image
        - dtype: Torch datatype to use for computations

        Keyword arguments:
        - l2_reg: Strength of L2 regularization on the image
        - learning_rate: How big of a step to take
        - num_iterations: How many iterations to use
        - blur_every: How often to blur the image as an implicit regularizer
        - max_jitter: How much to gjitter the image as an implicit regularizer
        - show_every: How often to show the intermediate result
        - generate_plots: to plot images or not (used for testing)
        """

        model.eval()

        # model.type(dtype)
        l2_reg = kwargs.pop('l2_reg', 1e-3)
        learning_rate = kwargs.pop('learning_rate', 25)
        num_iterations = kwargs.pop('num_iterations', 100)
        blur_every = kwargs.pop('blur_every', 10)
        max_jitter = kwargs.pop('max_jitter', 16)
        show_every = kwargs.pop('show_every', 25)
        generate_plots = kwargs.pop('generate_plots', True)

        # Randomly initialize the image as a PyTorch Tensor, and also wrap it in
        # a PyTorch Variable.
        img = torch.randn(1, 3, 224, 224).mul_(1.0).to(device)
        img_var = Variable(img, requires_grad=True)

        ########################################################################
        # TODO: Use the model to compute the gradient of the score for the     #
        # class target_y with respect to the pixels of the image, and make a   #
        # gradient step on the image using the learning rate. Don't forget the #
        # L2 regularization term. We use this function in our loop below.      #
        ########################################################################
        
        def compute_gradient(model, img_var):
            model.zero_grad()
            # img_var.retain_grad()
            out = model.forward(img_var)
            score = out[0][target_y]
            l2 = torch.pow(torch.linalg.norm(img_var.flatten(), ord=2), 2)*l2_reg
            reg_score = score - l2
            reg_score.backward()
            grads = img_var.grad
            grads = grads/torch.linalg.norm(grads.flatten(), ord=2)
            with torch.no_grad(): 
                img_var += grads*learning_rate
            pass

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        for t in tqdm(range(num_iterations), desc="Processing image", leave=True):
            # Randomly jitter the image a bit; this gives slightly nicer results
            ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
            img.copy_(self.jitter(img, ox, oy))
            # print('before', img)
            compute_gradient(model, img_var)
            # print('after', img)

            # Undo the random jitter
            img.copy_(self.jitter(img, -ox, -oy))
            

            # As regularizer, clamp and periodically blur the image
            for c in range(3):
                lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
                hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
                img[:, c].clamp_(min=lo, max=hi)
            if t % blur_every == 0:
                self.blur_image(img, sigma=0.5)

            # Periodically show the image
            if generate_plots:
                if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
                    plt.imshow(deprocess(img.clone().detach()))
                    class_name = class_names[target_y]
                    plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
                    plt.gcf().set_size_inches(4, 4)
                    plt.axis('off')
                    plt.savefig('visualization/class_visualization_iter_{}'.format(t + 1), bbox_inches='tight')
        return deprocess(img.cpu())

if __name__ == '__main__':
    # Check for GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.squeezenet1_1(weights='SqueezeNet1_1_Weights.DEFAULT').to(device)
    for param in model.parameters():
        param.requires_grad = False

    cv = ClassVisualization()
    X, y, labels, class_names = load_images(num=5)
    visuals = []
    for target in tqdm(y, desc="Creating class visualization", leave=True):
        out = cv.create_class_visualization(target, class_names, model, generate_plots=False)
        visuals.append(out)

    # Create a figure and a subplot with 2 rows and 4 columns
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.2, hspace=0.2)

    # Loop over the subplots and plot an image in each one
    for i in tqdm(range(2), desc="Creating plots", leave=True):
        for j in range(5):
            # Load image
            if i == 0:
                image = X[j]
            elif i == 1:
                image = visuals[j]

            # Plot the image in the current subplot
            ax[i, j].imshow(image, cmap='bone')
            ax[i, j].axis('off')

            # Add a label above each image in the bottom row
            if i == 1:
                ax[i, j].set_title(labels[j].title(), fontsize=12, y=1.2)

    # Save and display the subplots
    plt.savefig("./visualization/class_visualization.png")
    plt.show()