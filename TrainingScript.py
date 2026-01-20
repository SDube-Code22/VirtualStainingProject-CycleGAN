#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import torch # This is a powerful library for building and training neural networks.
import albumentations as A # This library helps with image augmentations (making small changes to images to help the model learn better).
from albumentations.pytorch import ToTensorV2 # This helps convert images to a format that PyTorch can understand.
import os # This library is used for interacting with the operating system, like working with files and folders.
import random # This library is used for generating random numbers and shuffling things.

# Set the device to use for training (GPU if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # This checks if a GPU is available and sets the 'DEVICE' variable accordingly. GPUs are much faster for training neural networks.

# Define the base paths for the image datasets. These paths point to where your image data is stored.
# Adjust these paths if your data is in a different location, especially if it's in Google Drive.
BASE_DIR_Grey = "/users/tmmakena/workspace/Project_Datasets_T/Unstained" # Path to the unstained images.
BASE_DIR_IHC = "/users/tmmakena/workspace/Project_Datasets_T/IHC" # Path to the IHC (Immunohistochemistry) stained images.

# Get lists of all image file paths in the training and testing folders for both unstained and IHC images.
# os.listdir() gets a list of all files in a directory, and os.path.join() combines directory paths and filenames correctly.
all_grey_train_images = [os.path.join(BASE_DIR_Grey, "Train", f) for f in os.listdir(os.path.join(BASE_DIR_Grey, "Train"))] # Get paths for all Grey training images.
all_grey_test_images = [os.path.join(BASE_DIR_Grey, "Test", f) for f in os.listdir(os.path.join(BASE_DIR_Grey, "Test"))] # Get paths for all Grey testing images.
all_ihc_train_images = [os.path.join(BASE_DIR_IHC, "Train", f) for f in os.listdir(os.path.join(BASE_DIR_IHC, "Train"))] # Get paths for all IHC training images.
all_ihc_test_images = [os.path.join(BASE_DIR_IHC, "Test", f) for f in os.listdir(os.path.join(BASE_DIR_IHC, "Test"))] # Get paths for all IHC testing images.

# Define the size of the training and testing subsets of images we want to use.
TRAIN_SIZE = 3896 # We will use this many images for training.
TEST_SIZE = 977 # We will use this many images for testing/validation.

# Randomly shuffle the lists of image paths. This is important to ensure the model doesn't learn the order of the images.
random.shuffle(all_grey_train_images) # Shuffle the Grey training images.
random.shuffle(all_grey_test_images) # Shuffle the Grey testing images.
random.shuffle(all_ihc_train_images) # Shuffle the IHC training images.
random.shuffle(all_ihc_test_images) # Shuffle the IHC testing images.

# Select a subset of the shuffled image paths based on the defined TRAIN_SIZE and TEST_SIZE.
TRAIN_DIR_A_LIST = all_grey_train_images[:TRAIN_SIZE] # Take the first TRAIN_SIZE images from the shuffled Grey training list. This will be our training set A.
TRAIN_DIR_B_LIST = all_ihc_train_images[:TRAIN_SIZE] # Take the first TRAIN_SIZE images from the shuffled IHC training list. This will be our training set B.
VAL_DIR_A_LIST = all_grey_test_images[:TEST_SIZE] # Take the first TEST_SIZE images from the shuffled Grey testing list. This will be our validation set A.
VAL_DIR_B_LIST = all_ihc_test_images[:TEST_SIZE] # Take the first TEST_SIZE images from the shuffled IHC testing list. This will be our validation set B.


# Define various hyperparameters (settings) for the training process.
BATCH_SIZE = 1 # The number of images processed at a time during training. A batch size of 1 means we process one image at a time.
LEARNING_RATE = 1e-5 # How much the model's weights are adjusted during training. A smaller learning rate means smaller adjustments.
LAMBDA_IDENTITY = 0.0 # A parameter for a specific type of loss function (identity loss) used in CycleGAN. 0.0 means this loss is not used in this case.
LAMBDA_CYCLE = 10 # A parameter for another type of loss function (cycle consistency loss) used in CycleGAN. This value (10) means this loss is quite important.
NUM_WORKERS = 2 # How many subprocesses to use for data loading. 0 means the data loading is done in the main process.
NUM_EPOCHS = 200 # How many times the entire dataset is passed forward and backward through the neural network during training.
LOAD_MODEL = False # A flag to indicate whether to load a pre-trained model (False means we start training from scratch).
SAVE_MODEL = True # A flag to indicate whether to save the trained model after training.

# Define the filenames for saving the trained model checkpoints.
CHECKPOINT_GEN_H = "genh.pth.tar" # File name for saving the generator model that transforms images from domain A to domain B.
CHECKPOINT_GEN_Z = "genz.pth.tar" # File name for saving the generator model that transforms images from domain B to domain A.
CHECKPOINT_CRITIC_H = "critich.pth.tar" # File name for saving the discriminator model that evaluates images in domain B.
CHECKPOINT_CRITIC_Z = "criticz.pth.tar" # File name for saving the discriminator model that evaluates images in domain A.


# In[2]:


from PIL import Image # Imports the Python Imaging Library (PIL) for working with images.
import os # Imports the operating system module for interacting with files and directories.
from torch.utils.data import Dataset # Imports the base class for creating custom datasets in PyTorch.
import numpy as np # Imports the NumPy library for numerical operations, especially with arrays.
import albumentations as A # Imports the Albumentations library for image augmentations.
from albumentations.pytorch import ToTensorV2 # Imports a function from Albumentations to convert images to PyTorch tensors.

class CycleGANDataset(Dataset): # Defines a custom dataset class named CycleGANDataset that inherits from PyTorch's Dataset class.
    def __init__(self, image_paths_a, image_paths_b, transform=None): # Constructor for the dataset class.
        self.image_paths_a = image_paths_a # Stores the list of file paths for images in domain A.
        self.image_paths_b = image_paths_b # Stores the list of file paths for images in domain B.
        self.transform = transform # Stores the image transformation object (e.g., for resizing, normalization, augmentation).

        self.length_dataset = max(len(self.image_paths_a), len(self.image_paths_b)) # Calculates the maximum length between the two image lists. This is used to determine the dataset's overall length.
        self.a_len = len(self.image_paths_a) # Stores the length of the image list for domain A.
        self.b_len = len(self.image_paths_b) # Stores the length of the domain B image list.


    def __len__(self): # Defines the method to get the size of the dataset.
        return self.length_dataset # Returns the maximum length calculated in the constructor.

    def __getitem__(self, index): # Defines the method to retrieve an item (a pair of images) from the dataset at a given index.
        a_path = self.image_paths_a[index % self.a_len] # Gets the file path for an image in domain A, using the modulo operator to cycle through the list if the index exceeds the list length.
        b_path = self.image_paths_b[index % self.b_len] # Gets the file path for an image in domain B, using the modulo operator to cycle through the list if the index exceeds the list length.

        a_img = np.array(Image.open(a_path).convert("RGB")) # Opens the image file from domain A, converts it to RGB format, and then to a NumPy array.
        b_img = np.array(Image.open(b_path).convert("RGB")) # Opens the image file from domain B, converts it to RGB format, and then to a NumPy array.

        if self.transform: # Checks if a transform object was provided during initialization.
            augmented = self.transform(image=a_img, image0=b_img) # Applies the defined transformations to both images. 'image' is the default key for the first image, and 'image0' is used for the second image as specified in the transform's additional_targets.
            a_img = augmented["image"] # Retrieves the transformed image A from the augmented dictionary.
            b_img = augmented["image0"] # Retrieves the transformed image B from the augmented dictionary.

        return a_img, b_img # Returns the pair of processed images (image from domain A and image from domain B).


# # Defining the dataset class
# 
# 
# 
# *   Creating a dataset class called "CycleGANdataset"
# *   Stores all information about the file paths and the length of the dataset
# *   Images are converted into RGB format and read as pixel values
# *   Image Transformations- resizing, flipped them, normalized the pizel values and turned the images into tensors
# *   Saved those images
# 
# 
# 
# 
# 

# In[3]:


from torch.utils.data import DataLoader # Imports the DataLoader class for creating data loaders, which help in efficiently loading data in batches during training.
import os # Imports the operating system module.
from PIL import Image # Imports the Image module from PIL for image processing.
import numpy as np # Imports the NumPy library for numerical operations.
import albumentations as A # Imports the Albumentations library for image augmentations.
from albumentations.pytorch import ToTensorV2 # Imports a function to convert images to PyTorch tensors.
from torch.utils.data import Dataset # Import Dataset here (Imports the base class for creating custom datasets).

class CycleGANDataset(Dataset): # Defines the custom dataset class again (this seems to be a duplicate definition - you might want to remove one).
    def __init__(self, image_paths_a, image_paths_b, transform=None): # Constructor for the dataset.
        self.image_paths_a = image_paths_a # Stores file paths for domain A images.
        self.image_paths_b = image_paths_b # Stores file paths for domain B images.
        self.transform = transform # Stores the image transformation object.

        self.length_dataset = max(len(self.image_paths_a), len(self.image_paths_b)) # Calculates the maximum length of the two image lists.
        self.a_len = len(self.image_paths_a) # Stores the length of the domain A image list.
        self.b_len = len(self.image_paths_b) # Stores the length of the domain B image list.


    def __len__(self): # Defines the method to get the dataset size.
        return self.length_dataset # Returns the maximum length.

    def __getitem__(self, index): # Defines the method to get an item from the dataset.
        a_path = self.image_paths_a[index % self.a_len] # Gets image path for domain A, cycling if needed.
        b_path = self.image_paths_b[index % self.b_len] # Gets image path for domain B, cycling if needed.

        a_img = np.array(Image.open(a_path).convert("RGB")) # Loads, converts to RGB, and makes NumPy array for image A.
        b_img = np.array(Image.open(b_path).convert("RGB")) # Loads, converts to RGB, and makes NumPy array for image B.

        if self.transform: # Checks if transformations are defined.
            augmented = self.transform(image=a_img, image0=b_img) # Applies transformations.
            a_img = augmented["image"] # Gets transformed image A.
            b_img = augmented["image0"] # Gets transformed image B.

        return a_img, b_img # Returns the transformed image pair.

# Augmentations (copied from config cell for self-containment)
# This section defines the image transformations to be applied.
transforms = A.Compose( # Creates a sequence of transformations.
    [
        A.Resize(width=1024, height=1024), # Resizes images to 256x256 pixels.
        A.HorizontalFlip(p=0.5), # Randomly flips images horizontally with 50% probability.
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255), # Normalizes pixel values.
        ToTensorV2(), # Converts images to PyTorch tensors.
    ],
    additional_targets={"image0": "image"}, # Specifies that the transformations should also be applied to 'image0'.
)


# Use the lists of selected image paths
# This section creates instances of the dataset and data loaders for training and validation.
train_dataset = CycleGANDataset( # Creates the training dataset.
    image_paths_a=TRAIN_DIR_A_LIST, # Uses the list of training image paths for domain A.
    image_paths_b=TRAIN_DIR_B_LIST, # Uses the list of training image paths for domain B.
    transform=transforms # Applies the defined transformations.
)

val_dataset = CycleGANDataset( # Creates the validation dataset.
    image_paths_a=VAL_DIR_A_LIST, # Uses the list of validation image paths for domain A.
    image_paths_b=VAL_DIR_B_LIST, # Uses the list of validation image paths for domain B.
    transform=transforms # Applies the defined transformations.
)

train_loader = DataLoader( # Creates the data loader for the training dataset.
    train_dataset, # Specifies the training dataset.
    batch_size=BATCH_SIZE, # Sets the batch size for training.
    shuffle=True, # Shuffles the data for each epoch.
    num_workers=NUM_WORKERS, # Sets the number of subprocesses for data loading.
    pin_memory= True  # Copies tensors to CUDA pinned memory for faster data transfer to GPU.
)

val_loader = DataLoader( # Creates the data loader for the validation dataset.
    val_dataset, # Specifies the validation dataset.
    batch_size=1,  # validation usually 1 image at a time (Sets the batch size for validation to 1).
    shuffle=False # Does not shuffle validation data.
)

print(f"Using {len(TRAIN_DIR_A_LIST)} images for training in domain A and {len(TRAIN_DIR_B_LIST)} in domain B.") # Prints the number of images used for training in each domain.
print(f"Using {len(VAL_DIR_A_LIST)} images for validation in domain A and {len(VAL_DIR_B_LIST)} in domain B.") # Prints the number of images used for validation in each domain.


# # **Building the Discriminator**
# 
# 
# 
# *   Kernel size of 4, stride 1, reflect padding.
# *   Instance norm to normalize output of convolution layer per batch
# *   Leaky ReLu, makes discriminator less strict to allow G to learn more complex features
# *   Next part is the actual architecture
# *   This function defines the layers and it ensures that the input of one layer is the output of the next layer
# *   stride of 2 for all blocks accept the last one where its 1
# *   At the very end we have a sigmoid function. Real/fake.
# 
# 
# 
# 
# 
# 
# 
# 

# In[4]:


import torch
import torch.nn as nn

# DISCRIMINATOR
# This class defines a basic building block for the Discriminator network.
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        # Initialize the parent class (nn.Module).
        super().__init__()
        # Define a sequential block of layers:
        # 1. Convolutional layer: Reduces spatial dimensions (if stride > 1) and changes the number of channels.
        #    - kernel_size=4: The size of the convolution filter.
        #    - stride: How many pixels the filter moves at each step.
        #    - padding=1: Adds padding to the input to maintain spatial dimensions (especially with stride=1).
        #    - padding_mode="reflect": Specifies how padding is handled at the edges of the image.
        #    - bias=True: Includes a learnable bias term.
        # 2. Instance Normalization: Normalizes the output of the convolutional layer based on individual instances (images) in the batch.
        # 3. Leaky ReLU activation: Introduces non-linearity, allowing the network to learn more complex patterns.
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    # Defines the forward pass of the Block.
    def forward(self, x):
        # Apply the sequential convolutional block to the input tensor x.
        return self.conv(x)


# This class defines the Discriminator network architecture.
class Discriminator(nn.Module):
    # Initialize the Discriminator.
    # - in_channels: Number of input channels (default is 3 for RGB images).
    # - features: A list defining the number of output channels for each convolutional layer in the main part of the network.
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        # Initialize the parent class (nn.Module).
        super().__init__()
        # Define the initial layer of the Discriminator:
        # - Convolutional layer: Processes the initial input image.
        #    - kernel_size=4, stride=2, padding=1: Standard parameters for the first layer in a PatchGAN-like discriminator, reducing spatial dimensions by half.
        # - Leaky ReLU activation: Introduces non-linearity.
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Build the subsequent layers of the Discriminator.
        layers = [] # Initialize an empty list to store the layers.
        in_channels = features[0] # Set the input channels for the first Block to the output channels of the initial layer.
        # Iterate through the features list (excluding the first element).
        for feature in features[1:]:
            # Append a Block to the layers list.
            # The stride is 2 for all blocks except the last one, where it's 1. This gradually reduces the spatial dimensions.
            layers.append(
                Block(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            # Update the input channels for the next Block.
            in_channels = feature
        # Add the final convolutional layer:
        # - Maps the output of the last Block to a single channel, which represents the discriminator's prediction (real or fake).
        # - kernel_size=4, stride=1, padding=1: Standard parameters for the final layer.
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        # Combine all the created layers into a sequential model.
        self.model = nn.Sequential(*layers)

    # Defines the forward pass of the Discriminator network.
    def forward(self, x):
        # Pass the input tensor through the initial layer.
        x = self.initial(x)
        # Pass the output of the initial layer through the subsequent blocks and the final convolutional layer.
        # Apply a sigmoid activation to the final output to produce a value between 0 and 1, representing the probability of the input being real.
        return torch.sigmoid(self.model(x))




# #Defining the Generator
# 
# 
# 
# *   First start with the convolutional2d block (downsampling)
# *   Normalize the output channels
# 
# 
# *   Increase the number of channels to the number of features (feature map)
# *   Reduce the resolution
# 
# 
# *   Then the defining the actual residual blocks
# *   Then upsampling using convtranspose which will increase the resolution and decrease the channels
# *   Summarise forward pass
# 
# 
# 
# 
# 
# 
# 

# In[5]:


import torch
import torch.nn as nn

# This class defines a convolutional block used within the Generator network.
class ConvBlock(nn.Module):
    # Initialize the ConvBlock.
    # - in_channels: Number of input channels.
    # - out_channels: Number of output channels.
    # - down: Boolean indicating if it's a downsampling (Conv2d) or upsampling (ConvTranspose2d) block.
    # - use_act: Boolean indicating whether to use a ReLU activation.
    # - **kwargs: Additional keyword arguments for the convolutional layers.
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        # Define a sequential block:
        # - Convolutional layer: Can be a standard Conv2d (downsampling) or ConvTranspose2d (upsampling).
        #    - padding_mode="reflect": Used for Conv2d to handle padding at edges.
        # - Instance Normalization: Normalizes the output.
        # - Activation: ReLU if use_act is True, otherwise an Identity layer (no activation).
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    # Defines the forward pass of the ConvBlock.
    def forward(self, x):
        return self.conv(x)


# This class defines a Residual Block, a key component in the Generator for learning identity mappings.
class ResidualBlock(nn.Module):
    # Initialize the ResidualBlock.
    # - channels: Number of channels in the input and output.
    def __init__(self, channels):
        super().__init__()
        # Define a sequential block containing two ConvBlocks.
        # The first ConvBlock uses ReLU, the second uses Identity (no activation) before adding the residual connection.
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    # Defines the forward pass of the ResidualBlock.
    def forward(self, x):
        # Add the input tensor 'x' to the output of the 'block' (residual connection).
        return x + self.block(x)


# This class defines the Generator network architecture based on the Pix2PixHD structure.
class Generator(nn.Module):
    # Initialize the Generator.
    # - img_channels: Number of input and output image channels (e.g., 3 for RGB).
    # - num_features: The number of features in the initial convolutional layer.
    # - num_residuals: The number of ResidualBlocks in the middle part of the network.
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        # Define the initial convolutional layer:
        # - Increases the number of channels to num_features.
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        # Define the downsampling blocks:
        # - Uses ConvBlocks to reduce spatial dimensions and increase channels.
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        # Define the Residual Blocks:
        # - A sequence of ResidualBlocks for feature transformation at a lower spatial resolution.
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        # Define the upsampling blocks:
        # - Uses ConvTranspose2d within ConvBlocks to increase spatial dimensions and decrease channels.
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        # Define the final convolutional layer:
        # - Maps the features back to the original number of image channels.
        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    # Defines the forward pass of the Generator network.
    def forward(self, x):
        # Pass the input through the initial layer.
        x = self.initial(x)
        # Pass through the downsampling blocks.
        for layer in self.down_blocks:
            x = layer(x)
        # Pass through the residual blocks.
        x = self.res_blocks(x)
        # Pass through the upsampling blocks.
        for layer in self.up_blocks:
            x = layer(x)
        # Pass through the final layer and apply tanh activation to output values between -1 and 1.
        return torch.tanh(self.last(x))



# # **My main training function + saving of example images/checkpoints**

# In[ ]:


import torch
from torch.utils.data import DataLoader # Import DataLoader here
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

# DISCRIMINATOR Model Definition
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        # Corrected input channels for the first Block in the sequential model
        in_channels_for_block = features[0] # Initialize with the output channels of the initial layer
        for feature in features[1:]:
            layers.append(
                Block(in_channels_for_block, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels_for_block = feature # Update for the next block
        layers.append(
            nn.Conv2d(
                in_channels_for_block, # Use the correct input channels for the final layer
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the initial layer first
        x = self.initial(x)
        # Then pass through the sequential model
        return torch.sigmoid(self.model(x))

# GENERATOR Model Definition
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features, num_features * 2, kernel_size=3, stride=2, padding=1
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(
                    num_features * 4,
                    num_features * 2,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvBlock(
                    num_features * 2,
                    num_features * 1,
                    down=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

# Custom Dataset Definition
class CycleGANDataset(Dataset):
    def __init__(self, image_paths_a, image_paths_b, transform=None):
        self.image_paths_a = image_paths_a
        self.image_paths_b = image_paths_b
        self.transform = transform

        self.length_dataset = max(len(self.image_paths_a), len(self.image_paths_b))
        self.a_len = len(self.image_paths_a)
        self.b_len = len(self.image_paths_b)


    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        a_path = self.image_paths_a[index % self.a_len]
        b_path = self.image_paths_b[index % self.b_len]

        a_img = np.array(Image.open(a_path).convert("RGB"))
        b_img = np.array(Image.open(b_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=a_img, image0=b_img)
            a_img = augmented["image"]
            b_img = augmented["image0"]

        return a_img, b_img


# Checkpoint saving and loading functions
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just continue with the same learning rate
    # as it was when we last saved the model
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# Definitions from other cells (Redundant definitions removed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths (adjust if your data is in Drive) - Using the BASE_DIR_UNSTAINED for grayscale images
BASE_DIR_UNSTAINED = "/users/thatomakena/workspace/Project_Datasets_Thato/Unstained"
BASE_DIR_IHC = "/users/thatomakena/workspace/Project_Datasets_Thato/IHC" # Keep IHC path as is

# Hyperparameters (These should be consistent with the config cell)
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 2 # Consider adjusting based on the warning
NUM_EPOCHS = 200 # Reduced for quicker testing, should be higher for actual training
LOAD_MODEL = False
SAVE_MODEL = True

# Checkpoint filenames
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

# Define the image augmentations to be applied to the images.
# Augmentations help the model generalize better by seeing slightly modified versions of the training images.
transforms = A.Compose(
    [
        A.Resize(width=256, height=256), # Resize all images to a fixed size of 256x256 pixels.
        A.HorizontalFlip(p=0.5), # Randomly flip images horizontally with a 50% probability.
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255), # Normalize the pixel values of the images to a standard range. This is important for neural networks.
        ToTensorV2(), # Convert the images to PyTorch tensors, which is the format PyTorch uses.
    ],
    additional_targets={"image0": "image"}, # This is needed when applying transformations to multiple images simultaneously (like in CycleGAN).
)


def train_fn(
    disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1_loss, mse_loss, d_scaler, g_scaler
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (a_img, b_img) in enumerate(loop):
        a_img = a_img.to(DEVICE)
        b_img = b_img.to(DEVICE)

        # ---------------------
        # Train Discriminators
        # ---------------------
        with torch.cuda.amp.autocast():
            fake_b = gen_H(a_img)
            D_H_real = disc_H(b_img)
            D_H_fake = disc_H(fake_b.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()

            D_H_loss = mse_loss(D_H_real, torch.ones_like(D_H_real)) + \
                       mse_loss(D_H_fake, torch.zeros_like(D_H_fake))

            fake_a = gen_Z(b_img)
            D_Z_real = disc_Z(a_img)
            D_Z_fake = disc_Z(fake_a.detach())
            D_Z_loss = mse_loss(D_Z_real, torch.ones_like(D_Z_real)) + \
                       mse_loss(D_Z_fake, torch.zeros_like(D_Z_fake))

            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # ---------------------
        # Train Generators
        # ---------------------
        with torch.cuda.amp.autocast():
            D_H_fake = disc_H(fake_b)
            D_Z_fake = disc_Z(fake_a)
            loss_G_H = mse_loss(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse_loss(D_Z_fake, torch.ones_like(D_Z_fake))

            # Cycle-consistency loss: Encourages the generated image to be converted back to the original
            cycle_a = gen_Z(fake_b)
            cycle_b = gen_H(fake_a)
            cycle_loss = l1_loss(a_img, cycle_a) * LAMBDA_CYCLE + l1_loss(b_img, cycle_b) * LAMBDA_CYCLE

            # Identity loss: Encourages the generator to not change the image if it's already in the target domain
            identity_a = gen_Z(a_img)
            identity_b = gen_H(b_img)
            identity_loss = l1_loss(a_img, identity_a) * LAMBDA_IDENTITY + l1_loss(b_img, identity_b) * LAMBDA_IDENTITY


            G_loss = loss_G_H + loss_G_Z + cycle_loss + identity_loss

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


# ---------------------
# Main Training Function
# ---------------------

disc_H = Discriminator(in_channels=3).to(DEVICE) # Discriminator for domain H (IHC)
disc_Z = Discriminator(in_channels=3).to(DEVICE) # Discriminator for domain Z (Unstained/Grayscale)
gen_Z = Generator(img_channels=3, num_residuals=9).to(DEVICE) # Generator H to Z (IHC to Unstained)
gen_H = Generator(img_channels=3, num_residuals=9).to(DEVICE) # Generator Z to H (Unstained to IHC)


# Optimizers
opt_disc = optim.Adam(
    list(disc_H.parameters()) + list(disc_Z.parameters()),
    lr=LEARNING_RATE, betas=(0.5, 0.999)
)
opt_gen = optim.Adam(
    list(gen_Z.parameters()) + list(gen_H.parameters()),
    lr=LEARNING_RATE, betas=(0.5, 0.999)
)

# Losses
l1_loss = nn.L1Loss() # L1 loss for cycle consistency and identity loss
mse_loss = nn.MSELoss() # MSE loss for adversarial loss

# Load checkpoints if needed
if LOAD_MODEL:
    # Ensure the checkpoint files exist in the specified folder
    checkpoint_folder = "/users/thatomakena/workspace/SavedCheckpoints"
    load_checkpoint(os.path.join(checkpoint_folder, CHECKPOINT_GEN_H), gen_H, opt_gen, LEARNING_RATE)
    load_checkpoint(os.path.join(checkpoint_folder, CHECKPOINT_GEN_Z), gen_Z, opt_gen, LEARNING_RATE)
    load_checkpoint(os.path.join(checkpoint_folder, CHECKPOINT_CRITIC_H), disc_H, opt_disc, LEARNING_RATE)
    load_checkpoint(os.path.join(checkpoint_folder, CHECKPOINT_CRITIC_Z), disc_Z, opt_disc, LEARNING_RATE)

# Datasets
# Use the lists of selected image paths (TRAIN_DIR_A_LIST, TRAIN_DIR_B_LIST, VAL_DIR_A_LIST, VAL_DIR_B_LIST)
# from the previous cell where they were defined and populated.
# Assuming Domain A is Unstained/Grayscale and Domain B is IHC
train_dataset = CycleGANDataset(
    image_paths_a=TRAIN_DIR_A_LIST, # Unstained training images
    image_paths_b=TRAIN_DIR_B_LIST, # IHC training images
    transform=transforms # Use the transforms defined previously
)
val_dataset = CycleGANDataset(
    image_paths_a=VAL_DIR_A_LIST, # Unstained validation images
    image_paths_b=VAL_DIR_B_LIST, # IHC validation images
    transform=transforms # Use the transforms defined previously
)


# DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, pin_memory=True
)

# GradScaler for mixed precision training (helps prevent gradient explosion with larger models/batches)
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

# ---------------------
# Training Loop
# ---------------------
for epoch in range(NUM_EPOCHS):
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    # Call the train_fn to perform one epoch of training
    # Pass val_loader to train_fn for consistent image saving
    train_fn(disc_H, disc_Z, gen_Z, gen_H, train_loader,
             opt_disc, opt_gen, l1_loss, mse_loss, d_scaler, g_scaler) # Removed epoch argument from train_fn call


    # Save model checkpoints after each epoch if SAVE_MODEL is True
    if SAVE_MODEL and (epoch + 1) % 5 == 0: # Save checkpoints every 5 epochs
        # Create the directory if it doesn't exist
        checkpoint_folder = "/users/thatomakena/workspace/SavedCheckpoints" # Define the folder for saving checkpoints
        os.makedirs(checkpoint_folder, exist_ok=True)
        save_checkpoint(gen_H, opt_gen, filename=os.path.join(checkpoint_folder, f"{CHECKPOINT_GEN_H.split('.')[0]}_epoch{epoch+1}.pth.tar"))
        save_checkpoint(gen_Z, opt_gen, filename=os.path.join(checkpoint_folder, f"{CHECKPOINT_GEN_Z.split('.')[0]}_epoch{epoch+1}.pth.tar")) # Corrected optimizer for gen_Z
        save_checkpoint(disc_H, opt_disc, filename=os.path.join(checkpoint_folder, f"{CHECKPOINT_CRITIC_H.split('.')[0]}_epoch{epoch+1}.pth.tar"))
        save_checkpoint(disc_Z, opt_disc, filename=os.path.join(checkpoint_folder, f"{CHECKPOINT_CRITIC_Z.split('.')[0]}_epoch{epoch+1}.pth.tar"))

    # ---------------------
    # Save sample images after each epoch if it's a saving epoch
    # ---------------------
    output_images_folder = "/users/thatomakena/workspace/ExampleImagesTrain"
    os.makedirs(output_images_folder, exist_ok=True)

    # Save images only at the end of epochs that are multiples of 2 (2, 4, 6...)
    if (epoch + 1) % 2 == 0: # Check if the current epoch is even
         with torch.no_grad(): # Don't track gradients for image saving
            # Get one batch from the validation loader for consistent saving
            # It's better to save from a fixed validation set rather than a random training batch
            # Assuming val_loader is accessible here. If not, pass it to main and then to train_fn.
            try:
                # Get a fixed batch from the validation loader for consistent saving
                # This requires iterating the validation loader or getting a specific batch
                # For simplicity, let's just get the first batch of the validation loader
                # Note: This will iterate the val_loader, which might not be desired
                # if you need to use it elsewhere. A more robust way is to get a fixed batch once.
                # Let's iterate for one batch for now.
                val_iter = iter(val_loader)
                a_val_img, b_val_img = next(val_iter)
                a_val_img = a_val_img.to(DEVICE)
                b_val_img = b_val_img.to(DEVICE)

                fake_b_val = gen_H(a_val_img)
                fake_a_val = gen_Z(b_val_img)

                save_image(fake_b_val * 0.5 + 0.5, os.path.join(output_images_folder, f"b_fake_epoch{epoch+1}.png"))
                save_image(fake_a_val * 0.5 + 0.5, os.path.join(output_images_folder, f"a_fake_epoch{epoch+1}.png"))
                # Optionally save real images for comparison
                save_image(a_val_img * 0.5 + 0.5, os.path.join(output_images_folder, f"a_real_epoch{epoch+1}.png"))
                save_image(b_val_img * 0.5 + 0.5, os.path.join(output_images_folder, f"b_real_epoch{epoch+1}.png"))

            except Exception as e:
                print(f"Could not save validation images for epoch {epoch+1}: {e}")


# In[ ]:




