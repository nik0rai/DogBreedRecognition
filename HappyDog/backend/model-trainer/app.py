import torch
import numpy as np
from PIL import ImageFile
from torch import nn, optim, cuda
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from flask import Flask, jsonify

# Initialize Flask app
app = Flask(__name__)

#region Global

ImageFile.LOAD_TRUNCATED_IMAGES = True
dataset_dir = "./shared-data/Images"
batch_size = 20
num_workers = 0

use_cuda = cuda.is_available()

# Transformations for the training data, which include augmentation
data_transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224 pixels
    transforms.RandomRotation(45),  # Randomly rotate images by up to 45 degrees
    transforms.RandomVerticalFlip(0.3),  # Randomly flip the image vertically with 30% probability
    transforms.RandomHorizontalFlip(0.3),  # Randomly flip the image horizontally with 30% probability
    transforms.ToTensor(),  # Convert image to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet's mean and std
])

# Transformations for validation and test data (no augmentation, just resizing and normalization)
data_transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224 pixels
    transforms.ToTensor(),  # Convert image to a PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet's mean and std
])

#endregion Global

#region ResNet50 for dogbreeds

#region Step 1: Load the dataset using ImageFolder with the corresponding transformations

# Apply the transformations to the full dataset (with augmentation for training)
dataset = datasets.ImageFolder(dataset_dir, transform=data_transform_train)  # Dataset with augmentation for training

#endregion Step 1

#region Step 2: Split the dataset into train, validation, and test sets (80%, 10%, 10%)

# Determine the size of each subset
train_size = int(0.8 * len(dataset))  # 80% for training
remaining_size = len(dataset) - train_size  # The remaining 20% for validation and test
val_size = int(0.5 * remaining_size)  # Split the remaining 20% equally between validation and test
test_size = remaining_size - val_size  # The remaining part is the test set

# Split the dataset into training, validation, and test sets
train_dataset, temp_dataset = random_split(dataset, [train_size, remaining_size])

# Apply validation and test transformations
val_dataset = datasets.ImageFolder(dataset_dir, transform=data_transform_val_test)
test_dataset = datasets.ImageFolder(dataset_dir, transform=data_transform_val_test)

# Apply the split to the validation and test sets
val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

#endregion Step 2

#region Step 3: Create data loaders for training, validation and test datasets

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
loaders_scratch = { # Store the DataLoaders in a dictionary
    'train': train_loader,
    'valid': val_loader,
    'test': test_loader
}

#endregion Step 3

#region Step 4: Load ResNet-50 model

loaders = loaders_scratch.copy() # Create a copy of the loaders dictionary for use
resnet_model = models.resnet50(pretrained=True) # Load the pre-trained ResNet-50 model

#endregion Step 4

#region Step 5: Setup layers and params of model

# Freeze all the parameters in ResNet-50 (so their weights won't be updated during training)
# This is done for transfer learning, because we need to fine-tune the last layer.
for param in resnet_model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer (fc) of the ResNet-50 model with a new one
# The new layer has 2048 input features (coming from the last convolutional block of ResNet-50)
# and 120 output units (We have 120 breed names i.e., classes)
resnet_model.fc = nn.Linear(2048, 120, bias=True)
fc_parameters = resnet_model.fc.parameters() # Access the parameters of the new fully connected layer

# Enable gradient computation for the parameters of the new fully connected layer
# This means that the weights of the new layer will be updated during backpropagation
# while the other layers (which are frozen) won't be updated.
for param in fc_parameters:
    param.requires_grad = True

# If using CUDA (GPU), move the model to the GPU for faster computation
if use_cuda:
    resnet_model = resnet_model.cuda()

loss_func = nn.CrossEntropyLoss() # Loss function for training

# Define the optimizer (Stochastic Gradient Descent with a learning rate of 0.03)
# We only pass the parameters of the final fully connected layer (resnet_transfer.fc.parameters())
# so only the weights of that layer will be updated during backpropagation.
optimizer = optim.SGD(resnet_model.fc.parameters(), lr=0.03)

#endregion Step 5

#endregion ResNet50 for dogbreeds

#region Functions

# Function to train model
def train(n_epochs, loaders, model, optimizer, loss_func, use_cuda, save_path):
    # Initialize the best validation loss to infinity
    valid_loss_min = np.inf 
    
    # Iterate over the number of epochs
    for epoch in range(1, n_epochs+1):
        # Initialize the training and validation losses to zero at the start of each epoch
        train_loss = 0.0
        valid_loss = 0.0
        
        # Set the model to training mode (activates dropout, batchnorm, etc.)
        model.train()
        
        # Iterate through the training data
        for batch_idx, (data, target) in enumerate(loaders['train']):
            if use_cuda:  # Move data and target to GPU if available
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()  # Clear gradients from the previous step
            output = model(data)   # Perform a forward pass through the model
            loss = loss_func(output, target)  # Calculate the loss (difference between output and target)
            loss.backward()  # Perform backpropagation to calculate gradients
            optimizer.step()  # Update the model's parameters using the optimizer

            # Update the running average of the training loss
            train_loss += (1/(batch_idx+1)) * (loss.data - train_loss)
            
            # Print the training loss for every 100th batch
            if batch_idx % 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' % (epoch, batch_idx + 1, train_loss))
        
        # Set the model to evaluation mode (disables dropout, uses running stats for batchnorm)
        model.eval()
        
        # Iterate through the validation data
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            if use_cuda:  # Move data and target to GPU if available
                data, target = data.cuda(), target.cuda()
            output = model(data)  # Perform a forward pass through the model
            loss = loss_func(output, target)  # Calculate the validation loss
            valid_loss += (1/(batch_idx+1)) * (loss.data - valid_loss)  # Running average of validation loss

        # Print the summary of training and validation losses after each epoch
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        # Check if the current validation loss is the best (lowest so far)
        if valid_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
            torch.save(model.state_dict(), save_path)  # Save the model's state dict (parameters) to the given path
            valid_loss_min = valid_loss  # Update the best validation loss
    
    # Return the trained model after all epochs are completed
    return model

# Function to test model
def test(loaders, model, criterion, use_cuda):
    # Initialize variables to track the test loss and accuracy
    test_loss = 0.0  # To accumulate the total test loss
    correct = 0  # Number of correctly classified samples
    total = 0  # Total number of samples

    # Set the model to evaluation mode (deactivates dropout, uses batchnorm running stats)
    model.eval()

    # Iterate over batches from the test data loader
    for batch_idx, (data, target) in enumerate(loaders['test']):
        if use_cuda:  # If using GPU, move data and target to CUDA device
            data, target = data.cuda(), target.cuda()

        # Perform a forward pass through the model
        output = model(data)
        # Calculate the loss using the provided criterion (e.g., CrossEntropyLoss)
        loss = criterion(output, target)
        # Update the running average of the test loss
        test_loss += (1 / (batch_idx + 1)) * (loss.data - test_loss)
        # Get the predicted class by finding the class with the maximum probability
        pred = output.data.max(1, keepdim=True)[1]
        # Calculate the number of correct predictions in the current batch
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0) # Update the total number of samples processed

    print('Test Loss: {:.6f}\n'.format(test_loss)) # Print the average test loss over the entire test set
 
    # Print the test accuracy as a percentage
    test_ac_text = '\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total)
    print(test_ac_text)
    
    return test_ac_text

#endregion

#region Routes

# Route to start training
@app.route('/train', methods=['GET'])
def train_api():
    train(13, loaders, resnet_model, optimizer, 
        loss_func, use_cuda, './shared-data/resnet_transfer.pt')
    return jsonify({'status': 'Training complete'}), 200

# Route to test
@app.route('/test', methods=['GET'])
def test_api():
    testing_model = resnet_model
    testing_model.load_state_dict(torch.load('./shared-data/resnet_transfer.pt','cpu'))
    return jsonify({'status': test(loaders, testing_model, loss_func, use_cuda)}), 200

#endregion

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)