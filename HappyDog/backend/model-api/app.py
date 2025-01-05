import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image, ImageFile
import base64
import io
from flask import Flask, request, jsonify

#region Global

ImageFile.LOAD_TRUNCATED_IMAGES = True
dataset_dir = "./shared-data/Images"
use_cuda = torch.cuda.is_available()
transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
#endregion Global

# Initialize Flask app
app = Flask(__name__)

#region Model setup

#region Loading Class names (labels)

# Load class names from CSV file
class_names = np.genfromtxt('./shared-data/breed_names.csv', delimiter=',', dtype=None, names=True, encoding='utf-8')

#endregion Loading Class names (i.e., labels)

#region Loading and setting up model

# Importing the ResNet-50 model from torchvision.models
model = models.resnet50(pretrained=True)

# Freezing all layers of the ResNet-50 model (i.e., freezing the weights)
for param in model.parameters():
    param.requires_grad = False

# Replacing the fully connected (fc) layer of ResNet-50 with a new Linear layer
# The original ResNet-50 output layer has 2048 features (for ResNet-50).
# We change it to output 133 features (which is typical for a classification task with 133 classes).
# `bias=True` means that the layer will include a bias term.
model.fc = nn.Linear(2048, 133, bias=True)

# Accessing the parameters of the new fully connected (fc) layer
fc_parameters = model.fc.parameters()

# Setting requires_grad = True for the parameters of the newly added fc layer.
# This means the weights of the new fully connected layer will be updated during backpropagation (i.e., they will learn).
for param in fc_parameters:
    param.requires_grad = True

# Load the trained model's state_dict (weights and biases) from a file
# The 'resnet_transfer.pt' file contains the saved state_dict of a previously trained model
model.load_state_dict(torch.load('./shared-data/resnet_transfer.pt', map_location='cpu'))

# Set the model to evaluation mode (i.e., it is not training anymore)
model.eval()

#endregion Loading and setting up model

#endregion Model setup

#region Functions

# Function to predict the breed from an image
def predict_breed_from_image(image, lang):
    img_tensor = transform(image)[:3,:,:].unsqueeze(0)
    if use_cuda:
        img_tensor = img_tensor.cuda()

    output = model(img_tensor)
    _,preds_tensor = torch.max(output,1)

    return (class_names[lang])[preds_tensor]

#endregion

#region Routes

# Route to predict the breed
@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests."""
    try:
        # Parse the base64 image from the request
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'Image not provided'}), 400
        
        base64_image = data['image']
        
        # Split js base64 image
        if base64_image.startswith("data:image"):
            base64_image = base64_image.split(",")[1]

        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Predict the breed
        breed = predict_breed_from_image(image, data['lang'])
        return jsonify({'predicted_breed': breed}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to check if service is running
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

#endregion Routes

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)