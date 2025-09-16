# Convolutional Deep Neural Network for Image Classification

## AIM

To develop a Convolutional Deep Neural Network (CNN) model for image classification and to verify the response for new images.

---

## PROBLEM STATEMENT AND DATASET

Image classification is one of the most fundamental problems in computer vision, where the goal is to assign a label or category to an input image. Traditional machine learning models face difficulties in capturing spatial patterns such as edges, textures, and shapes. Convolutional Neural Networks (CNNs) are specifically designed to extract these spatial features using convolutional layers and pooling layers, which makes them highly effective for image classification tasks.

In this project, we design and train a CNN model using the **MNIST dataset** (handwritten digit images).

* The MNIST dataset consists of **60,000 training images** and **10,000 testing images**.
* Each image is a **28 × 28 grayscale digit (0–9)**.
* The task is to correctly classify each image into one of the 10 classes (digits 0–9).

---

## THEORY:

Convolutional Neural Networks (CNNs) are deep learning models widely used for image classification. They automatically learn features like edges, textures, and shapes from images without manual feature extraction.

A CNN mainly consists of:

Convolution Layers – extract features using filters.

ReLU Activation – introduces non-linearity.

Pooling Layers – reduce dimensions while keeping key features.

Fully Connected Layers – perform final classification.

Output Layer – gives class probabilities (e.g., digits 0–9).

CNNs are effective because they reduce parameters, capture spatial patterns, and provide high accuracy in image-related tasks.

## NEURAL NETWORK MODEL

### CNN Model Architecture

![alt text](/Images/image-4.png)

---

## DESIGN STEPS

### STEP 1:

Load the dataset (MNIST) and perform preprocessing such as normalization and reshaping to match the CNN input.

### STEP 2:

Define the CNN architecture with convolutional, pooling, and fully connected layers.

### STEP 3:

Initialize the model, define the loss function (CrossEntropyLoss), and select the optimizer (Adam).

### STEP 4:

Train the model for multiple epochs by feeding batches of training data, calculating loss, and updating weights.

### STEP 5:

Evaluate the model on test data using accuracy, confusion matrix, and classification report.

### STEP 6:

Test the trained model on **new sample images** to verify its classification ability.

---

## PROGRAM

### Name: VENKATANATHAN P R

### Register Number: 212223240173

```python
import torch
import torch.nn as nn
import torch.optim as optim

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Model, Loss Function, Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Name: VENKATANATHAN P R")
        print("Register Number: 212223240173")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
```

---

## OUTPUT

### Training Loss per Epoch

![alt text](/Images/image.png)

### Confusion Matrix

![alt text](/Images/image-2.png)

### Classification Report

![alt text](/Images/image-1.png)

### New Sample Data Prediction

![alt text](/Images/image-3.png)

---

## RESULT

Thus, we successfully developed a **Convolutional Deep Neural Network (CNN)** for image classification using the MNIST dataset. 