# Transfer-Learning-Project
This was the final project for the Deep Learning 2023 course provided by the University of Oulu.
This README file provides instructions on how to run the provided PyTorch code for training and testing a ResNet18 model on the MiniImageNet dataset. The code includes data loading, model definition, training, and testing procedures.

## Prerequisites
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- Pillow
- torchsummary
- Google Colab (optional, for mounting Google Drive)
Install the required packages using the following command:
```bash
  !pip install torch torchvision scikit-learn matplotlib Pillow torchsummary
```
## Data Preparation
1. Download the MiniImageNet dataset and extract it to the specified directory. Uncomment and run the following lines in the code:
```bash
  #!unzip "/content/drive/MyDrive/DeepLearning_Project/MiniImage/train.rar" -d "/content/drive/MyDrive/DeepLearning_Project/MiniImage"
```
2. If you encounter any issues with the Google Colab GPU, modify the device settings in the code:
```bash
  #add GPU code
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```
## Training

1. Run the entire code, and the training process will begin.
2. The training progress, including epoch-wise losses, will be displayed.
3. Early stopping will be applied if the validation loss does not improve for a specified number of epochs.
4. Once training is complete, the trained model will be saved to '/content/drive/MyDrive/DL23/pretrained_model_miniimagenet_extended.pth'.

## Testing

1. The testing phase will automatically follow the training.
2. The test loss and accuracy will be displayed.
3. The final trained model is saved in the specified location.

## Customization

- Adjust the hyperparameters such as learning rate, batch size, and number of epochs according to your requirements.
- Modify the model architecture by uncommenting and adjusting the layers in the **CustomResNet18Extended** class.
- Change the early stopping criteria and other training parameters as needed.

  # EuroSAT Dataset Experiments

## Introduction
This code performs experiments on the EuroSAT dataset using a custom ResNet18-based model. The dataset is loaded, and several subsets are randomly selected for training and testing. The model is trained on each subset, and the performance is evaluated.

## Dataset Preparation
1. **Download the EuroSAT dataset:**
    ```bash
    !unzip "/content/drive/MyDrive/DL23/EuroSAT.zip" -d "/content/drive/MyDrive/DL23"
    ```

2. **Data Transformation:**
    - Resize images to (224, 224) pixels.
    - Convert images to PyTorch tensors.

## Experiment 1
### Subset Selection
Randomly select 5 classes from the EuroSAT dataset.

```python
selected_classes = np.random.choice(classes, 5, replace=False)
```

### Dataset Creation

Create a custom dataset with the selected classes.
```python
selected_dataset = CustomDataset(root=dataset_root, image_list=selected_images, transform=eurosat_transform)
```

### Dataset Statistics
Print the number of classes and images per class.
```python
class_count = defaultdict(int)
for _, class_name in selected_dataset:
    class_count[class_name] += 1

print(f"Number of classes: {len(class_count)}")
print("Number of images per class:")
for class_name, count in class_count.items():
    print(f"{class_name}: {count} images")
```
### Model Training

Train a custom ResNet18 model on the selected dataset.

```python
pretrained_model_path = '/content/drive/MyDrive/DL23/pretrained_model_miniimagenet_extended.pth'
model_schmodel = CustomResNet18Extended(num_classes=64)
model_schmodel.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))
model_schmodel.eval()
print(model_schmodel)
num_classes_new_dataset = 5
model_schmodel.fc = nn.Linear(model_schmodel.fc.in_features, num_classes_new_dataset)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_schmodel.parameters(), lr=0.001)
training_losses = []


num_epochs = 100
for epoch in range(num_epochs):
    model_schmodel.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model_schmodel(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    training_losses.append(epoch_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}") # avg training loss for this epoch
plt.plot(range(1, num_epochs + 1), training_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

### Model Evaluation

Evaluate the trained model on a test set.

```python
model_schmodel.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model_schmodel(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%") # print the accuracy on the test set
```

### Save Results

Save the trained model and dataset for future use.

```python
torch.save(model_schmodel.state_dict(), '/content/drive/MyDrive/DL23/model_schmodel.pth')
```
### Repeat Experiments 2-5

Repeat the above steps with different subsets and configurations.

## Summary and Comparison

Compare the performance of models trained on different subsets.

## Additional Experiments

Perform additional experiments with different configurations and hyperparameters.

## Conclusion

Considering the notable differences between the photos in the miniImageNet and ImageNet databases and those in the EuroSAT database, our results are deemed satisfactory. Given the limitation of using only 25 photos for the model's training in the project's final phase, the strategic application of transfer learning is responsible for our model's performance, which averages over 60% in test accuracy.
