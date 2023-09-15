# CIFAR-10_CLASSIFICATION_UISNG_CNN
Trainng CNN and Autoencoder on CIFAR10 dataset and comparig the results 
Train CNN with the following details
1) He initialization.

2)Data Augmentation Details:rotate by 10 degrees and gaussian noise.

3)use MaxPool operation.

4)Target Classification Details: target 5-classes 1,3,5,7,9.

5)Feature Extraction layers: network should have 6 conv layers and 1 pool layer and 12 filters in the first layer.

6)network must have 12 filters in the first layer.


## QUESTION 1

To deploy this project run
```bash
transform = transforms.Compose([
    transforms.RandomRotation(degrees=10), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
])

class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

transform.transforms.append(GaussianNoise(mean=0, std=0.1))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

target_classes = [1, 3, 5, 7, 9]
train_data = []
train_targets = []
for i, target in enumerate(train_dataset.targets):
    if target in target_classes:
        train_targets.append(target)
        train_data.append(train_dataset.data[i])
train_dataset.targets = train_targets
train_dataset.data = train_data

test_data = []
test_targets = []
for i, target in enumerate(test_dataset.targets):
    if target in target_classes:
        test_targets.append(target)
        test_data.append(test_dataset.data[i])
test_dataset.targets = test_targets
test_dataset.data = test_data
```


```bash
  class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.fc1 = nn.Linear(in_features=384 * 4 * 4, out_features=512) 
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = nn.functional.relu(self.conv4(x))
        x = self.pool(nn.functional.relu(self.conv5(x)))
        x = self.pool(nn.functional.relu(self.conv6(x)))
        x = x.view(-1, 384 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

Six convolutional layers with varying numbers of input and output channels make up this CNN model. Three-channel input images are fed into the first convolutional layer, which generates 12 output channels. The next four convolutional layers have respective output channels of 24, 48, 96, and 192 and input channels of 12, 24, 48, and 96. The final convolutional layer generates 384 output channels from 192 input channels.
convolutional layer is followed by relu activation funtion and max pooling with stride 2 and kernel_size 2
at the end we will flatten the output and pass this through two fully connected layers.


```bash
train_losses = []
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to('cuda'), data[1].to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            train_losses.append(running_loss / 100)
            running_loss = 0.0
    print(f"Epoch {epoch+1} loss: {train_losses[-1]:.3f}")

# Plot training loss vs epoch
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Epoch')
plt.show()

```
This programme trains the specified model for 10 iterations using the train loader data loader. The training loss at the conclusion of each epoch is stored in the train losses list. The average of the batch losses over all of the batches in an epoch is used to compute the training loss. Based on the calculated loss, the optimizer is used to update the model's parameters. The training loss is reported and a training loss vs. epoch plot is shown using matplotlib at the conclusion of each epoch. The string "Finished Training" is then displayed.
