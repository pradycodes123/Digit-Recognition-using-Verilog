import numpy as np
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os


class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    # magic number is a file identifier - if labels - 2049, if images - 2051
    # image is the actual handwritten digit, label is the corresponding digit class (0-9)
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049: 
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        
        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images.append(img)
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        


input_path = r"C:\Users\Pradyumna S\Projects\Digit recognition using verilog\data\mnist"

training_images_filepath = join(input_path, "train-images.idx3-ubyte")
training_labels_filepath = join(input_path, "train-labels.idx1-ubyte")
test_images_filepath = join(input_path, "t10k-images.idx3-ubyte")
test_labels_filepath = join(input_path, "t10k-labels.idx1-ubyte")

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for image, title_text in zip(images, title_texts):        
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
        
mnist_dataloader = MnistDataloader(training_images_filepath,
                                   training_labels_filepath,
                                   test_images_filepath,
                                   test_labels_filepath)

(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


# Convert to NumPy
x_train = np.array(x_train, dtype=np.float32)
x_test  = np.array(x_test, dtype=np.float32)

y_train = np.array(y_train, dtype=np.int64)
y_test  = np.array(y_test, dtype=np.int64)

# Downsample 28x28 -> 14x14 using 2x2 average pooling
def downsample_2x2(images):
    images = images.reshape(-1, 14, 2, 14, 2)
    images = images.mean(axis=(2, 4))
    return images

x_train = downsample_2x2(x_train)
x_test  = downsample_2x2(x_test)

# Normalize to [0,1]
x_train = x_train / 255.0
x_test  = x_test / 255.0

# Flatten to 196 features
x_train = x_train.reshape(-1, 196)
x_test  = x_test.reshape(-1, 196)

images_2_show = []
titles_2_show = []

for i in range(0, 10):
    r = random.randint(0, 59999)
    images_2_show.append(x_train[r].reshape(14, 14))
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(0, 9999)
    images_2_show.append(x_test[r].reshape(14, 14))
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

show_images(images_2_show, titles_2_show)
print(x_train.shape)
print(y_train.shape)



x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

x_test_t = torch.tensor(x_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)


train_dataset = TensorDataset(x_train_t, y_train_t)
test_dataset = TensorDataset(x_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(196, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

''' 
did not add BatchNorm and Dropout cuz small network - less performance
BatcNorm - normalizes data at each layer to make training more fast and stable  
dropout - randomly shut off a percentage of random neurons to avoid overfitting
'''
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # to update weights to reduce losses


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) 
#step lr - reduces learning rate for every 5 epochs by gamma

epochs = 10

if os.path.exists("mnist_fixed_point.pth"):
    print("Loading saved model...")
    model.load_state_dict(torch.load("mnist_fixed_point.pth"))
    model.eval()

else:
    print("Training model...")

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()  # clears out calcs
            outputs = model(images) #model making the guess
            loss = criterion(outputs, labels) 
            loss.backward() # backward prop
            optimizer.step() # update weights
            running_loss += loss.item()
        
        scheduler.step()  # update lr once every 5 epochs
        print(f"Epoch {epoch+1}, Loss = {running_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), "mnist_fixed_point.pth")


correct = 0
total = 0

with torch.no_grad(): # dont calc gradients 

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy =", 100 * correct / total, "%")


w1 = model.fc1.weight.data.numpy()
b1 = model.fc1.bias.data.numpy()


w2 = model.fc2.weight.data.numpy()
b2 = model.fc2.bias.data.numpy()

w3 = model.fc3.weight.data.numpy()
b3 = model.fc3.bias.data.numpy()

print(w1.shape, b1.shape)
print(w2.shape, b2.shape)
print(w3.shape, b3.shape)

scale = 256
# q8.8 quantization 
# in verilog sum +=  (a*B) 
# final = sum >> 8

def quantize(arr):
    return np.clip(arr * scale, -32768, 32767).astype(np.int16)

w1_q = quantize(w1)
b1_q = quantize(b1)

w2_q = quantize(w2)
b2_q = quantize(b2)

w3_q = quantize(w3)
b3_q = quantize(b3)

np.savetxt("w1.mem", w1_q.flatten() & 0xFFFF, fmt="%04x")
np.savetxt("b1.mem", b1_q.flatten() & 0xFFFF, fmt="%04x")

np.savetxt("w2.mem", w2_q.flatten() & 0xFFFF, fmt="%04x")
np.savetxt("b2.mem", b2_q.flatten() & 0xFFFF, fmt="%04x")

np.savetxt("w3.mem", w3_q.flatten() & 0xFFFF, fmt="%04x")
np.savetxt("b3.mem", b3_q.flatten() & 0xFFFF, fmt="%04x")