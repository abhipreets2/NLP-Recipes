import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.utils.parametrize as parametrize

# We will be training a model on the MNIST dataset and the finetuning on a particular digit

# Seeding seed
_ = torch.manual_seed(0)

transform =  transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.137,), (0.3081,))])

#Loading the dataset
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=10, shuffle=True)

mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=10, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Creating an expensive nn to classify MNIST digits

class ExpensiveNet(nn.Module):
    def __init__(self, hidden_size_1=1000, hidden_size_2=2000):
        super(ExpensiveNet, self).__init__()
        self.linear1 = nn.Linear(28*28, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x

net = ExpensiveNet().to(device)

# Training the network for only 1 epoch
def train(train_loader, net, epochs=5, total_iteration_limit=None):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    total_iterations = 0

    for epoch in range(epochs):
        net.train()

        loss_sum = 0
        num_iterations = 0

        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        if total_iteration_limit is not None:
            data_iterator.total = total_iteration_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = net(x.view(-1, 28*28))
            loss = loss_fn(output, y)
            loss_sum += loss
            avg_loss = loss_sum/ num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iteration_limit is not None and total_iterations >= total_iteration_limit:
                return

train(train_loader, net, epochs=1)

# Keeping a copy of the original weights
original_weights = {}
for name, param in net.named_parameters():
    original_weights[name] = param.clone().detach()


# Testing the performance of the model (here we will consider, this model as pre-trained model)
def test():
    correct = 0
    total = 0

    wrong_counts = [0 for i in range(10)]

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = net(x.view(-1, 784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                else:
                    wrong_counts[y[idx]] += 1
                total += 1
    print(f"Accuracy: {round(correct/total, 3)}")
    for i in range (len(wrong_counts)):
        print(f"Wrong counts for digit {i}: {wrong_counts[i]}")


test()

# In our case the wrong count for digit 3 is higher, so we will fine tune the model to better detect the 3 digit

print("---------Fine tuning for digit 3---------")


# Let's take a look at the parameters of our model
total_parameters_original = 0
for index, layer in enumerate([net.linear1, net.linear2, net.linear3]):
    total_parameters_original += layer.weight.nelement() + layer.bias.nelement()
    print(f"Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape}")
print(f"Total number of parameters: {total_parameters_original}")


# Defining the LoRA parameterization

class LoRA(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device="cpu"):
        super(LoRA, self).__init__()
        
        # The paper is using random Gaussian initialization for A and zero for B (Section 4.1)
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)

        # There is scaling factor also involved (Section 4.1)
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights


# Adding parameterization to our network

def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1):
    
    features_in, features_out = layer.weight.shape
    return LoRA(features_in, features_out, rank=rank, alpha=lora_alpha, device=device)


parametrize.register_parametrization(
        net.linear1, "weight", linear_layer_parameterization(net.linear1, device)
        )
parametrize.register_parametrization(
        net.linear2, "weight", linear_layer_parameterization(net.linear2, device)
        )
parametrize.register_parametrization(
        net.linear3, "weight", linear_layer_parameterization(net.linear3, device)
        )

def enable_disable_lora(enabled=True):
    for layer in [net.linear1, net.linear2, net.linear3]:
        layer.parametrizations["weight"][0].enabled = enabled


# Display number of parameters added by LoRA
total_parameters_lora = 0
total_parameters_non_lora = 0
for index, layer in enumerate([net.linear1, net.linear2, net.linear3]):
    total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement() + layer.parametrizations["weight"][0].lora_B.nelement()
    total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()
    print(
            f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape} + Lora_A: {layer.parametrizations["weight"][0].lora_A.shape} + Lora_B: {layer.parametrizations["weight"][0].lora_B.shape}')

# Non-LoRA parameter count must match the original network
assert total_parameters_non_lora == total_parameters_original
print(f"Total numbers of parameters (original): {total_parameters_non_lora}")
print(f"Total number of paramters (original + LoRA): {total_parameters_lora + total_parameters_non_lora}")
print(f"Parameters introduced by LoRA: {total_parameters_lora}")
parameters_increment = (total_parameters_lora / total_parameters_non_lora) * 100
print(f"Parameter increment: {parameters_increment:.3f}")

# Freezing the parameters of the original network and fine tuning the ones introduced by LoRA
for name, param in net.named_parameters():
    if 'lora' not in name:
        print(f"Freezing non-LoRA parameter {name}")
        param.requires_grad = False


# Load the MNIST dataset again, by keeping only the digit 3
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
exclude_indices = mnist_trainset.targets == 3
mnist_trainset.data = mnist_trainset.data[exclude_indices]
mnist_trainset.targets = mnist_trainset.targets[exclude_indices]
# Create a dataloader for the training
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

# Train the network with LoRA only on the digit 3 and only for 100 batches (hoping that it would improve the performance on the digit 3)
train(train_loader, net, epochs=1, total_iteration_limit=100)

# Check that the frozen parameters are still unchanged by the finetuning
assert torch.all(net.linear1.parametrizations.weight.original == original_weights['linear1.weight'])
assert torch.all(net.linear2.parametrizations.weight.original == original_weights['linear2.weight'])
assert torch.all(net.linear3.parametrizations.weight.original == original_weights['linear3.weight'])

enable_disable_lora(enabled=True)
# The new linear1.weight is obtained by the "forward" function of our LoRA parametrization
# The original weights have been moved to net.linear1.parametrizations.weight.original
# More info here: https://pytorch.org/tutorials/intermediate/parametrizations.html#inspecting-a-parametrized-module
assert torch.equal(net.linear1.weight, net.linear1.parametrizations.weight.original + (net.linear1.parametrizations.weight[0].lora_B @ net.linear1.parametrizations.weight[0].lora_A) * net.linear1.parametrizations.weight[0].scale)

enable_disable_lora(enabled=False)
# If we disable LoRA, the linear1.weight is the original one
assert torch.equal(net.linear1.weight, original_weights['linear1.weight'])


print("Test with LoRA enabled")
enable_disable_lora(enabled=True)
test()

print(" Test with LoRA disabled")
enable_disable_lora(enabled=False)
test()

