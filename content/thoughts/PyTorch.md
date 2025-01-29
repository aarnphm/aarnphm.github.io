---
id: PyTorch
tags:
  - ml
  - framework
date: "2024-11-11"
description: tidbits from PyTorch
modified: 2025-01-29 07:47:17 GMT-05:00
title: PyTorch
---

see also: [unstable docs](https://pytorch.org/docs/main/)

## MultiMarginLoss

Creates a criterion that optimizes a multi-class classification hinge loss (margin-based loss) between input $x$
(a 2D mini-batch `Tensor`) and output $y$ (which is a 1D tensor of target class indices, $0 \le y \le \text{x}.\text{size}(1) -1$):

For each mini-batch sample, loss in terms of 1D input $x$ and output $y$ is:

$$
\text{loss}(x,y) = \frac{\sum_{i} \max{0, \text{margin} - x[y] + x[i]}^p}{x.\text{size}(0)}
\\
\because i \in \{0, \ldots x.\text{size}(0)-1\} \text{ and } i \neq y
$$

## SGD

[[thoughts/Nesterov momentum]] is based on [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)

```pseudo
\begin{algorithm}
\caption{SGD in PyTorch}
\begin{algorithmic}
\State \textbf{input:} $\gamma$ (lr), $\theta_0$ (params), $f(\theta)$ (objective), $\lambda$ (weight decay),
\State $\mu$ (momentum), $\tau$ (dampening), nesterov, maximize
\For{$t = 1$ to $...$}
    \State $g_t \gets \nabla_\theta f_t(\theta_{t-1})$
    \If{$\lambda \neq 0$}
        \State $g_t \gets g_t + \lambda\theta_{t-1}$
    \EndIf
    \If{$\mu \neq 0$}
        \If{$t > 1$}
            \State $b_t \gets \mu b_{t-1} + (1-\tau)g_t$
        \Else
            \State $b_t \gets g_t$
        \EndIf
        \If{$\text{nesterov}$}
            \State $g_t \gets g_t + \mu b_t$
        \Else
            \State $g_t \gets b_t$
        \EndIf
    \EndIf
    \If{$\text{maximize}$}
        \State $\theta_t \gets \theta_{t-1} + \gamma g_t$
    \Else
        \State $\theta_t \gets \theta_{t-1} - \gamma g_t$
    \EndIf
\EndFor
\State \textbf{return} $\theta_t$
\end{algorithmic}
\end{algorithm}
```

## [[thoughts/knowledge distillation]]

examples on CIFAR

```python title="distill.py"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Check if the current `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# is available, and if not, use the CPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading the CIFAR-10 dataset:
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

# Deeper neural network class to be used as teacher:
class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Lightweight neural network class to be used as student:
class LightNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

if __name__ == "__main__":
    torch.manual_seed(42)
    nn_deep = DeepNN(num_classes=10).to(device)
    train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
    test_accuracy_deep = test(nn_deep, test_loader, device)

    # Instantiate the lightweight network:
    torch.manual_seed(42)
    nn_light = LightNN(num_classes=10).to(device)
    print("Norm of 1st layer of nn_light:", torch.norm(nn_light.features[0].weight).item())
    print("Norm of 1st layer of new_nn_light:", torch.norm(new_nn_light.features[0].weight).item())
    train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)
    test_accuracy_light_ce = test(nn_light, test_loader, device)

    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    print(f"Student accuracy: {test_accuracy_light_ce:.2f}%")

    # Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
    train_knowledge_distillation(teacher=nn_deep, student=new_nn_light, train_loader=train_loader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
    test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)

    # Compare the student test accuracy with and without the teacher, after distillation
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
    print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
```

### Cosine loss minimisation run

assumption: the teacher network will have a better internal [[thoughts/representations]] comparing to student's weights. Thus we need to artificially push the students' weight to "mimic" the teachers' weights.

We will apply `CosineEmbeddingLoss` such that students' internal representation would be a permutation of the teacher's:

$$
\text{loss}(x,y) = \begin{cases}
1 - \cos(x_1, x_2), & \text{if } y = 1 \\
\max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
\end{cases}
$$

The updated loops as follow [^internal]:

[^internal]: Naturally, we have to update the hidden representation:

    ```python
    sample_input = torch.randn(128, 3, 32, 32).to(device) # Batch size: 128, Filters: 3, Image size: 32x32
    logits, hidden_representation = modified_nn_light(sample_input)

    print("Student logits shape:", logits.shape) # batch_size x total_classes
    print("Student hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size

    logits, hidden_representation = modified_nn_deep(sample_input)

    print("Teacher logits shape:", logits.shape) # batch_size x total_classes
    print("Teacher hidden representation shape:", hidden_representation.shape) # batch_size x hidden_representation_size
    ```

```python title="modified_deep_cosine.py"

class ModifiedDeepNNCosine(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDeepNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output, 2)
        return x, flattened_conv_output_after_pooling

# Create a similar student class where we return a tuple. We do not apply pooling after flattening.
class ModifiedLightNNCosine(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedLightNNCosine, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        flattened_conv_output = torch.flatten(x, 1)
        x = self.classifier(flattened_conv_output)
        return x, flattened_conv_output

def train_cosine_loss(teacher, student, train_loader, epochs, learning_rate, hidden_rep_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model and keep only the hidden representation
            with torch.no_grad():
                _, teacher_hidden_representation = teacher(inputs)

            # Forward pass with the student model
            student_logits, student_hidden_representation = student(inputs)

            # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(device))

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def test_multiple_outputs(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs) # Disregard the second tensor of the tuple
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    # We do not have to train the modified deep network from scratch of course, we just load its weights from the trained instance
    modified_nn_deep = ModifiedDeepNNCosine(num_classes=10).to(device)
    modified_nn_deep.load_state_dict(nn_deep.state_dict())

    # Once again ensure the norm of the first layer is the same for both networks
    print("Norm of 1st layer for deep_nn:", torch.norm(nn_deep.features[0].weight).item())
    print("Norm of 1st layer for modified_deep_nn:", torch.norm(modified_nn_deep.features[0].weight).item())

    # Initialize a modified lightweight network with the same seed as our other lightweight instances. This will be trained from scratch to examine the effectiveness of cosine loss minimization.
    torch.manual_seed(42)
    modified_nn_light = ModifiedLightNNCosine(num_classes=10).to(device)
    print("Norm of 1st layer:", torch.norm(modified_nn_light.features[0].weight).item())
```
