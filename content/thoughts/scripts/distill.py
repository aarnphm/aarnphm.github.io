import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Check if the current `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# is available, and if not, use the CPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f'Using {device} device')

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
    self.classifier = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, num_classes))

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
    self.classifier = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_classes))

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

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')


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
  print(f'Test Accuracy: {accuracy:.2f}%')
  return accuracy


def train_knowledge_distillation(
  teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device
):
  ce_loss = nn.CrossEntropyLoss()
  optimizer = optim.Adam(student.parameters(), lr=learning_rate)

  teacher.eval()  # Teacher set to evaluation mode
  student.train()  # Student to train mode

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

      # Soften the student logits by applying softmax first and log() second
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

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')


if __name__ == '__main__':
  torch.manual_seed(42)
  nn_deep = DeepNN(num_classes=10).to(device)
  train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
  test_accuracy_deep = test(nn_deep, test_loader, device)

  # Instantiate the lightweight network:
  torch.manual_seed(42)
  nn_light = LightNN(num_classes=10).to(device)
  print('Norm of 1st layer of nn_light:', torch.norm(nn_light.features[0].weight).item())
  print('Norm of 1st layer of new_nn_light:', torch.norm(new_nn_light.features[0].weight).item())
  train(nn_light, train_loader, epochs=10, learning_rate=0.001, device=device)
  test_accuracy_light_ce = test(nn_light, test_loader, device)

  print(f'Teacher accuracy: {test_accuracy_deep:.2f}%')
  print(f'Student accuracy: {test_accuracy_light_ce:.2f}%')

  # Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
  train_knowledge_distillation(
    teacher=nn_deep,
    student=new_nn_light,
    train_loader=train_loader,
    epochs=10,
    learning_rate=0.001,
    T=2,
    soft_target_loss_weight=0.25,
    ce_loss_weight=0.75,
    device=device,
  )
  test_accuracy_light_ce_and_kd = test(new_nn_light, test_loader, device)

  # Compare the student test accuracy with and without the teacher, after distillation
  print(f'Teacher accuracy: {test_accuracy_deep:.2f}%')
  print(f'Student accuracy without teacher: {test_accuracy_light_ce:.2f}%')
  print(f'Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%')
