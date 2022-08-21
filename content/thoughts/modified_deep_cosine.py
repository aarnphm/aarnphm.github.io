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
    self.classifier = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, num_classes))

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
    self.classifier = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_classes))

  def forward(self, x):
    x = self.features(x)
    flattened_conv_output = torch.flatten(x, 1)
    x = self.classifier(flattened_conv_output)
    return x, flattened_conv_output


def train_cosine_loss(
  teacher, student, train_loader, epochs, learning_rate, hidden_rep_loss_weight, ce_loss_weight, device
):
  ce_loss = nn.CrossEntropyLoss()
  cosine_loss = nn.CosineEmbeddingLoss()
  optimizer = optim.Adam(student.parameters(), lr=learning_rate)

  teacher.to(device)
  student.to(device)
  teacher.eval()  # Teacher set to evaluation mode
  student.train()  # Student to train mode

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
      hidden_rep_loss = cosine_loss(
        student_hidden_representation, teacher_hidden_representation, target=torch.ones(inputs.size(0)).to(device)
      )

      # Calculate the true label loss
      label_loss = ce_loss(student_logits, labels)

      # Weighted sum of the two losses
      loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')


def test_multiple_outputs(model, test_loader, device):
  model.to(device)
  model.eval()

  correct = 0
  total = 0

  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs, labels = inputs.to(device), labels.to(device)

      outputs, _ = model(inputs)  # Disregard the second tensor of the tuple
      _, predicted = torch.max(outputs.data, 1)

      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f'Test Accuracy: {accuracy:.2f}%')
  return accuracy


if __name__ == '__main__':
  # We do not have to train the modified deep network from scratch of course, we just load its weights from the trained instance
  modified_nn_deep = ModifiedDeepNNCosine(num_classes=10).to(device)
  modified_nn_deep.load_state_dict(nn_deep.state_dict())

  # Once again ensure the norm of the first layer is the same for both networks
  print('Norm of 1st layer for deep_nn:', torch.norm(nn_deep.features[0].weight).item())
  print('Norm of 1st layer for modified_deep_nn:', torch.norm(modified_nn_deep.features[0].weight).item())

  # Initialize a modified lightweight network with the same seed as our other lightweight instances. This will be trained from scratch to examine the effectiveness of cosine loss minimization.
  torch.manual_seed(42)
  modified_nn_light = ModifiedLightNNCosine(num_classes=10).to(device)
  print('Norm of 1st layer:', torch.norm(modified_nn_light.features[0].weight).item())
