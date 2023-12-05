import enum, numpy as np, matplotlib.pyplot as plt; from scipy.io import loadmat

def parse(): return (data:=loadmat('points.mat'))['x'], data['labels']

def init_weights(hidden, output_size):
  W2,b2=np.random.randn(hidden,2)*0.01,np.zeros((hidden, 1))
  W3,b3=np.random.randn(hidden,hidden)*0.01,np.zeros((hidden, 1))
  W4,b4=np.random.randn(output_size,hidden)*0.01,np.zeros((output_size, 1))
  return W2, b2, W3, b3, W4, b4

class Activation(enum.IntEnum): LeakyReLU = enum.auto(); ReLU = enum.auto(); Sigmoid = enum.auto()
def leaky_relu(z,alpha=0.01): return np.maximum(alpha*z, z)
def leaky_derivative(z,alpha=0.01): return np.where(z>0, 1, alpha)
def sigmoid(z): return 1/(1+np.exp(-z))
def actfn(z, activation=1): return {Activation.LeakyReLU: leaky_relu, Activation.ReLU: np.maximum, Activation.Sigmoid: sigmoid}[Activation(activation)](z)

def cost(a4, labels, epsilon=1e-12): m = labels.shape[1]; return -np.sum(labels * np.log(a4 + epsilon) + (1 - labels) * np.log(1 - a4 + epsilon)) / m
def accuracy(a4, labels): return np.mean(np.argmax(a4, axis=0) == np.argmax(labels, axis=0)) * 100

def forward(x, W2, b2, W3, b3, W4, b4): return (a2:=actfn(W2@x+b2)),(a3:=actfn(W3@a2+b3)),actfn(W4@a3+b4, 3)
def backprop(x, labels, a2, a3, a4, W3, W4):
  m = labels.shape[1]
  d4=(a4-labels)*a4*(1-a4)
  d3=W4.T@d4*leaky_derivative(a3)
  d2=W3.T@d3*leaky_derivative(a2)

  gradW4,gradW3,gradW2=d4@a3.T/m,d3@a2.T/m,d2@x.T/m
  gradb4,gradb3,gradb2=np.sum(d4,axis=1,keepdims=True)/m,np.sum(d3,axis=1,keepdims=True)/m,np.sum(d2,axis=1,keepdims=True)/m

  return gradW4,gradb4,gradW3,gradb3,gradW2,gradb2

def train() -> int:
  x, labels = parse(); x = (x-np.mean(x, axis=1,keepdims=True))/np.std(x, axis=1,keepdims=True) # normalise datasets
  hidden, eta, alpha,lambda_, batch_size, epochs, decay_rate, decay_step = 20, 1e-3, 0.89, 1e-3, 24, int(1e6), 0.99, 10000

  W2, b2, W3, b3, W4, b4 = init_weights(hidden, labels.shape[0])
  mW2, mb2 = np.zeros_like(W2), np.zeros_like(b2)
  mW3, mb3 = np.zeros_like(W3), np.zeros_like(b3)
  mW4, mb4 = np.zeros_like(W4), np.zeros_like(b4)
  costs, accuracies = np.zeros(epochs), np.zeros(epochs)

  for count in range(epochs):
    batch_idx = np.random.choice(x.shape[1], batch_size, replace=False)
    a2, a3, a4 = forward((_xb := x[:, batch_idx]), W2, b2, W3, b3, W4, b4)
    gradW4, gradb4, gradW3, gradb3, gradW2, gradb2 = backprop(_xb, labels[:, batch_idx], a2, a3, a4, W3, W4)

    # update
    mW2=alpha*mW2-eta*(gradW2+lambda_*W2/batch_size)
    mW3=alpha*mW3-eta*(gradW3+lambda_*W3/batch_size)
    mW4=alpha*mW4-eta*(gradW4+lambda_*W4/batch_size)
    mb2=alpha*mb2-eta*gradb2
    mb3=alpha*mb3-eta*gradb3
    mb4=alpha*mb4-eta*gradb4
    W2+=mW2;W3+=mW3;W4+=mW4
    b2+=mb2;b3+=mb3;b4+=mb4

    if count % 1000 == 0 or count == epochs - 1:
      _, _, a4f = forward(x, W2, b2, W3, b3, W4, b4)
      costs[count] = cost(a4f, labels)
      accuracies[count] = accuracy(a4f, labels)
      print(f'Iteration {count}: Cost {costs[count]:.4f}, Accuracy {accuracies[count]:.2f}%')
      if accuracies[count] >= 95: print(f'Early stopping: Reached 95% accuracy at iteration {count}'); break
    if count % decay_step == 0: eta *= decay_rate

  # Plots
  plot_decision_boundary(x, labels, W2, b2, W3, b3, W4, b4); plot_relations(count, costs, accuracies)
  return 0

def plot_decision_boundary(x, labels, W2, b2, W3, b3, W4, b4, h=0.01):
    # Generate a grid of points to plot the decision boundaries
    x_min, x_max = x[0, :].min() - 1, x[0, :].max() + 1
    y_min, y_max = x[1, :].min() - 1, x[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = actfn(np.dot(W2, np.c_[xx.ravel(), yy.ravel()].T) + b2)
    Z = actfn(np.dot(W3, Z) + b3)
    Z = actfn(np.dot(W4, Z) + b4, 3)
    Z = np.argmax(Z, axis=0)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z.reshape(xx.shape), cmap=plt.cm.Spectral)
    plt.ylabel('x2'); plt.xlabel('x1')
    plt.scatter(x[0, :], x[1, :], c=np.argmax(labels, axis=0), cmap=plt.cm.Spectral)

def plot_relations(count, costs, accuracies):
  # Plotting cost and accuracy
  plt.figure(); plt.subplot(2, 1, 1); plt.plot(costs[:count+1]); plt.title('Cost over iterations')
  plt.ylabel('Cost')
  plt.subplot(2, 1, 2); plt.plot(accuracies[:count+1]); plt.title('Accuracy over iterations')
  plt.xlabel('Iterations'); plt.ylabel('Accuracy (%)')
  plt.show()

if __name__ == '__main__': np.random.seed(420); raise SystemExit(train())
