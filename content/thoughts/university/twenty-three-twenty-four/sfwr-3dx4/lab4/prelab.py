import matplotlib.pyplot as plt
import control


def p2a():
  # Define the transfer function
  num = [1]
  den = [1, 14, 45]
  sys = control.tf(num, den)

  # Generate the root locus plot
  control.root_locus(sys)

  # Add labels and title
  plt.xlabel('Real Axis')
  plt.ylabel('Imaginary Axis')
  plt.title('Root Locus Plot')

  # Display the plot
  plt.show()


def p2b():
  # Define the transfer function
  num = [1, -11, 28]
  den = [1, 19, 94, 120]
  sys = control.tf(num, den)

  # Generate the root locus plot
  control.root_locus(sys)

  # Add labels and title
  plt.xlabel('Real Axis')
  plt.ylabel('Imaginary Axis')
  plt.title('Root Locus Plot')

  # Display the plot
  plt.show()


def p2c():
  # Define the transfer function
  num = [1, 7]
  den = [1, 23, 183, 585, 648]
  sys = control.tf(num, den)

  # Generate the root locus plot
  control.root_locus(sys)

  # Add labels and title
  plt.xlabel('Real Axis')
  plt.ylabel('Imaginary Axis')
  plt.title('Root Locus Plot')

  # Display the plot
  plt.show()


if __name__ == '__main__':
  p2a()
  p2b()
  p2c()
