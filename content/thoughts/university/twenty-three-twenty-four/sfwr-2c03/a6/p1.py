from graphviz import Digraph
import matplotlib.pyplot as plt
import PIL.Image as Image
import io

# Define the color mapping for the graph
colors = {True: 'black', False: 'red'}


class Node:
  def __init__(self, value, color=True, left=None, right=None, parent=None):
    self.value = value
    self.color = color
    self.left = left
    self.right = right
    self.parent = parent

  def is_red(self):
    return not self.color

  def is_black(self):
    return self.color


class RedBlackTree:
  def __init__(self):
    self.root = None
    self.steps = []

  def left_rotate(self, x):
    y = x.right
    x.right = y.left
    if y.left:
      y.left.parent = x
    y.parent = x.parent
    if not x.parent:
      self.root = y
    elif x == x.parent.left:
      x.parent.left = y
    else:
      x.parent.right = y
    y.left = x
    x.parent = y

  def right_rotate(self, y):
    x = y.left
    y.left = x.right
    if x.right:
      x.right.parent = y
    x.parent = y.parent
    if not y.parent:
      self.root = x
    elif y == y.parent.left:
      y.parent.left = x
    else:
      y.parent.right = x
    x.right = y
    y.parent = x

  def insert(self, value):
    new_node = Node(value, color=False)  # New nodes are red
    y = None
    x = self.root

    while x:
      y = x
      if new_node.value < x.value:
        x = x.left
      else:
        x = x.right

    new_node.parent = y
    if not y:
      self.root = new_node
    elif new_node.value < y.value:
      y.left = new_node
    else:
      y.right = new_node

    self.fix_insert(new_node)
    self.record_step()

  def fix_insert(self, k):
    while k != self.root and k.parent.is_red():
      if k.parent == k.parent.parent.left:
        u = k.parent.parent.right
        if u and u.is_red():
          u.color = True
          k.parent.color = True
          k.parent.parent.color = False
          k = k.parent.parent
        else:
          if k == k.parent.right:
            k = k.parent
            self.left_rotate(k)
          k.parent.color = True
          k.parent.parent.color = False
          self.right_rotate(k.parent.parent)
      else:
        u = k.parent.parent.left
        if u and u.is_red():
          u.color = True
          k.parent.color = True
          k.parent.parent.color = False
          k = k.parent.parent
        else:
          if k == k.parent.left:
            k = k.parent
            self.right_rotate(k)
          k.parent.color = True
          k.parent.parent.color = False
          self.left_rotate(k.parent.parent)
    self.root.color = True

  def record_step(self):
    dot = Digraph(comment='Red Black Tree', format='png')

    def add_nodes_edges(node, parent=None, edge_label=''):
      if node:
        dot.node(str(node.value), str(node.value), color=colors[node.color], style='filled', fontcolor='white')
        if parent:
          dot.edge(str(parent.value), str(node.value), label=edge_label)
        add_nodes_edges(node.left, node, 'L')
        add_nodes_edges(node.right, node, 'R')

    add_nodes_edges(self.root)
    # Render the image
    output = io.BytesIO(dot.pipe())
    image = Image.open(output)
    self.steps.append(image)

  def insert_sequence(self, sequence):
    for value in sequence:
      self.insert(value)

  def display_steps(self):
    for step, image in enumerate(self.steps, start=1):
      plt.imshow(image)
      plt.axis('off')
      plt.title(f'Step {step}')
      plt.show()


# Values to be inserted into the tree
values = [3, 42, 39, 86, 49, 89, 99, 20, 88, 51, 64]

# Initialize the tree and insert values
rb_tree = RedBlackTree()
rb_tree.insert_sequence(values)

# Save each step image to file
step_images_paths = []
for i, image in enumerate(rb_tree.steps, start=1):
  step_image_path = f'./rb_tree_step_{i}.png'
  image.save(step_image_path)
  step_images_paths.append(step_image_path)
