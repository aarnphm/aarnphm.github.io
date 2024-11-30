import dis


def compare_not_in(x, y) -> bool:
  return x not in y


def compare_not_x_in(x, y) -> bool:
  return not x in y


# Create sample data
x = 5
y = [1, 2, 3, 4]

print("Bytecode for 'x not in y':")
dis.dis(compare_not_in)

print("\nBytecode for 'not x in y':")
dis.dis(compare_not_x_in)
