# 02.
def d(n): return sum([i for i in range(1, n) if n % i == 0])
def a(m): return sum([i for i in range(1, m) if i == d(d(i)) and i != d(i)])

# 04.

# 05.
def interleavings(a,b):
  s = set()
  def helper(pref,a,b):
    if not a and not b: s.add(pref)
    if a: helper(pref + a[0], a[1:], b)
    if b: helper(pref + b[0], a, b[1:])
  helper('', a, b)
  return s


# 06.
import threading

def parallelmax(a, p=2):
  n = len(a)
  if n == 0: return

  def max_sub(arr): return max(arr) if arr else float('-inf')

  chunk = n // p
  sub = [a[i:i+chunk] for i in range(0, n, chunk)]
  max_values = [0] * p
  threads = []
  for i in range(p):
    thread = threading.Thread(target=lambda i=i: max_values.__setitem__(i, max_sub(sub[i])))
    thread.start()
    threads.append(thread)
  
  # Wait for all threads to finish
  for thread in threads:
    thread.join()
  
  return max(max_values)
