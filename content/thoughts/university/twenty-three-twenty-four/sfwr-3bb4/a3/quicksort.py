import os
from threading import Thread, Semaphore
from random import randint
from time import time

MAX_THREADS = int(os.environ.get('MAX_THREADS', 100))
sem = Semaphore(MAX_THREADS)

def partition(p, r): 
  x, i = a[r], p - 1
  for j in range(p, r):
    if a[j] <= x: 
      i += 1; a[i], a[j] = a[j], a[i]
  a[i + 1], a[r] = a[r], a[i + 1]
  return i + 1;
    

def sequentialsort(p, r):
  if p < r:
    q = partition(p, r)
    sequentialsort(p, q - 1)
    sequentialsort(q + 1, r)

def parallelsort(p, r):
  if p<r:
    q = partition(p, r)
    left = q - p
    right = r - q
    if left > S and right > S:
      rthread = Thread(target=parallelsort, args=(q + 1, r))
      if sem.acquire(blocking=False):
        rthread.start()
        parallelsort(p, q - 1)
        rthread.join()
        sem.release()
      else:
        sequentialsort(p, r)
    else:
      sequentialsort(p, q-1)
      sequentialsort(q+1, r)

def quicksort(N, step):
  global a, S
  a, S = [randint(0, 10000) for _ in range(N)], step

  start = time()    
  parallelsort(0, N - 1)
  end = time()

  print(str(int((end - start) * 1000)) + " ms")
  for i in range(1, N): assert a[i - 1] <= a[i]    
