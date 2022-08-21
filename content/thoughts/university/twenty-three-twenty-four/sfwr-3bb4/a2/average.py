from __future__ import annotations
from typing import Any
from threading import Thread
import random, time

class Worker(Thread):
  def __init__(self, arr: list[Any], start: int, end: int) -> None:
    super().__init__()
    self._arr, self._start, self._end = arr, start, end
    self._result = 0.0

  def run(self) -> None:
    for i in range(self._start, self._end): self._result += self._arr[i]

def sequentialaverage(a: int): return sum(a) / len(a)

def parallelaverage(a: int, p: int):
  n = len(a)
  chunk_size = n // p
  workers: list[Worker]= []

  for i in range(p):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < p - 1 else n
    worker = Worker(a, start, end)
    workers.append(worker)
    worker.start()

  for worker in workers: worker.join()

  total_sum = 0.0
  for worker in workers: total_sum += worker._result

  return total_sum / n

def average(n, p = 1):
  a = [random.randint(0, 1000) for i in range(n)]

  start = time.time_ns() / 1000
  avg = sequentialaverage(a)
  end  = time.time_ns() / 1000
  print("Sequential:", avg, "Time:", end - start, "µs")

  start = time.time_ns() / 1000
  avg = parallelaverage(a, p)
  end  = time.time_ns() / 1000
  print("Parallel:", avg, "Time:", end - start, "µs")
