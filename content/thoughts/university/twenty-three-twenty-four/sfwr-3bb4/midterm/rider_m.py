from __future__ import annotations
from sys import stdout
from threading import Semaphore
from threading import Thread
from time import sleep

# class Rider(Thread):
#   def __init__(self, i):
#     Thread.__init__(self)
#     self.i = i
#
#   def run(self):
#     for _ in range(4):
#       global bicycles_
#       stdout.write(str(self.i) + ' riding\n')
#       sleep(2)
#       shop.acquire()
#
#       # Increment the counter for bicycles in the shop
#       counter.acquire()
#       bicycles_ += 1
#       stdout.write(f'Bicycles in shop: {bicycles_!s}\n')
#       counter.release()
#
#       stdout.write(str(self.i) + ' repairing\n')
#       sleep(1)
#
#       # Decrement the counter for bicycles in the shop
#       counter.acquire()
#       bicycles_ -= 1
#       stdout.write(f'Bicycles in shop: {bicycles_!s}\n')
#       counter.release()
#
#       shop.release()
#
#     stdout.write(str(self.i) + ' done\n')
#
#
# C, B = 3, 10
# shop = Semaphore(C)
# counter = Semaphore(1)  # Binary semaphore for the counter
# bicycles_ = 0  # Counter for bicycles in the shop
#
# riders = {Rider(i) for i in range(B)}
# for r in riders: r.start()
# for r in riders: r.join()

class Rider(Thread):
  def __init__(self, i):
    Thread.__init__(self)
    self.i = i

  def run(self):
    for _ in range(4):
      global avail
      stdout.write(str(self.i) + ' riding\n')
      sleep(2)
      mutex.acquire()
      avail -= 1
      stdout.write('avail after enter: ' + str(avail) + '\n')
      if avail < 0:
        mutex.release()
        free.acquire()
      else:
        mutex.release()
      stdout.write(str(self.i) + ' repairing\n')
      sleep(1)
      mutex.acquire()
      avail += 1
      stdout.write('avail after exit: ' + str(avail) + '\n')
      if avail <= 0:
        mutex.release()
        free.release()
      else:
        mutex.release()
    stdout.write(str(self.i) + ' done\n')

C, B = 3, 10
mutex, free, avail = Semaphore(1), Semaphore(0), C
stdout.write('occupancy: 0\n')
riders = {Rider(i) for i in range(B)}
for r in riders:
  r.start()
for r in riders:
  r.join()
