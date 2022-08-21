from __future__ import annotations
from sys import stdout
from threading import Semaphore
from threading import Thread
from time import sleep

# global variables and semaphores
e = Semaphore(1)
r, w = Semaphore(0), Semaphore(0)
nr, nw = 0, 0
dr, dw = 0, 0
data = 0

def reader() -> None:
  global nr, nw, dr, dw, data, maxdata
  maxdata = 0
  stdout.write('Reader starting\n')
  while True:
    # entry protocol
    e.acquire()
    if dw > 0 or nw > 0:
      dr += 1
      e.release()
      r.acquire()
      # dr -= 1
    nr += 1
    if dr > 0:
      dr -= 1
      r.release()
    else:
      e.release()

    # critical section
    if data > maxdata: stdout.write('Reader in critical section read ' + str(data) + '\n')
    maxdata = data if data > maxdata else maxdata

    # exit protocol
    e.acquire()
    nr -= 1
    # if no active readers and there's a waiting writer
    if nr == 0 and dw > 0: w.release()
    e.release()

def writer(numIters: int) -> None:
  global nr, nw, dr, dw, data
  stdout.write('Writer starting\n')
  for i in range(numIters):
    e.acquire()
    # if there's an active reader or writer
    if nr > 0 or nw > 0:
      dw += 1
      e.release()
      w.acquire()
      dw -= 1
    nw += 1
    e.release()

    # critical section
    stdout.write('Writer in critical section\n')
    data = (i + numIters // 2) % numIters
    stdout.write('Writer writing ' + str(data) + '\n')

    # exit protocol
    e.acquire()
    nw -= 1
    if dr > 0: r.release()  # if there are waiting readers
    elif dw > 0: w.release()  # if there's a waiting writer
    else: e.release()
    sleep(1)

def rw(numIters: int) -> None:
  r = Thread(target=reader, daemon=True)
  w = Thread(target=writer, args=(numIters,))
  r.start(); w.start()  # yapf: disable
