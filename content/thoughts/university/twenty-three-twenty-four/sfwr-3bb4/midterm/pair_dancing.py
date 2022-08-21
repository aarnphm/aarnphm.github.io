from __future__ import annotations
from random import random
from sys import stdout
from threading import Semaphore
from threading import Thread
from time import sleep

l, f = Semaphore(0), Semaphore(0)
count_mutx, c = Semaphore(1), 0

def pairleader(leadername: str):
  global c
  f.release()
  l.acquire()
  with count_mutx:
    c += 1
    stdout.write(f'{leadername} + {pairedFollower[leadername]}: {c}\n')

def pairfollower(followername: str):
  global c
  l.release()
  f.acquire()
  with count_mutx:
    pairedFollower[pairedLeader[followername]] = followername

class Leader(Thread):
  def __init__(self, name):
    Thread.__init__(self)
    self.name = name

  def run(self):
    sleep(random())  # up to 1 sec
    pairleader(self.name)

class Follower(Thread):
  def __init__(self, name):
    Thread.__init__(self)
    self.name = name

  def run(self):
    sleep(random())  # up to 1 sec
    pairfollower(self.name)

# Dictionaries to keep track of paired leaders and followers
pairedLeader = {}
pairedFollower = {}

for i in range(13):
  leaderName = chr(i + ord('A'))
  pairedLeader[chr(i + ord('N'))] = leaderName
  Leader(leaderName).start()

for i in range(13):
  followerName = chr(i + ord('N'))
  pairedFollower[chr(i + ord('A'))] = followerName
  Follower(followerName).start()
