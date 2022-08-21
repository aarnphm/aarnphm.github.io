import java.util.LinkedList;
import java.util.Queue;

class Pub {
  private final int capacity;
  private int inside = 0;
  private boolean isOpen = true;
  private final Queue<Villager> waitingQueue = new LinkedList<>();

  public Pub(int capacity) {
    this.capacity = capacity;
  }

  public synchronized boolean enter(Villager villager) {
    if (!isOpen) {
      return false; // Pub is closed, cannot enter
    }
    if (inside >= capacity) {
      waitingQueue.add(villager); // Villager has to wait
      while (isOpen && waitingQueue.peek() != villager) {
        try {
          wait(); // Wait until the villager is at the front of the queue
        } catch (InterruptedException e) {
          // Thread was interrupted while waiting
          waitingQueue.remove(villager);
          return false;
        }
      }
      if (!isOpen) {
        waitingQueue.remove(villager);
        return false; // Pub is closed while waiting
      }
      waitingQueue.remove(); // Villager is entering, remove from queue
    }
    inside++; // Villager enters
    return true;
  }

  public synchronized void exit() {
    inside--;
    if (!waitingQueue.isEmpty()) {
      notifyAll(); // Notify in case there are villagers waiting to enter
    }
  }

  public synchronized void closing() {
    isOpen = false;
    notifyAll(); // Notify all waiting threads that the pub is closing
    while (!waitingQueue.isEmpty()) {
      Villager villager = waitingQueue.poll();
      villager.interrupt(); // Interrupt all villagers in the queue
    }
  }
}

class Villager extends Thread {
  private final Pub p;

  Villager(Pub p) {
    this.p = p;
  }

  public void run() {
    try {
      Thread.sleep((long) (Math.random() * 6000));
    } catch (InterruptedException e) {
      // Thread was interrupted while sleeping
      return;
    }
    if (p.enter(this)) {
      System.out.print("🙂"); // entered pub
      try {
        Thread.sleep((long) (Math.random() * 1000));
        System.out.print("😋"); // full and happy
      } catch (InterruptedException e) {
        System.out.print("😞"); // interrupted while eating
      } finally {
        p.exit();
      }
    } else {
      System.out.print("🙁"); // turned down because the pub is closed or interrupted while waiting
    }
  }
}

class Manager extends Thread {
  private final Pub p;

  Manager(Pub p) {
    this.p = p;
  }

  public void run() {
    try {
      Thread.sleep(4000);
    } catch (InterruptedException e) {
      return;
    }
    System.out.print("🔒");
    p.closing();
  }
}

class Village {
  public static void main(String[] args) {
    Pub p = new Pub(8); // capacity 8
    new Manager(p).start();
    for (int i = 0; i < 20; i++) {
      new Villager(p).start();
    }
  }
}
