import java.util.concurrent.Semaphore;

class Pub {
  private final int capacity;
  private int inside = 0; // num villagers inside
  private boolean isOpen = true; // pub open
  private Semaphore semaphore; // to control access to the pub

  // Class invariant: inside <= capacity && inside >= 0
  // isOpen is true during operating hours and false otherwise

  public Pub(int capacity) {
    this.capacity = capacity;
    this.semaphore = new Semaphore(capacity, true);
  }

  // Precondition for enter(): must be called when pub is open
  // Postcondition: if returns true, inside has been incremented
  public synchronized boolean enter() {
    if (!isOpen || inside == capacity) {
      return false; // Can't enter if pub is closed or full
    }
    try {
      semaphore.acquire();
      inside++; // Villager enters the pub
      return true;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return false;
    }
  }

  // Precondition for exit(): Villager must be inside the pub
  // Postcondition: inside is decremented
  public synchronized void exit() {
    inside--; // Villager leaves the pub
    semaphore.release();
  }

  // Precondition for closing(): called by the manager to close the pub
  // Postcondition: isOpen is set to false
  public synchronized void closing() {
    isOpen = false;
    semaphore.drainPermits(); // Remove all permits to prevent new entries
  }
}

class Villager extends Thread {
  Pub p;

  Villager(Pub p) {
    this.p = p;
  }

  public void run() {
    try {
      Thread.sleep((long) (Math.random() * 6000));
    } catch (Exception e) {
    }
    if (p.enter()) {
      System.out.print("🙂"); // entered pub
      try {
        Thread.sleep((long) (Math.random() * 1000)); // eating
      } catch (Exception e) {
      }
      System.out.print("😋"); // full and happy
      p.exit();
    } else
      System.out.print("🙁"); // turned down
  }
}

class Manager extends Thread {
  Pub p;

  Manager(Pub p) {
    this.p = p;
  }

  public void run() {
    try {
      Thread.sleep(4000);
    } catch (Exception e) {
    }
    System.out.print("🔒");
    p.closing();
  }
}

class Village {
  public static void main(String[] args) {
    Pub p = new Pub(8); // capacity 8
    new Manager(p).start();
    for (int i = 0; i < 20; i++)
      new Villager(p).start();
  }
}
