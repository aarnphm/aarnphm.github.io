import java.util.concurrent.Semaphore;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Set;

class Pub {
  private final int capacity; // maximum number of villagers that can be inside at once
  private final Semaphore semaphore; // to control access to the pub
  private final Set<Villager> villagersInside; // track villagers inside the pub

  public Pub(int capacity) {
    this.capacity = capacity;
    this.semaphore = new Semaphore(capacity, true);
    this.villagersInside = ConcurrentHashMap.newKeySet();
  }

  public boolean enter(Villager villager) {
    if (!semaphore.tryAcquire()) {
      return false; // Pub is full
    }
    synchronized (this) {
      if (villagersInside.add(villager)) {
        return true;
      } else {
        semaphore.release();
        return false;
      }
    }
  }

  public void exit(Villager villager) {
    synchronized (this) {
      if (villagersInside.remove(villager)) {
        semaphore.release();
      }
    }
  }

  public void closing() {
    synchronized (this) {
      for (Villager v : villagersInside) {
        v.interrupt(); // Interrupt all villagers inside the pub
      }
      villagersInside.clear();
      semaphore.drainPermits(); // Remove all permits to prevent new entries
    }
  }
}

class Villager extends Thread {
  private final Pub p;

  Villager(Pub p) {
    this.p = p;
  }

  public void run() {
    if (p.enter(this)) {
      System.out.print("🙂"); // entered pub
      try {
        Thread.sleep((long) (Math.random() * 1000));
        System.out.print("😋"); // full and happy
      } catch (InterruptedException e) {
        System.out.print("😞"); // interrupted while eating
      } finally {
        p.exit(this);
      }
    } else {
      System.out.print("🙁"); // turned down because the pub is full
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
    Manager manager = new Manager(p);
    manager.start();
    for (int i = 0; i < 20; i++) {
      Villager v = new Villager(p);
      v.start();
    }
  }
}
