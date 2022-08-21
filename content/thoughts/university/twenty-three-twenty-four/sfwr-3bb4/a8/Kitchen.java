class Kitchen {
  private int servings = 0;
  private final int MAX_SERVINGS = 10;

  // Class invariant: 0 <= servings <= MAX_SERVINGS
  public synchronized void getServingFromPot() {
    if (servings > 0) {
      servings--;
      System.out.println("scout getting 1 serving");
      if (servings <= 2) {
        notifyAll(); // Wake the cook to refill if servings are two or fewer
      }
    } else {
      try {
        wait(); // Wait for the cook to refill the pot
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }
  }

  public synchronized void putServingsInPot(int count) {
    while (servings > 2) {
      try {
        wait(); // Wait until servings are two or fewer
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }
    servings = count;
    System.out.println("cook putting " + count + " servings");
    notifyAll(); // Notify waiting scouts that the pot has been refilled
  }

  public synchronized void fillUpPot() {
    if (servings <= 2) {
      putServingsInPot(MAX_SERVINGS); // Refill the pot to MAX_SERVINGS
    }
  }
}

class Cook extends Thread {
  Kitchen k;

  Cook(Kitchen k) {
    this.k = k;
    setDaemon(true);
  }

  public void run() {
    while (true) {
      k.fillUpPot();
    }
  }
}

class Scout extends Thread {
  Kitchen k;

  Scout(Kitchen k) {
    this.k = k;
    setDaemon(true);
  }

  public void run() {
    while (true) {
      k.getServingFromPot();
      try {
        Thread.sleep(500); // scouting for 0.5 seconds
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }
  }
}

class TestScouts {
  public static void main(String[] args) {
    Kitchen k = new Kitchen();
    Cook c = new Cook(k);
    c.start();
    Scout[] sc = new Scout[20];
    for (int i = 0; i < 20; i++) {
      sc[i] = new Scout(k);
      sc[i].start();
    }
    try {
      Thread.sleep(5000); // The main program runs for 5 seconds
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
    System.out.println("Done");
  }
}

