import java.util.Random;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

class HighFrequencyCounter {
  // Class invariant: The frequencies array must always have non-negative values,
  // and its length should be equal to the 'max' value passed during the
  // construction of the object.
  // Each index of the frequencies array is guarded by the corresponding lock in
  // the locks array.

  private final int[] frequencies;
  private final Lock[] locks;

  HighFrequencyCounter(int max) {
    // Preconditions: max > 0
    frequencies = new int[max];
    locks = new Lock[max];
    for (int i = 0; i < max; i++) {
      locks[i] = new ReentrantLock();
    }
  }

  void count(int event) {
    // Preconditions: event >= 0 && event < frequencies.length
    locks[event].lock();
    try {
      // Increment the frequency of the specified event by 1.
      frequencies[event]++;
    } finally {
      locks[event].unlock();
    }
  }

  int frequency(int event) {
    // Preconditions: event >= 0 && event < frequencies.length
    locks[event].lock();
    try {
      // Return the frequency of the specified event.
      return frequencies[event];
    } finally {
      locks[event].unlock();
    }
  }
}

class HEventer extends Thread {
  HighFrequencyCounter hfc;

  HEventer(HighFrequencyCounter hfc) {
    this.hfc = hfc;
  }

  public void run() {
    Random r = new Random();
    for (int i = 0; i < 20000; i++) {
      hfc.count(r.nextInt(10));
    }
  }
}

class TestHighFrequencyCounter {
  public static void main(String[] args) {
    final int E = 1000; // number of eventers
    HighFrequencyCounter hfc = new HighFrequencyCounter(10);
    HEventer[] hev = new HEventer[E];
    for (int i = 0; i < E; i++)
      hev[i] = new HEventer(hfc);
    long hstartTime = System.currentTimeMillis();
    for (int i = 0; i < E; i++)
      hev[i].start();
    for (int i = 0; i < E; i++) {
      try {
        hev[i].join();
      } catch (Exception e) {
      }
    }
    long hendTime = System.currentTimeMillis();
    System.out.println("Time taken: " + (hendTime - hstartTime) + " ms");
    // Optionally, print frequencies to verify correctness
    for (int i = 0; i < 10; i++) {
      System.out.println("Event " + i + ": " + hfc.frequency(i));
    }
  }
}
