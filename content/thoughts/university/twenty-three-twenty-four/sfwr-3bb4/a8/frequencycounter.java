import java.util.Random;

class FrequencyCounter {
  // The class invariant is that the frequencies array must always have
  // non-negative values,
  // and its length should be equal to the 'max' value passed during the
  // construction of the object.
  // The precondition for the count method is that the event number must be
  // between 0 and max - 1.
  // The precondition for the frequency method is the same as that for the count
  // method.

  private final int[] frequencies;
  private final Object writeLock = new Object();

  FrequencyCounter(int max) {
    // Preconditions: max > 0
    // Initialize an array to hold the frequencies of each event.
    frequencies = new int[max];
  }

  void count(int event) {
    // Preconditions: event >= 0 && event < frequencies.length
    synchronized (writeLock) {
      // Increment the frequency of the specified event by 1.
      frequencies[event]++;
    }
  }

  int frequency(int event) {
    // Preconditions: event >= 0 && event < frequencies.length
    synchronized (writeLock) {
      // Return the frequency of the specified event.
      return frequencies[event];
    }
  }
}

class Eventer extends Thread {
  FrequencyCounter fc;

  Eventer(FrequencyCounter fc) {
    this.fc = fc;
  }

  public void run() {
    Random r = new Random();
    for (int i = 0; i < 20000; i++) {
      fc.count(r.nextInt(10));
    }
  }
}

class TestFrequencyCounter {
  public static void main(String[] args) {
    final int E = 1000; // number of eventers
    FrequencyCounter fc = new FrequencyCounter(10);
    Eventer ev[] = new Eventer[E];
    for (int i = 0; i < E; i++)
      ev[i] = new Eventer(fc);
    long startTime = System.currentTimeMillis();
    for (int i = 0; i < E; i++)
      ev[i].start();
    for (int i = 0; i < E; i++) {
      try {
        ev[i].join();
      } catch (Exception e) {
      }
    }
    long endTime = System.currentTimeMillis();
    System.out.println(endTime - startTime);
  }
}
