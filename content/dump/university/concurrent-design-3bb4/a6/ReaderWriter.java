import java.util.concurrent.Semaphore;
import java.lang.String;

public class ReaderWriter {
  /* global variables and semaphores */
  private static Semaphore e = new Semaphore(1); // for exclusive access to protocol variables
  private static Semaphore r = new Semaphore(0); // for delaying readers
  private static Semaphore w = new Semaphore(0); // for delaying writers
  private static volatile int nr = 0; // number of readers
  private static volatile int nw = 0; // number of writers
  private static int dr = 0; // number of delayed readers
  private static int dw = 0; // number of delayed writers
  private static int data = 0; // shared data
  public static int numIters = 0;
  public static int maxdata = 0;

  static void reader() {
    while (true) {
      /* entry protocol */
      try {
        e.acquire();
        if (nw > 0) {
          dr++;
          e.release();
          r.acquire();
        }
        nr++;
        if (dr > 0) {
          dr--;
          r.release();
        } else {
          e.release();
        }
      } catch (InterruptedException ex) {
        ex.printStackTrace();
      }

      if (data > maxdata)
        System.out.printf("Reader in critical section read %d\n", data);
      maxdata = data > maxdata ? data : maxdata;

      /* exit protocol */
      try {
        e.acquire();
        nr--;
        if (nr == 0 && dw > 0) {
          dw--;
          w.release();
        } else {
          e.release();
        }
      } catch (InterruptedException ex) {
        ex.printStackTrace();
      }
    }
  }

  static void writer() {
    System.out.printf("Writer starting\n");
    for (int i = 0; i < numIters; i++) {
      /* entry protocol */
      try {
        e.acquire();
        if (nr > 0 || nw > 0) {
          dw++;
          e.release();
          w.acquire();
        }
        nw++;
        e.release();
      } catch (InterruptedException ex) {
        ex.printStackTrace();
      }

      System.out.printf("Writer in critical section\n");
      data = (i + numIters / 2) % numIters;
      System.out.printf("Writer writing %d\n", data);

      /* exit protocol */
      try {
        e.acquire();
        nw--;
        if (dr > 0) {
          dr--;
          r.release();
        } else if (dw > 0) {
          dw--;
          w.release();
        } else {
          e.release();
        }
      } catch (InterruptedException ex) {
        ex.printStackTrace();
      }

      try {
        Thread.sleep(1000); // sleep 1 sec
      } catch (Exception e) {
      }
    }
  }

  public static void main(String args[]) {
    numIters = Integer.parseInt(args[0]);
    Thread r = new Thread() {
      {
        setDaemon(true);
      }

      public void run() {
        reader();
      }
    };
    Thread w = new Thread() {
      public void run() {
        writer();
      }
    };

    r.start();
    w.start();
    try {
      w.join();
    } catch (Exception e) {
    }
    System.out.printf("Max data %d\n", maxdata);
  }
}
