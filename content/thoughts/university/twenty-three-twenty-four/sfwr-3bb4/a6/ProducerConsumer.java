import java.util.concurrent.Semaphore;
import java.lang.String;

public class ProducerConsumer {
  public static Semaphore empty = new Semaphore(1);
  public static Semaphore full = new Semaphore(0);
  public static volatile int data; // shared buffer
  public static int numIters;

  static Thread makeproducer() {
    return new Thread() {
      public void run() {
        System.out.println("Producer created");
        for (int produced = 0; produced < numIters; produced++) {
          try {
            empty.acquire();
          } catch (Exception e) {
          }
          data = produced;
          full.release();
        }
      }
    };
  }

  static Thread makeconsumer() {
    return new Thread() {
      public void run() {
        System.out.println("Consumer created");
        int sum = 0;
        for (int consumed = 0; consumed < numIters; consumed++) {
          try {
            full.acquire();
          } catch (Exception e) {
          }
          sum += data;
          empty.release();
        }
        System.out.println("For " + numIters + " iterations, the sum is " + sum);
      }
    };
  }

  public static void main(String args[]) {
    numIters = Integer.parseInt(args[0]);
    Thread p = makeproducer(), c = makeconsumer();

    p.start();
    c.start();
    try {
      p.join();
      c.join();
    } catch (Exception e) {
    }
  }
}
