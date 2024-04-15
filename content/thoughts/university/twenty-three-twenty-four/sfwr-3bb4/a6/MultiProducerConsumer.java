import java.util.concurrent.Semaphore;
import java.lang.String;

public class MultiProducerConsumer {
  public static Semaphore empty = new Semaphore(1);
  public static Semaphore full = new Semaphore(0);
  public static volatile int data; // shared buffer
  public static int numIters;

  static Thread makeproducer(int id) {
    return new Thread(() -> {
      System.out.println("Producer " + id + " created");
      for (int produced = 0; produced < numIters; produced++) {
        try {
          empty.acquire();
        } catch (Exception e) {
          e.printStackTrace();
        }
        data = produced;
        full.release();
      }
    });
  }

  static Thread makeconsumer(int id) {
    return new Thread(() -> {
      System.out.println("Consumer " + id + " created");
      int sum = 0;
      for (int consumed = 0; consumed < numIters; consumed++) {
        try {
          full.acquire();
        } catch (Exception e) {
          e.printStackTrace();
        }
        sum += data;
        empty.release();
      }
      System.out.println("For " + numIters + " iterations, the sum of consumer " + id + " is " + sum);
    });
  }

  public static void main(String args[]) {
    if (args.length != 2) {
      System.out.println("Usage: java MultiProducerConsumer <numIters> <numProducersAndConsumers>");
      return;
    }

    numIters = Integer.parseInt(args[0]);
    int numProducersAndConsumers = Integer.parseInt(args[1]);

    Thread[] producers = new Thread[numProducersAndConsumers];
    Thread[] consumers = new Thread[numProducersAndConsumers];

    for (int i = 0; i < numProducersAndConsumers; i++) {
      producers[i] = makeproducer(i);
      consumers[i] = makeconsumer(i);
      producers[i].start();
      consumers[i].start();
    }

    for (int i = 0; i < numProducersAndConsumers; i++) {
      try {
        producers[i].join();
        consumers[i].join();
      } catch (InterruptedException e) {
        e.printStackTrace();
      }
    }

    int expectedTotalSum = numProducersAndConsumers * numIters * (numIters - 1) / 2;
    System.out.println("The expected total sum is " + expectedTotalSum);
  }
}
