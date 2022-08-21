import java.util.Random;

class Worker extends Thread {
  private int[] arr;
  private int start, end;
  private double result;

  public Worker(int[] arr, int start, int end) {
      this.arr = arr;
      this.start = start;
      this.end = end;
      this.result = 0.0;
  }

  @Override
  public void run() {
      for (int i = start; i < end; i++) {
          result += arr[i];
      }
  }

  public double getResult() {
      return result;
  }
}

public class Average {
    static double sequentialaverage(int a[]) {
        // a.length > 0
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i];
        return s / a.length;
    }
    static double parallelaverage(int a[]) {
        // a.length > 0
        int n = a.length;
        int mid = n / 2;

        Worker worker1 = new Worker(a, 0, mid);
        Worker worker2 = new Worker(a, mid, n);

        worker1.start();
        worker2.start();

        try {
            worker1.join();
            worker2.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        double sum1 = worker1.getResult();
        double sum2 = worker2.getResult();

        return (sum1 + sum2) / n;
    }
    public static void main(String args[]) {
        int n = Integer.parseInt(args[0]); // compute the average of n random numbers
        int[] a = new int[n];
        Random rand = new Random();
        for (int i = 0; i < n; i++) a[i] = rand.nextInt(10000);
        
        long start = System.currentTimeMillis();
        double avg = sequentialaverage(a);
        long end = System.currentTimeMillis();
        System.out.println("Sequential: " + avg + " Time: " + (end - start) + " ms");

        start = System.currentTimeMillis();
        avg = parallelaverage(a);
        end = System.currentTimeMillis();
        System.out.println("Parallel: " + avg + " Time: " + (end - start) + " ms");
    }
}
