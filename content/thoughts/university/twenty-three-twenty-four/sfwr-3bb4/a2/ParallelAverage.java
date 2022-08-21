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
    static double parallelaverage(int a[], int p) {
        // a.length > 0 && p > 0
        int n = a.length;
        Worker[] workers = new Worker[p];
        int chunkSize = n / p;
        
        for (int i = 0; i < p; i++) {
            int start = i * chunkSize;
            int end = (i == p - 1) ? n : (i + 1) * chunkSize;
            workers[i] = new Worker(a, start, end);
            workers[i].start();
        }

        try {
            for (int i = 0; i < p; i++) {
                workers[i].join();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        double totalSum = 0.0;
        for (int i = 0; i < p; i++) {
            totalSum += workers[i].getResult();
        }

        return totalSum / n;
    }
    public static void main(String args[]) {
        int n = Integer.parseInt(args[0]); // compute the average of n random numbers ...
        int p = Integer.parseInt(args[1]); // ... using p threads
        int[] a = new int[n];
        Random rand = new Random();
        for (int i = 0; i < n; i++) a[i] = rand.nextInt(10000);
        
        long start = System.currentTimeMillis();
        double avg = sequentialaverage(a);
        long end = System.currentTimeMillis();
        System.out.println("Sequential: " + avg + " Time: " + (end - start) + " ms");

        start = System.currentTimeMillis();
        avg = parallelaverage(a, p);
        end = System.currentTimeMillis();
        System.out.println("Parallel: " + avg + " Time: " + (end - start) + " ms");
    }
}

