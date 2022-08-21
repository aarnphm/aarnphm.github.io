import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Future;
import java.util.concurrent.Callable;

public class Fibonacci {
    static int sequentialfib(int n) {
        if (n <= 1) return n;
        else {
            int x = sequentialfib(n - 1);
            int y = sequentialfib(n - 2);
            return x + y;
        }
    }
    static int parallelfib(int n, int p) {
        if (n <= 1) return n;
        
        ExecutorService executor = Executors.newFixedThreadPool(p);

        // Create a Callable task to calculate fib(n) in parallel.
        Callable<Integer> task = new Callable<Integer>() {
            @Override
            public Integer call() {
                return sequentialfib(n);
            }
        };

        // Submit the task for execution.
        Future<Integer> future = executor.submit(task);
        
        try {
            // Get the result from the Callable and shut down the thread pool.
            int result = future.get();
            executor.shutdown();
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            return result;
        } catch (Exception e) {
            e.printStackTrace();
            return -1;
        }
    }
    public static void main(String args[]) {
        int n = Integer.parseInt(args[0]);
        int p = Integer.parseInt(args[1]);

        long start = System.currentTimeMillis();
        int r = parallelfib(n, p);
        long end = System.currentTimeMillis();
        System.out.println("Parallel: " + r + " by " + (end - start) + " ms");

        start = System.currentTimeMillis();
        r = sequentialfib(n);
        end = System.currentTimeMillis();
        System.out.println("Sequential: " + r + " by " + (end - start) + " ms");
    }
}
