import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Quicksort {
    static int N;   // number of elements to be sorted
    static int S;   // threshold for creating a sub-thread
    static int a[]; // array to be sorted
    static ExecutorService threadPool;

    static int partition(int p, int r) {
        int x = a[r];
        int i = p - 1;
        for (int j = p; j <= r - 1; j++) {
            if (a[j] <= x) {
                i++;
                int t = a[i];
                a[i] = a[j];
                a[j] = t;
            }
        }
        int t = a[i + 1];
        a[i + 1] = a[r];
        a[r] = t;
        return i + 1;
    }

    static void sequentialsort(int p, int r) {
        if (p < r) {
            int q = partition(p, r);
            sequentialsort(p, q - 1);
            sequentialsort(q + 1, r);
        }
    }

    static void parallelsort(int p, int r) {
        if (p < r) {
            if (r - p + 1 <= S) {
                sequentialsort(p, r);
            } else {
                int q = partition(p, r);

                if (threadPool == null) {
                    // Create a thread pool with a fixed number of threads (e.g., equal to the number of CPU cores)
                    threadPool = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
                }

                threadPool.execute(() -> parallelsort(p, q - 1));
                threadPool.execute(() -> parallelsort(q + 1, r));
            }
        }
    }

    public static void main(String args[]) {
        N = Integer.parseInt(args[0]);
        S = Integer.parseInt(args[1]);
        a = new int[N];
        Random random = new Random();
        for (int i = 0; i < N; i++) a[i] = random.nextInt(10000);

        final long start = System.currentTimeMillis();
        parallelsort(0, N - 1);

        if (threadPool != null) {
            threadPool.shutdown();
            try {
                threadPool.awaitTermination(Long.MAX_VALUE, java.util.concurrent.TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        final long end = System.currentTimeMillis();

        for (int i = 1; i < N; i++) assert a[i - 1] <= a[i];
        System.out.println((end - start) + " ms");
    }
}
