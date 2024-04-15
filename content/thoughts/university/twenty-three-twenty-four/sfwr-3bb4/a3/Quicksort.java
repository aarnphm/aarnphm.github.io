import java.util.Random;

public class Quicksort {
    static int N;   // number of elements to be sorted
    static int S;   // threshold for creating a sub-thread
    static int a[]; // array to be sorted
    static int activeThreads = 0;

    static int partition(int p, int r) {
        int x = a[r];
        int i = p - 1;
        for (int j = p; j <= r - 1; j++) {
            if (a[j] <= x) {
                i++; 
                int t = a[i]; a[i] = a[j]; a[j] = t;
            }
        }
        int t = a[i + 1]; a[i + 1] = a[r]; a[r] = t;
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
            int q = partition(p, r);
            int left = q - p;
            int right = r - q;
    
            if (left > S && right > S && activeThreads < S) {
                Thread rightThread = new Thread(() -> {
                    activeThreads++;
                    parallelsort(q + 1, r);
                    activeThreads--;
                });
    
                rightThread.start();
                parallelsort(p, q - 1);
    
                try {
                    rightThread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                sequentialsort(p, q - 1);
                sequentialsort(q + 1, r);
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
        final long end = System.currentTimeMillis();
        
        for (int i = 1; i < N; i++) assert a[i - 1] <= a[i];
        System.out.println((end - start) + " ms");
    }
}
