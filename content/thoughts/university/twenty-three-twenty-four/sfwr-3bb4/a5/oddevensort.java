import java.util.Random;
import java.util.Arrays;
import java.util.concurrent.Semaphore;

public class oddevensort {
    public static int [] a;
    public static Semaphore[] barriers;

    public static int partition(int p, int r){
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

    public static void sequentialsort(int p, int r) {
        if (p < r) {
            int q = partition(p, r);
            sequentialsort(p, q - 1);
            sequentialsort(q + 1, r);
        }
    }

    private static int counter = 0; // shared counter
    private static final Object lock = new Object(); // lock for shared counter

    public static void barriersync(int p) { // p is the current thread id
        synchronized (lock) {
            counter++;
            if (counter == barriers.length) {
                // If this is the last thread, release all semaphores and reset counter
                for (Semaphore barrier : barriers) {
                    barrier.release();
                }
                counter = 0;
            }
        }
        try {
            barriers[p].acquire();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static Thread makesorter (int l0, int u0, int l1, int u1, int p, int r) {
        return new Thread () {
            public void run() {
                for (int i = 0; i < r; i++) {
                    // System.out.printf("%d sorts %d to %d round  %d\n", p, l0, u0, i);
                    sequentialsort(l0, u0); barriersync(p);
                    // System.out.printf("%d sorts %d to %d round  %d\n", p, l1, u1, i);
                    sequentialsort(l1, u1); barriersync(p);
                }
            }
        };
    }

    public static void main(String args[]) {
        int P = Integer.parseInt(args[0]); // number of sorting threads
        int M = Integer.parseInt(args[1]); // number of elements each thread sorts sequentially
        int N = P * M + M / 2;             // number of elements to be sorted
        System.out.println("size " + N);

        a = new int[N];
        Random random = new Random();
        for (int i = 0; i < N; i++) a[i] = random.nextInt(10000);
        System.out.println(Arrays.toString(a));

        barriers = new Semaphore[P];
        for (int p = 0; p < P; p++) barriers[p] = new Semaphore(0);

        Thread sorters[] = new Thread[P];
        for (int p = 0; p < P; p++) sorters[p] =
            makesorter(p * M, (p + 1) * M - 1, p * M + M / 2, (p + 1) * M + M / 2 - 1, p, P + 1);
        for (Thread s: sorters) s.start();
        try {for(Thread s: sorters) s.join();
        } catch (Exception e) {}

        for (int i = 1; i < N; i++) assert a[i - 1] <= a[i];
         System.out.println(Arrays.toString(a));
    }
}
