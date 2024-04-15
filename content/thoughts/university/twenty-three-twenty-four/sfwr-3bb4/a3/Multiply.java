import java.util.Random;
import java.util.Arrays;


public class Multiply {

    static int N;        // number of rows in Matrix
    static int P;        // number of workers
    static int[][] a, b; // randomly generated input matrices

    static void sequentialmultiply(int c[][]) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                c[i][j] = 0; 
                for (int k = 0; k < N; k++) {
                    c[i][j] = c[i][j] + a[i][k] * b[k][j];
                }
            }
        }
    }

    static class Worker extends Thread {
      int w;
      int[][] c;
      Worker(int w, int c[][]) { this.w = w; this.c = c; }
      public void run() {
        int first = w*N/P;
        int last = (w+1)*N/P-1;
        for (int i = first; i<=last; i++) {
          for (int j=0; j<N; j++) {
            c[i][j] = 0;
            for (int k = 0; k<N; k++) {
              c[i][j] += a[i][k] * b[k][j];
            }
          }
        }
      }
    }
    static void parallelmultiply(int c[][]) {
      Worker[] workers = new Worker[P];
      for (int w = 0; w<P;w++) {
        workers[w] = new Worker(w, c);
        workers[w].start();
      }
      try {
        for (int w = 0; w < P; w++) {
            workers[w].join();
        }
      } catch (InterruptedException e) {
          e.printStackTrace();
      }
    }

    public static void main(String args[]) {
        N = Integer.parseInt(args[0]);
        P = Integer.parseInt(args[1]);
        a = new int[N][N]; b = new int[N][N];
        int[][] cp = new int[N][N], cs = new int[N][N];
        Random random = new Random();
        for (int i = 0; i < N; i++) {
            for (int j=0; j < N; j++) {
                a[i][j] = random.nextInt(1000);
                b[i][j] = random.nextInt(1000);
            }
        }

        final long start = System.currentTimeMillis();
        parallelmultiply(cp);
        final long end = System.currentTimeMillis();
        
        sequentialmultiply(cs);  // check the correctness
        assert Arrays.deepEquals(cp, cs);
        
        System.out.println((end - start) + " ms"); 
    }
}
