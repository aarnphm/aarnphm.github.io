import java.util.Random;
import java.util.regex.Pattern;
import java.util.concurrent.locks.ReentrantLock;

class Node {
  int e;
  Node next = null;
}

class Bag {
  // invariant:
  // - first != null && last != null && last.next == null
  // - from first, following the next fields, last can be reached
  // - lock not null and used to control access to
  // last and first

  Node first, last; // first points to empty node
  String log = "";
  // Single lock for synchronization
  private final ReentrantLock lock = new ReentrantLock();

  Bag() {
    first = new Node();
    last = first;
  }

  void insert(int e) {
    lock.lock(); // Lock for all operations
    try {
      synchronized (log) {
        log += "I";
      }
      Node n = new Node();
      n.e = e;
      last.next = n;
      last = n;
      synchronized (log) {
        log += "i";
      }
    } finally {
      lock.unlock(); // Unlock
    }
  }

  void delete(int e) {
    lock.lock(); // Lock for all operations
    try {
      synchronized (log) {
        log += "D";
      }
      Node n = first;
      while (n != last && n.next.e != e)
        n = n.next;
      if (n != last) {
        if (n.next == last)
          last = n;
        n.next = n.next.next;
      }
      synchronized (log) {
        log += "d";
      }
    } finally {
      lock.unlock(); // Unlock
    }
  }

  boolean has(int e) {
    lock.lock(); // Lock for all operations
    try {
      synchronized (log) {
        log += "H";
      }
      Node n = first;
      while (n != last && n.next.e != e)
        n = n.next;
      boolean found = n != last;
      synchronized (log) {
        log += "h";
      }
      return found;
    } finally {
      lock.unlock(); // Unlock
    }
  }

  void print() {
    Node n = first;
    while (n != last) {
      n = n.next;
      System.out.println(n.e);
    }
  }

}

class Inserter extends Thread {
  Bag b;
  int s;

  Inserter(Bag b, int s) {
    this.b = b;
    this.s = s;
  }

  public void run() {
    Random r = new Random();
    for (int i = 0; i < s; i++) {
      b.insert(r.nextInt(100));
    }
  }
}

class Deleter extends Thread {
  Bag b;
  int s;

  Deleter(Bag b, int s) {
    Random r = new Random();
    this.b = b;
    this.s = s;
  }

  public void run() {
    Random r = new Random();
    for (int i = 0; i < s; i++) {
      b.delete(r.nextInt(100));
    }
  }
}

class Searcher extends Thread {
  Bag b;
  int s;

  Searcher(Bag b, int s) {
    this.b = b;
    this.s = s;
  }

  public void run() {
    Random r = new Random();
    for (int i = 0; i < s; i++) {
      b.has(r.nextInt(100));
    }
  }
}

class TestBag {
  public static void main(String[] args) {
    final int P = 20; // number of inserter, deleter, and searcher threads
    final int S = 100; // number of repetitions by each thread
    Bag b = new Bag();
    Inserter[] in = new Inserter[P];
    Deleter[] dl = new Deleter[P];
    Searcher[] sr = new Searcher[P];
    for (int i = 0; i < P; i++) {
      in[i] = new Inserter(b, S);
      dl[i] = new Deleter(b, S);
      sr[i] = new Searcher(b, S);
    }
    for (int i = 0; i < P; i++) {
      in[i].start();
      dl[i].start();
      sr[i].start();
    }
    for (int i = 0; i < P; i++)
      try {
        in[i].join();
        dl[i].join();
        sr[i].join();
      } catch (Exception x) {
      }
    assert !Pattern.matches(".*I[IDd].*", b.log) : "An inserter must exclude other inserters and deleters";
    assert !Pattern.matches(".*D[DIiHh].*", b.log) : "A deleter must exclude other deleters, inserters, and searchers";
    assert !Pattern.matches(".*H[Dd].*", b.log) : "A searcher must exclude deleters";
    if (Pattern.matches(".*I[^i]*H.*", b.log))
      System.out.println("Searcher concurrent with Inserter");
    if (Pattern.matches(".*H[^h]*I.*", b.log))
      System.out.println("Inserter concurrent with Searcher");
    if (Pattern.matches(".*H[^h]*H.*", b.log))
      System.out.println("Searcher concurrent with Searcher");
    System.out.println(b.log);
  }
}
