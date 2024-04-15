// A bag (multiset) is a data structure in which the order of elements does not matter,
// like in a set, but in which an element can occur multiple times, like in a sequence.
// Here, we assume that all elements are integers:
//
// insert(e) adds integer e to the bag,
// delete(e) removes one occurrence of e from the bag, if there is one, otherwise does nothing,
// has(e) returns true if e occurs at least once in the bag.
//
// The implementation below uses a singly linked list of nodes with a pointer to the
// first and last node. A bag always has an "empty" first node that does not contain an element but simplifies deletion.
//
// insert(e) adds a node with e at the end of the list,
// delete(e) removes the first node with e, assuming that a node with e exists, otherwise does nothing,
// has(e) returns true if a node with e is found.
//
class Node {
  int e;
  Node next = null;
}

class Bag {
  Node first, last; // first points to empty node
  // invariant: first != null && last != null && last.next == null &&
  // from first, following the next fields, last can be reached

  void print() {
    Node n = first;
    while (n != last) {
      n = n.next;
      System.out.print(n.e + " ");
    }
    System.out.println();
  }

  Bag() {
    first = new Node();
    last = first;
  }

  void insert(int e) {
    Node n = new Node();
    n.e = e;
    last.next = n;
    last = n;
  }

  void delete(int e) {
    Node n = first;
    while (n != last && n.next.e != e)
      n = n.next;
    if (n != last) {
      if (n.next == last)
        last = n;
      n.next = n.next.next;
    }
  }

  boolean has(int e) {
    Node n = first;
    while (n != last && n.next.e != e)
      n = n.next;
    return n != last;
  }
}

class TestBag {
  public static void main(String[] args) {
    Bag b = new Bag();
    b.insert(5);
    b.print();
    b.insert(7);
    b.print();
    b.delete(7);
    b.print();
    System.out.println(b.has(5)); // false
    System.out.println(b.has(7)); // true
    System.out.println(b.has(3)); // false
  }
}

// Not thread-safe because
// 1. No synchrnozation mechanism, meaning control from multiple threads to
// access its shared resource (`first` and `last`) are not found. Therefore,
// multiple
// threads can simultaneously access and modify the shared resources
// 2. Race condition: With two or more threads access shared resource
// concurrently and
// at least one threads modify the data, then race condition occures:
// - If an _inserter_ and a _deleter_ access the bag concurrently, then the
// _deleter_
// might try to delete the node that in the process of being added, which can
// throw `NullPointerException`
// 3. Without synchrnozation, there is no guarantee that changes made by one
// threads is reflected to the
// other threads, which leads to inconsistent state of the shared resource
//
// Example
// - Suppose thread A (inserter) execute `insert(10)` and just about to set
// `last.next = n`,
// where `n` is the new node with element `10`
// - Concurrently, thread B (deleter) is executing `delete(10)` and has just
// check that
// `n.next.e != 10` (since thread A is yet to update `last.next`). Now Thread B
// proceeds past the while
// loop and reaches the if condition.
// - Thread A now completes its insertion, setting `last.next=n` and `last=n`
// - Meanwhile, Thread B proceeds with the deletion logic, which operate on
// outdated reference of `last`,
// leading to incorrect modification of list structure
