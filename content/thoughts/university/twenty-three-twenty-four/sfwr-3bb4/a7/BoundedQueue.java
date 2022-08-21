class BoundedQueue<T> {
  T[] buf;
  int in = 0, out = 0, n = 0;

  BoundedQueue(int cap) {
    buf = (T[]) new Object[cap];
    assert buf != null : "Buffer pointer is null!";
  }

  void put(T x) {
    assert n < buf.length : "Queue is full!";
    assert buf != null : "Buffer pointer is null!";
    buf[in] = x;
    in = (in + 1) % buf.length;
    n += 1;
    assert in >= 0 && in < buf.length : "Invalid 'in' index!";
    assert n >= 0 && n <= buf.length : "Invalid size 'n'!";
  }

  T get() {
    assert n > 0 : "Queue is empty!";
    assert buf != null : "Buffer pointer is null!";
    T x = buf[out];
    out = (out + 1) % buf.length;
    n -= 1;
    assert out >= 0 && out < buf.length : "Invalid 'out' index!";
    assert n >= 0 && n <= buf.length : "Invalid size 'n'!";
    return x;
  }

  int size() {
    assert n >= 0 && n <= buf.length : "Invalid size 'n'!";
    return n;
  }
}
