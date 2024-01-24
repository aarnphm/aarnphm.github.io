class IntSet {
  int a[];
  int n;

  boolean intSetInvariantOK() {
    // true: 0 ≤ n ≤ a.length ∧ (∀ i ∈ 0 .. n - 1, j ∈ i + 1 .. n - 1 · a[i] ≠ a[j])
    // otherwise fail
    assert 0 <= n && n <= a.length;
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        assert a[i] != a[j];
      }
    }
    return true;
  }

  IntSet(int capacity) {
    a = new int[capacity];
    n = 0;
    assert intSetInvariantOK();
  }

  boolean loopInvariantOK(int x, int i) {
    for (int j = 0; j < i; j++) {
      assert a[j] != x;
    }
    return true;
  }

  void add(int x) {
    int i = 0;
    assert loopInvariantOK(x, i);
    while (i < n && a[i] != x) {
      i += 1;
      assert loopInvariantOK(x, i);
    }
    if (i == n) {
      a[n] = x;
      n += 1;
    }
    assert intSetInvariantOK();
  }

  boolean has(int x) {
    int i = 0;
    assert loopInvariantOK(x, i);
    while (i < n && a[i] != x) {
      i += 1;
      assert loopInvariantOK(x, i);
    }
    return i < n;
  }
}

class MaxIntSet extends IntSet {
  int m;

  boolean maxIntSetInvariantOK() {
    assert intSetInvariantOK();
    if (n > 0) {
      int max = a[0];
      for (int i = 1; i < n; i++) {
        max = Math.max(max, a[i]);
      }
      assert m == max;
    }
    return true;
  }

  MaxIntSet(int capacity) {
    super(capacity);
    assert maxIntSetInvariantOK();
  }

  void add(int x) {
    super.add(x);
    if (n == 1)
      m = x;
    else
      m = m > x ? m : x;
    assert maxIntSetInvariantOK();
  }

  int maximum() {
    return m;
  }
}

class TestMaxIntSet {
  public static void main(String[] args) {
    MaxIntSet s = new MaxIntSet(3);
    s.add(5);
    s.add(7);
    System.out.println(s.maximum());
  }
}
