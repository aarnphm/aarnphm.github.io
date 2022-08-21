import java.util.ArrayList;
import java.util.List;

// NOTE: We will need to use Encapsulation to ensure the invariants of the group
// This requires the following:
// 1. private constructors for shape: To make sure shapes can't be created outside of the factory methods
// 2. construct factory methods for each shape: provide a static factory methods to create shape. This return the base type `Shape` instead of their specific type. By doing so, we won't keep a direct reference to the created shape.
// 3. Deep copy in group: Shape will now implement a deepCopy method. If there is even a reference, then it won't affect the shape inside the group

abstract class Shape {
  int x0, y0, x1, y1; // (x0, y0) lower left, (x1, y1) upper right

  boolean shapeInvariantOK() {
    return x0 <= x1 && y0 <= y1;
  }

  @Override
  public String toString() {
    return "(" + x0 + ", " + y0 + ") -> (" + x1 + ", " + y1 + ")";
  }

  protected Shape(int x0, int y0, int x1, int y1) {
    assert x0 <= x1 && y0 <= y1 : "Invalid coordinates!";
    this.x0 = x0;
    this.y0 = y0;
    this.x1 = x1;
    this.y1 = y1;
  }

  void move(int dx, int dy) {
    x0 += dx;
    y0 += dy;
    x1 += dx;
    y1 += dy;
    assert shapeInvariantOK() : "Invalid move!";
  }

  Rectangle boundingBox() {
    return (Rectangle) Rectangle.create(x0, y0, x1 - x0, y1 - y0);
  }

  // Deep copy method to be implemented by subclasses
  protected abstract Shape deepCopy();
}

class Line extends Shape {
  int x, y, u, v; // from (x, y) to (u, v)

  boolean lineInvariantOK() {
    return (x != u || y != v) &&
        x0 == Math.min(x, u) &&
        y0 == Math.min(y, v) &&
        x1 == Math.max(x, u) &&
        y1 == Math.max(y, v) &&
        shapeInvariantOK();
  }

  private Line(int x, int y, int u, int v) {
    super(Math.min(x, u), Math.min(y, v), Math.max(x, u), Math.max(y, v));
    this.x = x;
    this.y = y;
    this.u = u;
    this.v = v;
    assert lineInvariantOK() : "Invalid line!";
  }

  void move(int dx, int dy) {
    super.move(dx, dy);
    x += dx;
    y += dy;
    u += dx;
    v += dy;
    assert lineInvariantOK() : "Invalid move!";
  }

  // Factory method to create a Line
  public static Shape create(int x, int y, int u, int v) {
    return new Line(x, y, u, v);
  }

  @Override
  protected Shape deepCopy() {
    return new Line(x, y, u, v);
  }
}

class Rectangle extends Shape {
  int x, y, w, h;

  boolean rectangleInvariantOK() {
    return w > 0 && h > 0 &&
        x0 == x && y0 == y &&
        x1 == x + w && y1 == y + h &&
        shapeInvariantOK();
  }

  private Rectangle(int x, int y, int w, int h) {
    super(x, y, x + w, y + h);
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    assert rectangleInvariantOK() : "Invalid rectangle!";
  }

  void move(int dx, int dy) {
    super.move(dx, dy);
    x += dx;
    y += dy;
    assert rectangleInvariantOK() : "Invalid move!";
  }

  // Factory method to create a Rectangle
  public static Shape create(int x, int y, int w, int h) {
    return new Rectangle(x, y, w, h);
  }

  @Override
  protected Shape deepCopy() {
    return new Rectangle(x, y, w, h);
  }
}

class Group extends Shape {
  List<Shape> elts = new ArrayList<Shape>();

  boolean groupInvariantOK() {
    if (elts.isEmpty() || elts.contains(null))
      return false;
    int minX = Integer.MAX_VALUE, minY = Integer.MAX_VALUE;
    int maxX = Integer.MIN_VALUE, maxY = Integer.MIN_VALUE;
    for (Shape s : elts) {
      if (s.x0 < minX)
        minX = s.x0;
      if (s.y0 < minY)
        minY = s.y0;
      if (maxX < s.x1)
        maxX = s.x1;
      if (maxY < s.y1)
        maxY = s.y1;
    }
    return minX == x0 && minY == y0 && maxX == x1 && maxY == y1 && shapeInvariantOK();
  }

  private Group(Shape s) {
    super(s.x0, s.y0, s.x1, s.y1);
    elts.add(s);
    assert groupInvariantOK() : "Invalid group!";
  }

  void move(int dx, int dy) {
    super.move(dx, dy);
    for (Shape s : elts)
      s.move(dx, dy);
    assert groupInvariantOK() : "Invalid move!";
  }

  void add(Shape s) {
    Shape copy = s.deepCopy();
    assert copy != null : "Cannot add null shape!";
    elts.add(copy);
    if (copy.x0 < x0)
      x0 = copy.x0;
    if (copy.y0 < y0)
      y0 = copy.y0;
    if (x1 < copy.x1)
      x1 = copy.x1;
    if (y1 < copy.y1)
      y1 = copy.y1;
    assert groupInvariantOK() : "Invalid group after addition!";
  }

  // Factory method to create a Group
  public static Group create(Shape s) {
    return new Group(s.deepCopy());
  }

  @Override
  protected Shape deepCopy() {
    Group copy = new Group(elts.get(0));
    for (int i = 1; i < elts.size(); i++) {
      copy.add(elts.get(i));
    }
    return copy;
  }
}

class TestShapes {
  public static void main(String[] args) {
    Shape r = Rectangle.create(1, 1, 5, 5);
    Shape l = Line.create(0, 0, 4, 4);
    Group g = Group.create(r);
    g.add(l);
    Rectangle bb = g.boundingBox();
    System.out.println(bb.toString());
    System.out.println(g.x0);
    System.out.println(g.x1);
    System.out.println(g.y0);
    System.out.println(g.y1);
    g.move(10, 10);
    System.out.println(g.x0);
    System.out.println(g.x1);
    System.out.println(g.y0);
    System.out.println(g.y1);
    r.move(3, 3); // BAD, breaks invariant of g, fixed by deepCopy and encapsulation
    l.move(-1, -1); // BAD
    assert g.groupInvariantOK();
  }
}
