let
  lib = import <nixpkgs/lib>;

  raw = builtins.readFile ./d9.txt;

  lines = lib.filter (x: x != "") (lib.splitString "\n" raw);

  parse = line: let
    parts = lib.splitString "," line;
  in {
    x = lib.strings.toInt (builtins.elemAt parts 0);
    y = lib.strings.toInt (builtins.elemAt parts 1);
  };

  points = map parse lines;

  n = builtins.length points;
  abs = x:
    if x < 0
    then -x
    else x;
  min = a: b:
    if a < b
    then a
    else b;
  max = a: b:
    if a > b
    then a
    else b;

  area = p1: p2: (abs (p2.x - p1.x) + 1) * (abs (p2.y - p1.y) + 1);

  allAreas = lib.concatMap (p1: map (p2: area p1 p2) points) points;

  p1 = lib.foldl' lib.max 0 allAreas;

  # edges of the polygon loop
  getPair = i: {
    a = builtins.elemAt points i;
    b = builtins.elemAt points (lib.mod (i + 1) n);
  };
  edges = map getPair (lib.range 0 (n - 1));

  # rectangle is valid iff no polygon edge cuts through its interior
  valid = pa: pb: let
    x1 = min pa.x pb.x;
    x2 = max pa.x pb.x;
    y1 = min pa.y pb.y;
    y2 = max pa.y pb.y;

    cutThrough = edge: let
      horizontal = edge.a.y == edge.b.y;
      # horizontal edge
      ey = edge.a.y;
      ex1 = min edge.a.x edge.b.x;
      ex2 = max edge.a.x edge.b.x;
      # vertical edge
      ex = edge.a.x;
      ey1 = min edge.a.y edge.b.y;
      ey2 = max edge.a.y edge.b.y;

      hCross = horizontal && y1 < ey && ey < y2 && (max x1 ex1) < (min x2 ex2);
      vCross = (!horizontal) && x1 < ex && ex < x2 && (max y1 ey1) < (min y2 ey2);
    in
      hCross || vCross;
  in
    # require at least 3 polygon edges in rectangle's coordinate range
    !(lib.any cutThrough edges);

  validAreas =
    lib.concatMap (
      pa:
        lib.concatMap (
          pb:
            if valid pa pb
            then [(area pa pb)]
            else []
        )
        points
    )
    points;

  p2 = lib.foldl' max 0 validAreas;
in {
  inherit p1 p2;
}
