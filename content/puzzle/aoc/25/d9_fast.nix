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

  # edges of the polygon loop, separated by orientation
  mkEdges =
    lib.imap0 (i: _: let
      a = builtins.elemAt points i;
      b = builtins.elemAt points (lib.mod (i + 1) n);
    in {
      inherit a b;
      horiz = a.y == b.y;
    })
    points;

  # group horizontal edges by y, vertical edges by x
  hEdges = lib.filter (e: e.horiz) mkEdges;
  vEdges = lib.filter (e: !e.horiz) mkEdges;

  # sorted arrays for binary search
  sortedYs = lib.sort (a: b: a < b) (lib.unique (map (e: e.a.y) hEdges));
  sortedXs = lib.sort (a: b: a < b) (lib.unique (map (e: e.a.x) vEdges));
  arrYs = builtins.listToAttrs (lib.imap0 (i: v: {
      name = toString i;
      value = v;
    })
    sortedYs);
  arrXs = builtins.listToAttrs (lib.imap0 (i: v: {
      name = toString i;
      value = v;
    })
    sortedXs);
  lenYs = builtins.length sortedYs;
  lenXs = builtins.length sortedXs;

  # binary search: find first index where arr[i] > val (upper bound)
  bisectRight = arr: len: val: let
    go = lo: hi:
      if lo >= hi
      then lo
      else let
        mid = (lo + hi) / 2;
      in
        if arr.${toString mid} <= val
        then go (mid + 1) hi
        else go lo mid;
  in
    go 0 len;

  # binary search: find first index where arr[i] >= val (lower bound)
  bisectLeft = arr: len: val: let
    go = lo: hi:
      if lo >= hi
      then lo
      else let
        mid = (lo + hi) / 2;
      in
        if arr.${toString mid} < val
        then go (mid + 1) hi
        else go lo mid;
  in
    go 0 len;

  # get slice of sorted list by index range
  sliceYs = lo: hi: lib.sublist lo (hi - lo) sortedYs;
  sliceXs = lo: hi: lib.sublist lo (hi - lo) sortedXs;

  # group edges by coordinate for fast lookup
  hByY = lib.groupBy (e: toString e.a.y) hEdges;
  vByX = lib.groupBy (e: toString e.a.x) vEdges;

  # check if horizontal edge cuts through rectangle
  hCuts = rx1: rx2: edge: let
    ex1 = min edge.a.x edge.b.x;
    ex2 = max edge.a.x edge.b.x;
  in
    (max rx1 ex1) < (min rx2 ex2);

  # check if vertical edge cuts through rectangle
  vCuts = ry1: ry2: edge: let
    ey1 = min edge.a.y edge.b.y;
    ey2 = max edge.a.y edge.b.y;
  in
    (max ry1 ey1) < (min ry2 ey2);

  # rectangle valid iff no edge cuts through
  valid = pa: pb: let
    x1 = min pa.x pb.x;
    x2 = max pa.x pb.x;
    y1 = min pa.y pb.y;
    y2 = max pa.y pb.y;

    # binary search to find y-coords in [y1, y2]
    yLoInc = bisectLeft arrYs lenYs y1;  # first y >= y1
    yHiInc = bisectRight arrYs lenYs y2; # first y > y2
    relevantYsInc = sliceYs yLoInc yHiInc;

    # binary search to find x-coords in [x1, x2]
    xLoInc = bisectLeft arrXs lenXs x1;
    xHiInc = bisectRight arrXs lenXs x2;
    relevantXsInc = sliceXs xLoInc xHiInc;

    # require at least 3 edges in rectangle's coordinate range
    numRelevant = (builtins.length relevantYsInc) + (builtins.length relevantXsInc);

    # binary search to find y-coords in (y1, y2) for cut-through check
    yLo = bisectRight arrYs lenYs y1; # first y > y1
    yHi = bisectLeft arrYs lenYs y2; # first y >= y2
    relevantYs = sliceYs yLo yHi;
    hCut =
      lib.any (
        y:
          lib.any (hCuts x1 x2) (hByY.${toString y} or [])
      )
      relevantYs;

    # binary search to find x-coords in (x1, x2) for cut-through check
    xLo = bisectRight arrXs lenXs x1;
    xHi = bisectLeft arrXs lenXs x2;
    relevantXs = sliceXs xLo xHi;
    vCut =
      lib.any (
        x:
          lib.any (vCuts y1 y2) (vByX.${toString x} or [])
      )
      relevantXs;
  in
    numRelevant >= 3 && !hCut && !vCut;

  # generate unique pairs (i < j) with areas, sort descending
  pairsWithArea = lib.concatMap (
    i:
      map (j: rec {
        pa = builtins.elemAt points i;
        pb = builtins.elemAt points j;
        a = area pa pb;
      }) (lib.range (i + 1) (n - 1))
  ) (lib.range 0 (n - 2));

  sortedPairs = lib.sort (a: b: a.a > b.a) pairsWithArea;

  # early termination
  firstValid = lib.findFirst (p: valid p.pa p.pb) null sortedPairs;

  p2 =
    if firstValid == null
    then 0
    else firstValid.a;
in {
  inherit p1 p2;
}
