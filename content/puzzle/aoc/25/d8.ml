let read_input filename =
  let ic = open_in filename in
  let rec read_lines acc =
    try
      let line = input_line ic in
      read_lines (line :: acc)
    with
    | End_of_file ->
      close_in ic;
      List.rev acc
  in
  read_lines []
;;

let parse_points lines =
  List.filter_map
    (fun line ->
       match String.split_on_char ',' line with
       | [ x; y; z ] -> Some (int_of_string x, int_of_string y, int_of_string z)
       | _ -> None)
    lines
;;

let distance (x1, y1, z1) (x2, y2, z2) =
  let dx = float_of_int (x2 - x1) in
  let dy = float_of_int (y2 - y1) in
  let dz = float_of_int (z2 - z1) in
  sqrt ((dx *. dx) +. (dy *. dy) +. (dz *. dz))
;;

module UnionFind = struct
  type t =
    { parent : int array
    ; rank : int array
    ; size : int array
    }

  let create n =
    { parent = Array.init n (fun i -> i); rank = Array.make n 0; size = Array.make n 1 }
  ;;

  let rec find uf x =
    if uf.parent.(x) = x
    then x
    else (
      uf.parent.(x) <- find uf uf.parent.(x);
      uf.parent.(x))
  ;;

  let union uf x y =
    let rx = find uf x in
    let ry = find uf y in
    if rx = ry
    then false
    else (
      if uf.rank.(rx) < uf.rank.(ry)
      then (
        uf.parent.(rx) <- ry;
        uf.size.(ry) <- uf.size.(ry) + uf.size.(rx))
      else if uf.rank.(rx) > uf.rank.(ry)
      then (
        uf.parent.(ry) <- rx;
        uf.size.(rx) <- uf.size.(rx) + uf.size.(ry))
      else (
        uf.parent.(ry) <- rx;
        uf.size.(rx) <- uf.size.(rx) + uf.size.(ry);
        uf.rank.(rx) <- uf.rank.(rx) + 1);
      true)
  ;;

  let sizes uf n =
    let sizes = Hashtbl.create n in
    for i = 0 to n - 1 do
      let root = find uf i in
      Hashtbl.replace sizes root uf.size.(root)
    done;
    Hashtbl.fold (fun _ size acc -> size :: acc) sizes []
  ;;
end

let p1 points =
  let n = List.length points in
  let arr = Array.of_list points in
  let edges = ref [] in
  for i = 0 to n - 2 do
    for j = i + 1 to n - 1 do
      let d = distance arr.(i) arr.(j) in
      edges := (d, i, j) :: !edges
    done
  done;
  let sorted = List.sort (fun (d1, _, _) (d2, _, _) -> compare d1 d2) !edges in
  let uf = UnionFind.create n in
  let rec process count edges =
    if count = 1000
    then ()
    else (
      match edges with
      | [] -> ()
      | (_, i, j) :: rest ->
        let _ = UnionFind.union uf i j in
        process (count + 1) rest)
  in
  process 0 sorted;
  let sizes = UnionFind.sizes uf n in
  let ssorted = List.sort (fun a b -> compare b a) sizes in
  match ssorted with
  | a :: b :: c :: _ -> a * b * c
  | _ -> 0
;;

let p2 points =
  let n = List.length points in
  let arr = Array.of_list points in
  let edges = ref [] in
  for i = 0 to n - 2 do
    for j = i + 1 to n - 1 do
      let d = distance arr.(i) arr.(j) in
      edges := (d, i, j) :: !edges
    done
  done;
  let sorted = List.sort (fun (d1, _, _) (d2, _, _) -> compare d1 d2) !edges in
  let uf = UnionFind.create n in
  let components = ref n in
  let rec process edges =
    match edges with
    | [] -> 0
    | (_, i, j) :: rest ->
      if UnionFind.union uf i j
      then (
        components := !components - 1;
        if !components = 1
        then (
          let x1, _, _ = arr.(i) in
          let x2, _, _ = arr.(j) in
          x1 * x2)
        else process rest)
      else process rest
  in
  process sorted
;;

let () =
  let lines = read_input "d8.txt" in
  let points = parse_points lines in
  Printf.printf "p1: %d\n" (p1 points);
  Printf.printf "p2: %d\n" (p2 points)
;;
