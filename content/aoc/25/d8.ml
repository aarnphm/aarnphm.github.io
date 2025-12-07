let read_input filename =
  let ic = open_in filename in
  let rec read_lines acc =
    try
      let line = input_line ic in
      read_lines (line :: acc)
    with End_of_file ->
      close_in ic;
      List.rev acc
  in
  read_lines []

let read_grid filename =
  let lines = read_input filename in
  Array.of_list (List.map (fun s -> Array.init (String.length s) (String.get s)) lines)

let p1 _grid = 0

let p2 _grid = 0

let () =
  let grid = read_grid "d8.txt" in
  Printf.printf "p1: %d\n" (p1 grid);
  Printf.printf "p2: %d\n" (p2 grid)
