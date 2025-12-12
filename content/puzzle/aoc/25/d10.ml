module IntSet = Set.Make(Int)

type rational = { num: int; den: int }

let gcd a b =
  let rec aux a b = if b = 0 then abs a else aux b (a mod b) in
  aux a b

let rat_make n d =
  if d = 0 then failwith "division by zero"
  else
    let g = gcd n d in
    let sign = if d < 0 then -1 else 1 in
    { num = sign * n / g; den = sign * d / g }

let rat_of_int n = { num = n; den = 1 }
let rat_zero = { num = 0; den = 1 }
let rat_one = { num = 1; den = 1 }
let rat_neg r = { num = -r.num; den = r.den }
let rat_is_zero r = r.num = 0
let rat_ge_zero r = r.num >= 0
let rat_is_int r = r.den = 1 || r.num mod r.den = 0
let rat_to_int r = r.num / r.den

let rat_add a b = rat_make (a.num * b.den + b.num * a.den) (a.den * b.den)
let rat_sub a b = rat_make (a.num * b.den - b.num * a.den) (a.den * b.den)
let rat_mul a b = rat_make (a.num * b.num) (a.den * b.den)
let rat_div a b = rat_make (a.num * b.den) (a.den * b.num)
let rat_le a b = a.num * b.den <= b.num * a.den

let rat_floor r =
  if r.num >= 0 then r.num / r.den
  else (r.num - r.den + 1) / r.den

let rat_ceil r =
  if r.num >= 0 then (r.num + r.den - 1) / r.den
  else r.num / r.den

(* parsing *)
let parse_line s =
  let pat_start = String.index s '[' + 1 in
  let pat_end = String.index s ']' in
  let pattern = String.sub s pat_start (pat_end - pat_start) in

  let parse_nums str =
    String.split_on_char ',' str
    |> List.map String.trim
    |> List.map int_of_string
  in

  let rec find_buttons acc rest =
    match String.index_opt rest '(' with
    | None -> List.rev acc
    | Some i ->
        let j = String.index rest ')' in
        let nums = parse_nums (String.sub rest (i+1) (j-i-1)) in
        find_buttons (nums :: acc) (String.sub rest (j+1) (String.length rest - j - 1))
  in
  let buttons = find_buttons [] (String.sub s pat_end (String.length s - pat_end)) in

  let t_start = String.index s '{' + 1 in
  let t_end = String.index s '}' in
  let targets = parse_nums (String.sub s t_start (t_end - t_start)) in

  (pattern, buttons, targets)

(* part 1: BFS over XOR states *)
let pattern_to_mask pat =
  String.to_seqi pat
  |> Seq.fold_left (fun acc (i, c) ->
    if c = '#' then acc lor (1 lsl i) else acc) 0

let button_to_mask btn =
  List.fold_left (fun acc i -> acc lxor (1 lsl i)) 0 btn

let min_press_lights target masks =
  if target = 0 then 0
  else begin
    let seen = Hashtbl.create 1024 in
    Hashtbl.add seen 0 ();
    let q = Queue.create () in
    Queue.add (0, 0) q;
    let rec bfs () =
      let state, dist = Queue.take q in
      let rec try_masks = function
        | [] -> bfs ()
        | m :: ms ->
            let nxt = state lxor m in
            if nxt = target then dist + 1
            else if not (Hashtbl.mem seen nxt) then begin
              Hashtbl.add seen nxt ();
              Queue.add (nxt, dist + 1) q;
              try_masks ms
            end else try_masks ms
      in
      try_masks masks
    in
    bfs ()
  end

(* part 2: gaussian elimination *)
let gauss_jordan a b =
  let rows = Array.length a in
  let cols = if rows > 0 then Array.length a.(0) else 0 in
  let mat = Array.map Array.copy a in
  let rhs = Array.copy b in
  let pivots = ref [] in
  let r = ref 0 in
  for c = 0 to cols - 1 do
    if !r < rows then begin
      let pivot_row = ref None in
      for i = !r to rows - 1 do
        if !pivot_row = None && not (rat_is_zero mat.(i).(c)) then
          pivot_row := Some i
      done;
      match !pivot_row with
      | None -> ()
      | Some pr ->
          let tmp = mat.(!r) in mat.(!r) <- mat.(pr); mat.(pr) <- tmp;
          let tmp = rhs.(!r) in rhs.(!r) <- rhs.(pr); rhs.(pr) <- tmp;

          let scale = mat.(!r).(c) in
          for j = 0 to cols - 1 do
            mat.(!r).(j) <- rat_div mat.(!r).(j) scale
          done;
          rhs.(!r) <- rat_div rhs.(!r) scale;

          for i = 0 to rows - 1 do
            if i <> !r && not (rat_is_zero mat.(i).(c)) then begin
              let factor = mat.(i).(c) in
              for j = 0 to cols - 1 do
                mat.(i).(j) <- rat_sub mat.(i).(j) (rat_mul factor mat.(!r).(j))
              done;
              rhs.(i) <- rat_sub rhs.(i) (rat_mul factor rhs.(!r))
            end
          done;
          pivots := c :: !pivots;
          incr r
    end
  done;
  (mat, rhs, List.rev !pivots)

let solve_square mat rhs =
  let n = Array.length mat in
  let mat2, rhs2, _ = gauss_jordan mat rhs in
  let sol = Array.make n rat_zero in
  try
    for i = n - 1 downto 0 do
      let lead = ref None in
      for j = 0 to n - 1 do
        if !lead = None && not (rat_is_zero mat2.(i).(j)) then
          lead := Some j
      done;
      match !lead with
      | None -> raise Exit
      | Some l ->
          let rest = ref rat_zero in
          for j = l + 1 to n - 1 do
            rest := rat_add !rest (rat_mul mat2.(i).(j) sol.(j))
          done;
          sol.(l) <- rat_sub rhs2.(i) !rest
    done;
    Some (Array.to_list sol)
  with Exit -> None

let rec combinations n k =
  if k = 0 then [[]]
  else if n < k then []
  else
    let with_n = List.map (fun c -> (n-1) :: c) (combinations (n-1) (k-1)) in
    let without_n = combinations (n-1) k in
    with_n @ without_n

let bounds_for_free constraints f =
  let subsets = combinations (List.length constraints) f in
  let mins = Array.make f None in
  let maxs = Array.make f rat_zero in
  List.iter (fun subset ->
    let mat = Array.of_list (List.map (fun i -> Array.of_list (fst (List.nth constraints i))) subset) in
    let rhs = Array.of_list (List.map (fun i -> snd (List.nth constraints i)) subset) in
    match solve_square mat rhs with
    | None -> ()
    | Some sol ->
        let feasible = List.for_all (fun (c, b) ->
          let dot = List.fold_left2 (fun acc ci si -> rat_add acc (rat_mul ci si)) rat_zero c sol in
          rat_le dot b
        ) constraints in
        if feasible then begin
          List.iteri (fun j v ->
            mins.(j) <- (match mins.(j) with
              | None -> Some v
              | Some m -> Some (if rat_le v m then v else m));
            if rat_le maxs.(j) v then maxs.(j) <- v
          ) sol
        end
  ) subsets;
  let mins_list = Array.to_list (Array.map (function Some m -> m | None -> rat_zero) mins) in
  let maxs_list = Array.to_list maxs in
  (mins_list, maxs_list)

let min_press_jolts buttons targets =
  let m = List.length targets in
  let n = List.length buttons in
  let a = Array.init m (fun i ->
    Array.of_list (List.map (fun btn ->
      if List.mem i btn then rat_one else rat_zero
    ) buttons)
  ) in
  let b = Array.of_list (List.map rat_of_int targets) in
  let a_red, b_red, pivots = gauss_jordan a b in
  let free = List.filter (fun c -> not (List.mem c pivots)) (List.init n Fun.id) in
  let f_count = List.length free in

  if f_count = 0 then begin
    let pivot_vals = List.mapi (fun r _ -> b_red.(r)) pivots in
    if List.for_all (fun v -> rat_ge_zero v && rat_is_int v) pivot_vals then
      List.fold_left (+) 0 (List.map rat_to_int pivot_vals)
    else failwith "no feasible solution"
  end else begin
    let row_info = List.mapi (fun r _ ->
      (b_red.(r), List.map (fun f -> a_red.(r).(f)) free)
    ) pivots in

    let constraints = List.map (fun (b_val, coeff) -> (coeff, b_val)) row_info in
    let neg_constraints = List.init f_count (fun j ->
      (List.init f_count (fun k -> if k = j then rat_neg rat_one else rat_zero), rat_zero)
    ) in
    let all_constraints = constraints @ neg_constraints in

    let mins, maxs = bounds_for_free all_constraints f_count in
    let bounds = List.map2 (fun mi mx ->
      (max 0 (rat_ceil mi), rat_floor mx)
    ) mins maxs in

    let rec search idx vals =
      if idx = f_count then begin
        let pivot_vals = List.map (fun (b_val, coeff) ->
          let sum = List.fold_left2 (fun acc c v -> rat_add acc (rat_mul c (rat_of_int v))) rat_zero coeff vals in
          rat_sub b_val sum
        ) row_info in
        if List.for_all (fun v -> rat_ge_zero v && rat_is_int v) pivot_vals then
          List.fold_left (+) 0 vals + List.fold_left (+) 0 (List.map rat_to_int pivot_vals)
        else max_int / 4
      end else begin
        let lo, hi = List.nth bounds idx in
        let rec try_vals v best =
          if v > hi then best
          else try_vals (v + 1) (min best (search (idx + 1) (vals @ [v])))
        in
        try_vals lo (max_int / 4)
      end
    in
    search 0 []
  end

let () =
  let ic = open_in "d10.txt" in
  let rec read_lines acc =
    try read_lines (input_line ic :: acc)
    with End_of_file -> List.rev acc
  in
  let lines = List.filter (fun s -> String.trim s <> "") (read_lines []) in
  close_in ic;

  let p1 = List.fold_left (fun acc ln ->
    let pat, btns, _ = parse_line ln in
    let target = pattern_to_mask pat in
    let masks = List.map button_to_mask btns in
    acc + min_press_lights target masks
  ) 0 lines in

  let p2 = List.fold_left (fun acc ln ->
    let _, btns, targets = parse_line ln in
    acc + min_press_jolts btns targets
  ) 0 lines in

  Printf.printf "p1: %d\n" p1;
  Printf.printf "p2: %d\n" p2
