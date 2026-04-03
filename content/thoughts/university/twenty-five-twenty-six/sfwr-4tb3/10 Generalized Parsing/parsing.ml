(* Emil Sekerinski, McMaster University, Fall 2025 *)

This file is meant as a template for cutting and pasting pieces.
Trying to run it will lead to (intentional) errors.
*)

(* Functional parser for sample grammar:

A = "a" A "c" | "b"

*)

let rec aa = function
  'a'::t -> (match aa t with 'c'::u -> u) | 'b'::t -> t
;;

aa ['a'; 'c'];;
aa ['a'; 'b'; 'a'];;
aa ['a'; 'b'; 'c'];;

(* Functional parser for expression grammar:

expression = term { ("+" | "-") term }
term       = factor { ( "*" | "/") factor }
factor     = id | id "(" exprList ")" | "(" expression ")"
exprList   = expression { "," expression }

Here, id is any lower case letter. The parser below additionally produces meaningful error messages.
*)

let rec expression s =
  match term s with
    | '+'::t -> expression t
    | '-'::t -> expression t
    | any -> any

and term s =
  match factor s with
    | '*'::t -> term t
    | '/'::t -> term t
    | any -> any

and factor = function
  | '('::t ->
      (match expression t with
        | ')'::u -> u
        | _ -> raise (Failure "missing )"))
  | id::'('::t when id >= 'a' && id <= 'z' ->
      (match exprList t with
        | ')'::u -> u
        | _ -> raise (Failure "missing )"))
  | id::t when id >= 'a' && id <= 'z' -> t
  | _::_ -> raise (Failure "id or ( expected")
  | [] -> raise (Failure "unexpected end")

and exprList s =
    match expression s with
      | ','::t -> exprList t
      | any -> any
;;

expression ['a'; '+'; '('; 'b'; '/'; 'c'; ')'];;
expression ['a'; '+'; 'b'; '/'; 'c'];;
expression ['a'; '*'; 'b'; '*'; 'c'];;
expression ['a'; '+'; '('; 'b'];;
expression ['1'; '+'; '*'];;
expression ['a'; '+'];;
expression ['a'; 'b'];;

(* Constructing the abstract syntax tree *)

type exp =
  | Sum of exp * exp
  | Diff of exp * exp
  | Prod of exp * exp
  | Quot of exp * exp
  | Id of char
  | Fun of char * exp list;;

let rec expression s =
  let rec moreterms = function
      | (p, '+'::t) -> let (q, u) = term t in moreterms (Sum (p, q), u)
      | (p, '-'::t) -> let (q, u) = term t in moreterms (Diff (p, q), u)
      | any -> any
  in moreterms (term s)

and term s =
  let rec morefactors = function
    | (p, '*'::t) -> let (q, u) = factor t in morefactors (Prod (p, q), u)
    | (p, '/'::t) -> let (q, u) = factor t in morefactors (Quot (p, q), u)
    | any -> any
  in morefactors (factor s)

and factor = function
  | '('::t ->
      (match expression t with
        | (p, ')'::u) -> (p, u)
        | _ -> raise (Failure "missing )"))
  | id::'('::t when id >= 'a' && id <= 'z' ->
      (match exprList t with
        | (p, ')'::u) -> (Fun(id, p), u)
        | _ -> raise (Failure "missing )"))
  | id::t when id >= 'a' && id <= 'z' -> (Id (id), t)
  | _::_ -> raise (Failure "id or ( expected")
  | [] -> raise (Failure "unexpected end")

and exprList s =
    match expression s with
      | (p, ','::t) -> let (q, n) = exprList t in (p::q, n)
      | (p, t) -> ([p], t)
;;

expression ['a'; '*'; '('; 'b'; '+'; 'c'; ')'];;
expression ['a'; '+'; 'f'; '('; 'x'; ','; 'y'; ')'];;
expression ['a'; '+'; '('; 'b'; '/'; 'c'; ')'];;
expression ['a'; '+'; 'b'; '/'; 'c'];;
expression ['a'; '*'; 'b'; '*'; 'c'];;
expression ['a'; '+'; '('; 'b'];;
expression ['1'; '+'; '*'];;
expression ['a'; '+'];;
expression ['a'; 'b'];;

(* Producing error messages with position in input. All parsing
functions are now of type source -> source, where source is a
a pair consisting of the input position and the remainder of
the input *)

type source = int * char list;;
exception Source_error of int * string ;;

let rec expression (p, s) =
  match term (p, s) with
    | q, '+'::t -> expression (q + 1, t)
    | q, '-'::t -> expression (q + 1, t)
    | any -> any

and term (p, s) =
  match factor (p, s) with
    | q, '*'::t -> term (q + 1, t)
    | q, '/'::t -> term (q + 1, t)
    | any -> any

and factor = function
  | p, '('::t ->
      (match expression (p + 1, t) with
        | q, ')'::u -> (q + 1, u)
        | q, _ -> raise (Source_error (q, "missing )")))
  | p, id::'('::t when id >= 'a' && id <= 'z' ->
      (match exprList (p + 2, t) with
        | r, ')'::u -> (r + 1, u)
        | r, _ -> raise (Source_error (r, "missing )")))
  | p, id::t when id >= 'a' && id <= 'z' -> (p + 1, t)
  | p, _::_ -> raise (Source_error (p, "id or ( expected"))
  | p, [] -> raise (Source_error (p, "unexpected end"))

and exprList (p, s) =
    match expression (p, s) with
      | q, ','::t -> exprList (q + 1, t)
      | any -> any
;;

let parse s =
  try
    match expression (1, s) with
      p, t -> print_string (string_of_int (p - 1) ^ " characters parsed\n")
  with Source_error (p, m) ->
    print_string ("error at " ^ string_of_int p ^ ": " ^ m ^ "\n")
;;

parse ['a'; '+'; '('; 'b'; '/'; 'c'; ')'];;
parse ['a'; '+'; 'b'; '/'; 'c'];;
parse ['a'; '*'; 'b'; '*'; 'c'];;
parse ['a'; '+'; '('; 'b'];;
parse ['1'; '+'; '*'];;
parse ['a'; '+'];;
parse ['a'; 'b'];;

(* With scanning:

  expression = term { ("+" | "-") term }
  term       = factor { ( "*" | "div" | "mod") factor }
  factor     = integer | id | "(" expression ")"

where symbols can be separa ted by blanks and

  integer = digit {digit}
  id      = letter {letter | digit}

Here letter is any lower case letter.
The function symbol is of type source -> symbol * source.
Parser functions are (still) of type source -> source.
*)

type symbol = LPAREN | RPAREN | PLUS | MINUS | TIMES | DIV | MOD |
  INT of int | ID of char list | EOF;;

let isDigit c = c >= '0' && c <= '9' ;;

let isLetter c = c >= 'a' && c <= 'z' ;;

let toDigit c = Char.code c - Char.code '0' ;;

let sep = function (* true if input is empty or starts with a separator *)
  | [] -> true
  | c::_ -> not (isDigit c || isLetter c);;

let rec symbol (l, s) =
  match s with
    | c::t when c <= ' ' -> symbol (l + 1, t)
    | '('::t -> (LPAREN, (l + 1, t))
    | ')'::t -> (RPAREN, (l + 1, t))
    | '+'::t -> (PLUS, (l + 1, t))
    | '-'::t -> (MINUS, (l + 1, t))
    | '*'::t -> (TIMES, (l + 1, t))
    | 'd'::'i'::'v'::t when sep t -> (DIV, (l + 3, t))
    | 'm'::'o'::'d'::t when sep t -> (MOD, (l + 3, t))
    | c::t when isDigit c ->
        let rec num n = function (* returns integer and remainder of source *)
          | l, d::u when isDigit d -> num (n * 10 + toDigit d) (l + 1, u)
          | any -> (n, any)
        in let (n, u) = num (toDigit c) (l + 1, t) in (INT n, u)
    | c::t when isLetter c ->
        let rec id s = function (* returns char list and remainder of source *)
          | l, d::u when isLetter d || isDigit d -> id (s @ [d]) (l + 1, u)
          | any -> (s, any)
        in let (r, u) = id [c] (l + 1, t) in (ID r, u)
    | _ -> (EOF, (l, s));;

let rec expression s =
  let t = term s in
  match symbol t with
    | PLUS, u -> expression u
    | MINUS, u -> expression u
    | _ -> t

and term s =
  let t = factor s in
  match symbol t with
    | TIMES, u -> term u
    | DIV, u -> term u
    | _ -> t

and factor s =
  match symbol s with
    | LPAREN, t ->
        let u = expression t in
        (match symbol u with
          | RPAREN, v -> v
          | _, (p, _) -> raise (Source_error (p, "missing )")))
    | INT n, t -> t
    | ID id, t -> t
    | EOF, (p, _) -> raise (Source_error (p, "unexpected end"))
    | _, (p, _) -> raise (Source_error (p, "id or num expected"))
;;

let parse s =
  try
    match expression (1, s) with
      p, t -> print_string (string_of_int (p - 1) ^ " characters parsed")
  with Source_error (p, m) ->
    print_string ("error at " ^ string_of_int p ^ ": " ^ m ^ "");;

parse ['a'; ' '; '+'; ' '; '('; 'b'; ' '; 'd'; 'i'; 'v'; ' '; 'c'; ')'];;
parse ['a'; '+'; 'b'; '*'; 'c'];;
parse ['1'; '2'; '*'; '3'; '4'; '5'; '*'; '6'];;
parse ['a'; '+'; '('; 'b'];;
parse ['1'; '+'; '*'];;
parse ['a'; '+'];;
parse ['a'; 'b'];;

(* Now with scanning, syntax tree, and error positions, with grammar:

  expression = term { ("+" | "-") term }
  term       = factor { ( "*" | "div") factor }
  factor     = integer | id | id "(" exprList ")" | "(" expression ")"
  exprList   = expression { "," expression }

where symbols can be separated by blanks and

  integer = digit {digit}
  id      = letter {letter | digit}

Here letter is any lower case letter.
The function symbol is of type source -> symbol * source.
Parser functions are  of type source -> expr * source.
*)

type exp =
  | Sum of exp * exp
  | Diff of exp * exp
  | Prod of exp * exp
  | Quot of exp * exp
  | Int of int
  | Id of char list
  | Fun of char list * exp list ;;

type symbol = COMMA | LPAREN | RPAREN | PLUS | MINUS | TIMES | DIV | MOD |
  INT of int | ID of char list | EOF ;;

let isDigit c = c >= '0' && c <= '9' ;;

let isLetter c = c >= 'a' && c <= 'z' ;;

let toDigit c = Char.code c - Char.code '0' ;;

let sep = function (* true if input is empty or starts with a separator *)
  | [] -> true
  | c::_ -> not (isDigit c || isLetter c) ;;

let rec symbol (l, s) =
  match s with
    | c::t when c <= ' ' -> symbol (l + 1, t)
    | ','::t -> (COMMA, (l + 1, t))
    | '('::t -> (LPAREN, (l + 1, t))
    | ')'::t -> (RPAREN, (l + 1, t))
    | '+'::t -> (PLUS, (l + 1, t))
    | '-'::t -> (MINUS, (l + 1, t))
    | '*'::t -> (TIMES, (l + 1, t))
    | 'd'::'i'::'v'::t when sep t -> (DIV, (l + 3, t))
    | 'm'::'o'::'d'::t when sep t -> (MOD, (l + 3, t))
    | c::t when isDigit c ->
        let rec num n = function (* returns integer and remainder of source *)
          | l, d::u when isDigit d -> num (n * 10 + toDigit d) (l + 1, u)
          | any -> (n, any)
        in let (n, u) = num (toDigit c) (l + 1, t) in (INT n, u)
    | c::t when isLetter c ->
        let rec id s = function (* returns char list and remainder of source *)
          | l, d::u when isLetter d || isDigit d -> id (s @ [d]) (l + 1, u)
          | any -> (s, any)
        in let (r, u) = id [c] (l + 1, t) in (ID r, u)
    | _ -> (EOF, (l, s)) ;;

let rec expression s =
  let rec moreterms (p, t) =
    match symbol t with
      | PLUS, u -> let (q, v) = term u in moreterms (Sum (p, q), v)
      | MINUS, u -> let (q, v) = term u in moreterms (Diff (p, q), v)
      | _ -> (p, t)
  in moreterms (term s)

and term s =
  let rec morefactors (p, t) =
    match symbol t with
    | TIMES, u -> let (q, v) = factor u in morefactors (Prod (p, q), v)
    | DIV, u -> let (q, v) = factor u in morefactors (Quot (p, q), v)
    | _ -> (p, t)
  in morefactors (factor s)

and factor s =
  match symbol s with
    | LPAREN, t ->
        let p, u = expression t in
        (match symbol u with
          | RPAREN, v -> (p, v)
          | _, (p, _) -> raise (Source_error (p, "missing )")))
    | INT n, t -> (Int n, t)
    | ID id, t ->
        (match symbol t with
          | LPAREN, u ->
              let p, v = exprList u in
              (match symbol v with
                | RPAREN, w -> (Fun (id, p), w)
                | _, (p, _) -> raise (Source_error (p, "missing )")))
          | _ -> (Id id, t))
    | EOF, (p, _) -> raise (Source_error (p, "unexpected end"))
    | _, (p, _) -> raise (Source_error (p, "id or num expected"))

and exprList s =
  let p, t = expression s in
  match symbol t with
    | COMMA, u -> let (q, n) = exprList u in (p::q, n)
    | _ -> ([p], t)
;;

let parse s =
  try
    match expression (1, s) with
      p, (l, t) ->
        print_string (string_of_int (l - 1) ^ " characters parsed"); p
  with Source_error (p, m) ->
    raise (Failure ("error at " ^ string_of_int p ^ ": " ^ m ^ "")) ;;

parse ['a'; ' '; '+'; ' '; '('; 'b'; ' '; 'd'; 'i'; 'v'; ' '; 'c'; ')'];;
parse ['a'; '+'; 'f'; '('; 'x'; ','; 'y'; ')'];;
parse ['a'; '+'; 'b'; '*'; 'c'];;
parse ['1'; '2'; '*'; '3'; '4'; '5'; '*'; 'a'; 'b'; 'c'];;
parse ['a'; '+'; '('; 'b'];;
parse ['1'; '+'; '*'];;
parse ['a'; '+'];;
parse ['a'; 'b'];;

(* Running above examples while tracing symbol (with "#trace symbol;;")
reveals that symbol will repeatedly be applied to the same argument,
e.g. 4 times to recognize RPAREN. This can be avoided by letting
the input sequence start with its first symbol followed by the rest
of the character sequence, thus mimicking an imperative parser that
that stores the next symbol in a global variable and repeatedly
accesses that.

symbol : source -> symbol * source
expression, term, factor : symbol * source -> exp * (symbol * source)
exprList : source -> expr List * symbol * source

Hence the scanner is unchanged.
*)

let rec expression s =
  let rec moreterms (p, t) =
    match t with
      | PLUS, u -> let (q, v) = term (symbol u) in moreterms (Sum (p, q), v)
      | MINUS, u -> let (q, v) = term (symbol u) in moreterms (Diff (p, q), v)
      | _ -> (p, t)
  in moreterms (term s)

and term s =
  let rec morefactors (p, t) =
    match t with
    | TIMES, u -> let (q, v) = factor (symbol u) in morefactors (Prod (p, q), v)
    | DIV, u -> let (q, v) = factor (symbol u) in morefactors (Quot (p, q), v)
    | _ -> (p, t)
  in morefactors (factor s)

and factor s =
  match s with
    | LPAREN, t ->
        let p, u = expression (symbol t) in
        (match u with
          | RPAREN, v -> (p, symbol v)
          | _, (p, _) -> raise (Source_error (p, "missing )")))
    | INT n, t -> (Int n, symbol t)
    | ID id, t ->
        (match symbol t with
          | LPAREN, u ->
              let p, v = exprList (symbol u) in
              (match v with
                | RPAREN, w -> (Fun (id, p), symbol w)
                | _, (p, _) -> raise (Source_error (p, "missing )")))
          | any -> (Id id, any))
    | EOF, (p, _) -> raise (Source_error (p, "unexpected end"))
    | _, (p, _) -> raise (Source_error (p, "id or num expected"))

and exprList s =
  let p, t = expression s in
  match t with
    | COMMA, u -> let (q, n) = exprList (symbol u) in (p::q, n)
    | any -> ([p], any)
;;

let parse s =
  try
    match expression (symbol (1, s)) with
      p, (x, (l, t)) ->
        print_string (string_of_int (l - 1) ^ " characters parsed"); p
  with Source_error (p, m) ->
    raise (Failure ("error at " ^ string_of_int p ^ ": " ^ m ^ "")) ;;

parse ['a'; ' '; '+'; ' '; '('; 'b'; ' '; 'd'; 'i'; 'v'; ' '; 'c'; ')'];;
parse ['a'; '+'; 'f'; '('; 'x'; ','; 'y'; ')'];;
parse ['a'; '+'; 'b'; '*'; 'c'];;
parse ['1'; '2'; '*'; '3'; '4'; '5'; '*'; 'a'; 'b'; 'c'];;
parse ['a'; '+'; '('; 'b'];;
parse ['1'; '+'; '*'];;
parse ['a'; '+'];;
parse ['a'; 'b'];;

(* Note that tracing will still reveal that some symbols are returned
several times. However, this is due to the fact that function symbol
consumes blanks in a recursive way, and hence legitimate. *)



(* Parsing combinators *)

exception Noparse;;

let sym x = function
  h::t when x = h -> t | _ -> raise Noparse ;;

let option pr s =
  try pr s
  with Noparse -> s ;;

let rec repeat pr s =
  try repeat pr (pr s)
  with Noparse -> s ;;

let seq pr1 pr2 s =
  pr2 (pr1 s) ;;

let choice pr1 pr2 s =
  try pr1 s
  with Noparse -> pr2 s ;;

(* parser for A = {a} b *)

let aa = seq (repeat (sym 'a')) (sym 'b');;

aa ['b'];;
aa ['a'; 'a'; 'b'];;
aa ['a'; 'b'; 'b'];;

(* parser for A = a A c | b *)

let rec aa s =
  choice (seq (sym 'a') (seq aa (sym 'c'))) (sym 'b') s;;

(* parser for A = a a A | a b *)

let rec aa s =
  choice (seq (sym 'a') (seq (sym 'a') aa)) (seq (sym 'a') (sym 'b')) s;;

aa ['a'; 'a'];;
aa ['a'; 'b'];;
aa ['a'; 'a'; 'b'];;
aa ['a'; 'a'; 'a'; 'b'];;

(* parser for
   S = A | B
   A = x A | y
   B = x B | z
*)

let rec ss s = choice aa bb s
and aa s = choice (seq (sym 'x') aa) (sym 'y') s
and bb s = choice (seq (sym 'x') bb) (sym 'z') s;;

ss ['x'; 'x'];;
ss ['y'];;
ss ['z'; 'z'];;
ss ['x'; 'y'];;
ss ['x'; 'x'; 'x'; 'x'; 'x'; 'x'; 'z'];;

(* Parsing combinator with syntax tree *)

let sym x = function
  h::t when x = h -> (h, t) |
  _ -> raise Noparse;;

let seq pr1 pr2 f s =
  let (p, t) = pr1 s in
  let (q, u) = pr2 t in
  (f(p, q), u);;

let choice pr1 pr2 s =
  try let (p, t) = pr1 s in (p, t)
  with Noparse ->
    let (p, t) = pr2 s in (p, t);;

let rec repeatlist pr p s =
  try let (q, t) = pr s in repeatlist pr (p @ [q]) t
  with Noparse -> (p, s);;

let apply f = function
  (p, s) -> (f p, s) |
  _ -> raise Noparse;;

let test f = function
  (p, s) when f p -> (p, s) |
  _ -> raise Noparse;;

let id x = x;;

(* parser for A = {a | b} *)

repeatlist (choice (sym 'a') (sym 'b')) [] ['a'; 'b'; 'b'; 'a'];;

let rec repeat pr f p s =
  try let (q, t) = pr s in repeat pr f (f p q) t
  with Noparse ->  (p, s);;

(* Parser for
  A = id {"," id}
where id is a lower case character, producing Fun (char, exp list)
*)

type exp =  Id of char | Fun of char * exp list ;;

let symCh = function
  ch::t when ch >= 'a' && ch <= 'z' -> (ch, t) |
  _ -> raise Noparse;;

let symId s =
  let (id, t) = symCh s in (Id id, t);;

let append f l = f @ [l];;

let expfun (ch, l) = Fun (ch, l);;

let aa = seq symCh (repeat (seq (sym ',') symId snd) append []) expfun;;

aa ['a'; ','; 'b'];;

(* Exercise: write a combinator parser for the expression grammar! *)
