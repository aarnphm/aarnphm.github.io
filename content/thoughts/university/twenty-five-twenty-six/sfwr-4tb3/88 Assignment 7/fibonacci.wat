(module
(import "P0lib" "write" (func $write (param i32)))
(import "P0lib" "writeln" (func $writeln))
(import "P0lib" "read" (func $read (result i32)))
;;  procedure fib(n: integer) → (r: integer)
(func $fib (param $n i32) (result i32)
  ;;    if n ≤ 1 then r := n
  local.get $n
  i32.const 1
  i32.le_s
  if (result i32)
    local.get $n
  ;;    else a ← fib(n - 1); b ← fib(n - 2); r := a + b
  else
    local.get $n
    i32.const 1
    i32.sub
    call $fib
    local.get $n
    i32.const 2
    i32.sub
    call $fib
    i32.add
  end
)
;;  program fibonacci
(func $program
  ;;  var x: integer
  ;;    x ← read(); x ← fib(x); write(x)
  call $read
  call $fib
  call $write
)
(memory 1)
(start $program)
)
