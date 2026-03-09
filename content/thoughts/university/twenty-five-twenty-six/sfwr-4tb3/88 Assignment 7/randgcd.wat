(module
(import "P0lib" "write" (func $write (param i32)))
(import "P0lib" "writeln" (func $writeln))
(import "P0lib" "read" (func $read (result i32)))
;;  var r: integer
(global $r (mut i32) i32.const 0)
;;  procedure randint(bound: integer) → (rand: integer)
(func $randint (param $bound i32) (result i32)
(local $rand i32)
(local $0 i32)
  ;;  const a = 16807
  ;;  const c = 11
  ;;  const m = 65535
  ;;    r := (a × r + c) mod m
  i32.const 16807
  global.get $r
  i32.mul
  i32.const 11
  i32.add
  i32.const 65535
  i32.rem_s
  global.set $r
  ;;    rand := r mod bound
  global.get $r
  local.get $bound
  i32.rem_s
  local.set $rand
  local.get $rand
)
;;  procedure gcd(x: integer, y: integer) → (d: integer)
(func $gcd (param $x i32) (param $y i32) (result i32)
(local $d i32)
(local $0 i32)
  ;;  while x ≠ y do
  loop
    local.get $x
    local.get $y
    i32.ne
    if
      ;;    if x > y then x := x - y
      local.get $x
      local.get $y
      i32.gt_s
      if
        ;;      x := x - y
        local.get $x
        local.get $y
        i32.sub
        local.set $x
      ;;    else y := y - x
      else
        local.get $y
        local.get $x
        i32.sub
        local.set $y
      end
      br 1
    end
  end
  ;;  d := x
  local.get $x
  local.set $d
  local.get $d
)
;;  program randgcd
(func $program
  ;;  var x, y, d: integer
  (local $x i32)
  (local $y i32)
  (local $d i32)
  (local $0 i32)
  ;;    r ← read()
  call $read
  global.set $r
  ;;    x ← randint(100); write(x)
  i32.const 100
  call $randint
  local.set $x
  local.get $x
  call $write
  ;;    y ← randint(100); write(y)
  i32.const 100
  call $randint
  local.set $y
  local.get $y
  call $write
  ;;    d ← gcd(x, y); write(d)
  local.get $x
  local.get $y
  call $gcd
  local.set $d
  local.get $d
  call $write
)
(memory 1)
(start $program)
)
