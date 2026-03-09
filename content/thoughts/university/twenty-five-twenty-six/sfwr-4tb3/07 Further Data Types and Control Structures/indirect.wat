(module
  (import "P0lib" "write" (func $write (param i32)))
  (import "P0lib" "read" (func $read (result i32)))
  (func $plus1 (param $x i32) (result i32)
    i32.const 1
    local.get $x
    i32.add)
  (func $plus2 (param $x i32) (result i32)
    i32.const 2
    local.get $x
    i32.add)
  (func $program
    call $read ;; push function parameter on stack
    call $read ;; push function index on stack
    call_indirect (param i32) (result i32)
    call $write)
  (table 2 funcref)
  (elem (i32.const 0) $plus1)
  (elem (i32.const 1) $plus2)
  (start $program)
)
