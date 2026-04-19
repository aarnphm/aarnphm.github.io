(module
  (import "P0lib" "write" (func $write (param i32)))
  (import "P0lib" "writeln" (func $writeln))
  (import "P0lib" "read" (func $read (result i32)))
  (export "program" (func $program))
  (func $QuotRem (param $x i32) (param $y i32) 
    (local $q i32)
    (local $r i32)
    i32.const 0
    local.set $q
    local.get $x
    local.set $r
    loop $label0
      local.get $r
      local.get $y
      i32.ge_s
      if
        local.get $r
        local.get $y
        i32.sub
        local.set $r
        local.get $q
        i32.const 1
        i32.add
        local.set $q
        br $label0
      end
    end
    local.get $q
    call $write
    local.get $r
    call $write
  )
  (func $program
    (local $x i32)
    (local $y i32)
    call $read
    local.set $x
    call $read
    local.set $y
    local.get $x
    local.get $y
    call $QuotRem
  )
  (memory 1)
  (start $program)
)
