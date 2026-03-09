(module
  (import "P0lib" "write" (func $write (param i32)))
  (import "P0lib" "writeln" (func $writeln))
  (import "P0lib" "read" (func $read (result i32)))
  (tag $e)
 (func $q (param $x i32) (result i32)
    (local $y i32)
    i32.const 1
    call $write
    try
      try
        i32.const 3
        call $write
        throw $e
        i32.const 5
        call $write
      catch $e
        i32.const 7
        call $write
      end
      i32.const 9
      call $write
      throw $e
      i32.const 11
      call $write
    catch $e
      i32.const 13
      call $write
    end
    i32.const 15
  )
  (func $program
    i32.const 1
    call $q
    call $write
  )
  (memory 1)
  (start $program)
)
