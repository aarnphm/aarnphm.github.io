(module
(import "P0lib" "write" (func $write (param i32)))
(import "P0lib" "writeln" (func $writeln))
(import "P0lib" "read" (func $read (result i32)))
(global $_memsize (mut i32) i32.const 0)
(func $program
(local $x i32)
(local $y i32)
(local $0 i32)
call $read
local.set $x
call $read
local.set $y
local.get $x
local.get $y
i32.gt_s
if
local.get $x
call $write
else
local.get $y
call $write
end
)
(memory 1)
(start $program)
)