(module
(import "P0lib" "write" (func $write (param i32)))
(import "P0lib" "writeln" (func $writeln))
(import "P0lib" "read" (func $read (result i32)))
(func $eratosthenes  (result i32)
(local $p i32)
(local $i i32)
(local $j i32)
(local $0 i32)
i32.const 0
i32.const 0xffffffff
i32.xor
local.set $p
i32.const 2
local.set $i
loop
local.get $i
i32.const 5
i32.le_s
if
local.get $i
local.set $0
i32.const 1
local.get $0
i32.shl
local.get $p
i32.and
if
local.get $i
local.get $i
i32.mul
local.set $j
loop
local.get $j
i32.const 32
i32.lt_s
if
local.get $j
local.set $0
i32.const 1
local.get $0
i32.shl
i32.const 0xffffffff
i32.xor
local.get $p
i32.and
local.get $j
local.get $i
i32.add
local.set $j
local.set $p
br 1
end
end
end
local.get $i
i32.const 1
i32.add
local.set $i
br 1
end
end
local.get $p
)
(global $_memsize (mut i32) i32.const 0)
(func $program
(local $p i32)
(local $j i32)
(local $0 i32)
call $eratosthenes
local.set $p
i32.const 2
local.set $j
loop
local.get $j
i32.const 32
i32.lt_s
if
local.get $j
local.set $0
i32.const 1
local.get $0
i32.shl
local.get $p
i32.and
if
local.get $j
call $write
end
local.get $j
i32.const 1
i32.add
local.set $j
br 1
end
end
call $writeln
i32.const 2
local.set $j
loop
local.get $j
i32.const 32
i32.lt_s
if
local.get $j
local.set $0
i32.const 1
local.get $0
i32.shl
local.get $p
i32.and
i32.eqz
if
local.get $j
call $write
end
local.get $j
i32.const 1
i32.add
local.set $j
br 1
end
end
)
(memory 1)
(start $program)
)