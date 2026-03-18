(module
(import "P0lib" "write" (func $write (param i32)))
(import "P0lib" "writeln" (func $writeln))
(import "P0lib" "read" (func $read (result i32)))
(import "P0lib" "seed" (func $seed (result i32)))
(global $r (mut i32) i32.const 0)
(func $randint (param $lower i32) (param $upper i32) (result i32)
(local $rand i32)
(local $0 i32)
i32.const 16807
global.get $r
i32.mul
i32.const 11
i32.add
i32.const 65535
i32.rem_s
global.set $r
global.get $r
local.get $upper
local.get $lower
i32.sub
i32.rem_s
local.get $lower
i32.add
local.set $rand
local.get $rand
)
(func $power (param $a i32) (param $n i32) (param $p i32) (result i32)
(local $res i32)
(local $0 i32)
i32.const 1
local.get $a
local.get $p
i32.rem_s
local.set $a
local.set $res
loop
local.get $n
i32.const 0
i32.gt_s
if
local.get $n
i32.const 2
i32.rem_s
i32.const 1
i32.eq
if
local.get $res
local.get $a
i32.mul
local.get $p
i32.rem_s
local.get $n
i32.const 1
i32.sub
local.set $n
local.set $res
end
local.get $a
local.get $a
i32.mul
local.get $p
i32.rem_s
local.get $n
i32.const 2
i32.div_s
local.set $n
local.set $a
br 1
end
end
local.get $res
)
(func $likelyPrime (param $n i32) (param $k i32)
(local $i i32)
(local $p i32)
(local $a i32)
(local $0 i32)
local.get $k
i32.const 1
local.set $p
local.set $i
loop
local.get $i
i32.const 0
i32.gt_s
if (result i32)
local.get $p
i32.const 1
i32.eq
else
i32.const 0
end
if
i32.const 1
local.get $n
i32.const 1
i32.sub
call $randint
local.set $a
local.get $a
local.get $n
i32.const 1
i32.sub
local.get $n
call $power
local.set $p
local.get $i
i32.const 1
i32.sub
local.set $i
br 1
end
end
local.get $p
i32.const 1
i32.eq
if
i32.const 1
call $write
else
i32.const 0
call $write
end
)
(global $_memsize (mut i32) i32.const 0)
(func $program
(local $n i32)
(local $k i32)
(local $0 i32)
call $seed
global.set $r
call $read
local.set $n
call $read
local.set $k
local.get $n
local.get $k
call $likelyPrime
)
(memory 1)
(start $program)
)
