(module
  (import "P0lib" "write" (func $write (param i32)))
  (import "P0lib" "read" (func $read (result i32)))
  (func $program
    (local $i i32)
    call $read
    local.set $i
    block $done
      block $else
        block $4
          block $3
            block $1
              local.get $i
              i32.const 1
              i32.sub
              br_table $1 $else $3 $4 $else
            end
            i32.const 1
            call $write
            br $done
          end
          i32.const 3
          call $write
          br $done
        end
        i32.const 4
        call $write
        br $done
      end
      i32.const 0
      call $write
    end)
  (start $program)
)
