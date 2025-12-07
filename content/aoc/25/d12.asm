.global _main
.align 4

.data
filename:     .asciz "d12.txt"
p1_fmt:       .asciz "p1: %ld\n"
p2_fmt:       .asciz "p2: %ld\n"

.bss
.align 4
buffer:       .skip 65536           ; 64KB input buffer
buffer_len:   .skip 8

.text

; syscall numbers (macos = 0x2000000 + unix number)
.equ SYS_EXIT,  0x2000001
.equ SYS_READ,  0x2000003
.equ SYS_OPEN,  0x2000005
.equ SYS_CLOSE, 0x2000006
.equ O_RDONLY,  0

; read_input: reads file into buffer
; returns: bytes read in x0
_read_input:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x19, [sp, #16]

    ; open(filename, O_RDONLY)
    adrp x0, filename@PAGE
    add x0, x0, filename@PAGEOFF
    mov x1, O_RDONLY
    mov x16, SYS_OPEN
    svc #0x80
    cmp x0, #0
    b.lt .read_error
    mov x19, x0                     ; save fd

    ; read(fd, buffer, 65536)
    mov x0, x19
    adrp x1, buffer@PAGE
    add x1, x1, buffer@PAGEOFF
    mov x2, #65536
    mov x16, SYS_READ
    svc #0x80

    ; store bytes read
    adrp x1, buffer_len@PAGE
    add x1, x1, buffer_len@PAGEOFF
    str x0, [x1]
    mov x19, x0                     ; save length

    ; close(fd)
    mov x0, x19
    mov x16, SYS_CLOSE
    svc #0x80

    mov x0, x19                     ; return bytes read
    b .read_done

.read_error:
    mov x0, #0

.read_done:
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

; p1: solve part 1
; args: x0 = buffer ptr, x1 = buffer len
; returns: result in x0
_p1:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x0, #0                      ; return 0 for now
    ldp x29, x30, [sp], #16
    ret

; p2: solve part 2
; args: x0 = buffer ptr, x1 = buffer len
; returns: result in x0
_p2:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    mov x0, #0                      ; return 0 for now
    ldp x29, x30, [sp], #16
    ret

_main:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    str x21, [sp, #32]

    bl _read_input
    cbz x0, .done

    ; solve part 1
    adrp x0, buffer@PAGE
    add x0, x0, buffer@PAGEOFF
    adrp x1, buffer_len@PAGE
    add x1, x1, buffer_len@PAGEOFF
    ldr x1, [x1]
    bl _p1
    mov x19, x0

    ; solve part 2
    adrp x0, buffer@PAGE
    add x0, x0, buffer@PAGEOFF
    adrp x1, buffer_len@PAGE
    add x1, x1, buffer_len@PAGEOFF
    ldr x1, [x1]
    bl _p2
    mov x20, x0

    ; print p1 result
    adrp x0, p1_fmt@PAGE
    add x0, x0, p1_fmt@PAGEOFF
    mov x1, x19
    bl _printf

    ; print p2 result
    adrp x0, p2_fmt@PAGE
    add x0, x0, p2_fmt@PAGEOFF
    mov x1, x20
    bl _printf

.done:
    mov x0, #0
    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret
