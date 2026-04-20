	.data
x_:	.space 4
y_:	.space 4
z_:	.space 4
	.text
	.globl main
main:	
	jal ra, sum
	addi a0, zero, 0
	addi a7, zero, 93
	scall
	.globl sum
sum:	
	addi sp, sp, -16
	sw ra, 12(sp)
	sw s0, 8(sp)
	addi s0, sp, 16
	mv s4, a0
	addi a0, zero, 0
	addi a7, zero, SCALL_READINT
	scall
	la s8, y_
	sw a1, 0(s8)
	mv a0, s4
	mv s7, a0
	addi a0, zero, 0
	addi a7, zero, SCALL_READINT
	scall
	la s2, z_
	sw a1, 0(s2)
	mv a0, s7
	la s10, y_
	lw s5, 0(s10)
	la s9, z_
	lw s6, 0(s9)
	add s11, s5, s6
	la s3, x_
	sw s11, 0(s3)
	la s8, z_
	lw s4, 0(s8)
	sw s4, -4(sp)
	mv s7, a0
	la s2, z_
	lw a1, 0(s2)
	addi a0, zero, 1
	addi a2, zero, 4
	addi a7, zero, SCALL_WRITEINT
	scall
	mv a0, s7
	lw ra, 12(sp)
	lw s0, 8(sp)
	addi sp, sp, 16
	ret