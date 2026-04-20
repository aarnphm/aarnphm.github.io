	.data
	.text
	.globl main
main:	
	jal ra, hex
	addi a0, zero, 0
	addi a7, zero, 93
	scall
	.globl hex
hex:	
	addi sp, sp, -16
	sw ra, 12(sp)
	sw s0, 8(sp)
	addi s0, sp, 16
	beq zero, zero, L1
L2:	
	addi s10, zero, 7
	sw s10, -4(sp)
	mv s8, a0
	addi a1, zero, 7
	addi a0, zero, 1
	addi a2, zero, 4
	addi a7, zero, SCALL_WRITEINT
	scall
	mv a0, s8
	j L3
L1:	
	addi s10, zero, 28
	sw s10, -4(sp)
	mv s8, a0
	addi a1, zero, 28
	addi a0, zero, 1
	addi a2, zero, 4
	addi a7, zero, SCALL_WRITEINT
	scall
	mv a0, s8
L3:	
	lw ra, 12(sp)
	lw s0, 8(sp)
	addi sp, sp, 16
	ret