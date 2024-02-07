package main

import (
	"reflect"
	"sync"
)

const N = 20

type IntPair = struct {
	i int
	f int
}

var r chan IntPair
var c [N]chan int
var wg sync.WaitGroup

func fib(i int) {
	defer wg.Done()

	var f int
	if i == 0 {
		f = 0
	} else if i == 1 {
		f = 1
	} else {
		f1 := <-c[i-1]
		f2 := <-c[i-2]
		f = f1 + f2
	}

	// Send the result to the next two channels if they exist
	if i < N-1 {
		c[i] <- f
		if i != 0 && i != N-2 {
			c[i] <- f
		}
	}

	// Send the result to the result channel
	r <- IntPair{i, f}
}

func main() {
	r = make(chan IntPair, N)
	for i := range c {
		c[i] = make(chan int, 2)
	}

	wg.Add(N)
	for i := 0; i < N; i++ {
		go fib(i)
	}
	wg.Wait()

	close(r)
	for _, ch := range c {
		close(ch)
	}

	m := make(map[int]int)
	for i := 0; i < N; i++ {
		p := <-r
		m[p.i] = p.f
	}

	e := map[int]int{0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55, 11: 89, 12: 144, 13: 233, 14: 377, 15: 610, 16: 987, 17: 1597, 18: 2584, 19: 4181}
	println("computed and expected Fibonacci numbers are the same: ", reflect.DeepEqual(m, e))
}
