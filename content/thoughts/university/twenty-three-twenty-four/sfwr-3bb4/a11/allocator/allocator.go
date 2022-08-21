package main

import (
	"math/rand"
	"time"
)

func printboolarray(a []bool) {
	for _, e := range a {
		if e {
			print(" ")
		} else {
			print("X")
		}
	}
	println()
}

func allocator(capacity int, request chan chan int, release chan int) {
	avail := make([]bool, capacity)
	for i := 0; i < capacity; i++ {
		avail[i] = true
	}
	var timeout <-chan time.Time
	var pending chan int
	isActive := false

	// Loop Invariants:
	// 1. avail is an array with each element representing the availability of a resource.
	//    avail[i] == true if and only if resource i is available.
	// 2. pending is nil if there is no pending request waiting for a resource.
	//    If non-nil, it represents the channel to which a resource or a timeout signal should be sent.
	// 3. timeout is nil if there is no active timeout. If non-nil, it represents the channel that will
	//    receive a signal after a timeout duration.
	// 4. isActive indicates whether the allocator is currently within a timeout period.

	for {
		// Timeout handling
		if timeout != nil && !isActive {
			select {
			case unit := <-release:
				avail[unit] = true
				if pending != nil {
					pending <- unit
					pending = nil
					timeout = nil
				}
			case <-timeout:
				if pending != nil {
					pending <- -1
					pending = nil
				}
				isActive = false
				timeout = nil
			}
		}

		// Request handling
		if !isActive {
			select {
			case unit := <-release:
				avail[unit] = true
				printboolarray(avail)
			case reply := <-request:
				allocated := false
				for i, available := range avail {
					if available {
						avail[i] = false
						reply <- i
						allocated = true
						printboolarray(avail)
						break
					}
				}
				if !allocated {
					if pending == nil {
						pending = reply
						timeout = time.After(1 * time.Second)
						isActive = true
					} else {
						reply <- -1
					}
				}
			}
		} else {
			// Reject all requests while timeout is active
			select {
			case reply := <-request:
				reply <- -1
			default:
				// Do nothing if no request is coming in
			}
		}
	}
}

func main() {
	const C = 5   // capacity
	const R = 100 // repetions of non-timed tests
	const T = 10  // repetitions of timed tests
	request, release, reply := make(chan chan int), make(chan int), make(chan int)

	go allocator(C, request, release)

	// testing allocator up to capacity
	var a [C]bool // available resources, initially false
	for i := 0; i < R; i++ {
		u := rand.Int() % C // randomly request or release a resource
		if a[u] {
			release <- u
			a[u] = false
		} else {
			request <- reply
			r := <-reply
			if r < 0 {
				panic("available resource not allocated")
			} else if r >= C {
				panic("improper resource allocated")
			} else if a[r] {
				panic("same resource used twice")
			} else {
				a[r] = true
			}
		}
	}
	for u := 0; u < C; u++ {
		if a[u] {
			release <- u
			a[u] = false
		}
	}

	// testing allocator above capacity but without timeout
	for u := 0; u < C; u++ {
		request <- reply
		a[<-reply] = true
	} // request all resources
	for u := 0; u < C; u++ {
		if !a[u] {
			panic("resources not properly allocated")
		}
	}
	for i := 0; i < R; i++ {
		request <- reply // request one more resource; cannot be satisfied immediatley
		release <- 0     // release one resource; above request can now be satisfied
		select {         // we give the resource allocator .2 seconds to satisfy the request
		case <-time.After(200 * time.Millisecond):
			panic("released resouce not taken")
		case <-reply: // ok; now all resources are taken
		}
	}

	// testing allocator above capacity with timeout
	for i := 0; i < T; i++ {
		request <- reply                    // request one more resource; cannot be satisfied immediatley
		time.Sleep(1200 * time.Millisecond) // resource allocator should timeout after 1 second
		select {
		case r := <-reply:
			if r >= 0 {
				panic("negative reply expected")
			}
		default:
			panic("timeout within 1 second expected")
		}
	}

	// testing allocator with repeated requests above capacity
	request <- reply // request one more resource; cannot be satisfied immediatley
	for j := 0; j < R; j++ {
		reply2 := make(chan int)
		request <- reply2 // request another resources; request should be rejected immediately
		select {          // we still give it .2 seconds to be rejected
		case r := <-reply2:
			if r >= 0 {
				panic("negative reply expected")
			}
		case <-time.After(200 * time.Millisecond):
			panic("immediate rejection of second requests expected")
		}
	}
}
