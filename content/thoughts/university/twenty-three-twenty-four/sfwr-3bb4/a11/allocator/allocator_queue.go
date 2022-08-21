package main

import (
	"container/list"
	"math/rand"
	"time"
)

// requestEntry holds the request channel and the time when the request was made.
type requestEntry struct {
	reply     chan int  // reply holds the channel to which the allocator should send the resource.
	timeAdded time.Time // timeAdded holds the time when the request was made.
}

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
	for i := range avail {
		avail[i] = true
	}
	requestQueue := list.New()

	// Loop invariants:
	// 1. Every element in avail is true if the resource is available, and false if it is not.
	// 2. requestQueue is a list.List where each element is a requestEntry.
	//    The list is ordered by the time when each request was added.
	// 3. Every pending request in requestQueue has a 1-second timeout starting from timeAdded.

	ticker := time.NewTicker(100 * time.Millisecond) // Tick every 100ms to check timeouts
	defer ticker.Stop()

	for {
		select {
		case r := <-release:
			// Resource release handling
			avail[r] = true
			printboolarray(avail)

			// Allocate resource to the earliest request that hasn't timed out.
			for e := requestQueue.Front(); e != nil; e = e.Next() {
				entry := e.Value.(requestEntry)
				if time.Since(entry.timeAdded) < 1*time.Second {
					avail[r] = false
					entry.reply <- r
					requestQueue.Remove(e)
					printboolarray(avail)
					break
				}
			}

		case rep := <-request:
			allocated := false
			for i, available := range avail {
				if available {
					// Resource allocation
					avail[i] = false
					rep <- i
					allocated = true
					printboolarray(avail)
					break
				}
			}

			if !allocated {
				// Add request to queue with current time
				requestQueue.PushBack(requestEntry{rep, time.Now()})
			}

		case <-ticker.C:
			// Check for any requests that have timed out
			var next *list.Element
			for e := requestQueue.Front(); e != nil; e = next {
				next = e.Next()
				entry := e.Value.(requestEntry)
				if time.Since(entry.timeAdded) >= 1*time.Second {
					entry.reply <- -1
					requestQueue.Remove(e)
				}
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
