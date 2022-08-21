package main

import (
	"container/heap"
	"math/rand"
	"sort"
	"strconv"
)

type PriorityMessage struct {
	Priority  int // non-negative
	Message   string
	Timestamp int // used to maintain order of insertion
}

type PriorityQueue []PriorityMessage

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	if pq[i].Priority == pq[j].Priority {
		return pq[i].Timestamp < pq[j].Timestamp
	}
	return pq[i].Priority < pq[j].Priority
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(PriorityMessage))
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	x := old[n-1]
	*pq = old[0 : n-1]
	return x
}

func priorityQueue(capacity int, west chan PriorityMessage, east chan string) {
	pq := make(PriorityQueue, 0, capacity)
	heap.Init(&pq)
	timestamp := 0

	for {
		if pq.Len() == 0 {
			// Queue is empty, only receive messages
			message := <-west
			heap.Push(&pq, PriorityMessage{message.Priority, message.Message, timestamp})
			timestamp++
		} else if pq.Len() < capacity {
			// Queue is neither full nor empty, can receive or send messages
			select {
			case message := <-west:
				heap.Push(&pq, PriorityMessage{message.Priority, message.Message, timestamp})
				timestamp++
			case east <- pq[0].Message:
				heap.Pop(&pq)
			}
		} else {
			// Queue is full, can only send messages
			east <- pq[0].Message
			heap.Pop(&pq)
		}
	}
}

func sendMessages(n int, ch0 chan PriorityMessage, ch1 chan string) { // 0 <= n <= 90, number of messages
	for s := 10; s < n+10; s++ { // 2-digit serial number
		prio := rand.Intn(10) // 1-digit priority
		m := strconv.Itoa(prio) + "." + strconv.Itoa(s)
		ch0 <- PriorityMessage{Priority: prio, Message: m}
		ch1 <- m
	}
}

func main() {
	const C, R = 20, 10 // capacity, rounds of testing
	west := make(chan PriorityMessage)
	south, east := make(chan string), make(chan string)

	go priorityQueue(C, west, east)

	// testing priority queue exactly at capacity: received messages must be sent messages in ascending order
	for t := 0; t < 10; t++ {
		var in, out [C]string
		go sendMessages(C, west, south) // priority queue is filled up
		for i := 0; i < C; i++ {        // messages sent to priority queue are copied to array in
			in[i] = <-south
		}
		for i := 0; i < C; i++ { // messages received from priority queue are stored in array out
			out[i] = <-east
			print(out[i], " ") // printed in ascending order
		}
		sort.Strings(in[:]) // sort the sent messages
		if in != out {
			panic("received messages must be sent messages in ascending order")
		}
		println()
	}

	// testing with more messages than capacity: received messages may not always be in ascending order
	done := make(chan bool)
	for t := 0; t < 10; t++ {
		var in, out [2 * C]string
		go sendMessages(2*C, west, south) // priority queue is filled up
		go func() {
			for i := 0; i < 2*C; i++ { // messages sent to priority queue are copied to array in
				in[i] = <-south
			}
			done <- true
		}()
		for i := 0; i < 2*C; i++ { // messages received from priority queue are stored in array out
			out[i] = <-east
			print(out[i], " ") // printed in not necessarily ascending order
		}
		<-done
		sort.Strings(in[:])
		sort.Strings(out[:]) // sort the sent and received messages
		if in != out {
			panic("all sent messages must be received (in some order)")
		}
		println()
	}
}
