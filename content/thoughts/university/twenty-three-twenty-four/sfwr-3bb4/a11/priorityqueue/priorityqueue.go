package main

import (
	"math/rand"
	"sort"
	"strconv"
)

type PriorityMessage struct {
	Priority int // non-negative
	Message  string
}

func priorityQueue(capacity int, west chan PriorityMessage, east chan string) {
	queue := make([]PriorityMessage, 0, capacity)

	for {
		if len(queue) == 0 {
			// Queue is empty, only receive messages
			msg := <-west
			queue = append(queue, msg)
		} else if len(queue) == capacity {
			// Queue is full, only send messages
			sortQueue(&queue)
			east <- queue[0].Message
			queue = queue[1:]
		} else {
			// Queue is neither full nor empty, can receive or send messages
			select {
			case msg := <-west:
				queue = append(queue, msg)
			case east <- queue[0].Message:
				queue = queue[1:]
			}
		}

		sortQueue(&queue)
	}
}

func sortQueue(queue *[]PriorityMessage) {
	sort.SliceStable(*queue, func(i, j int) bool {
		return (*queue)[i].Priority < (*queue)[j].Priority
	})
}

func sendMessages(n int, ch0 chan PriorityMessage, ch1 chan string) { // 0 <= n <= 90, number of messages
	for s := 10; s < n+10; s++ { // 2-digit serial number
		prio := rand.Intn(10) // 1-digit priority
		m := strconv.Itoa(prio) + "." + strconv.Itoa(s)
		ch0 <- PriorityMessage{prio, m}
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
