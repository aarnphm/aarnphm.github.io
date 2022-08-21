package main

import (
	"math/rand"
	"time"
)

type Item struct {
	Publisher string
	Message   int
}

const R = 4 // rounds of publishing

func mediator(pch chan Item, sch chan chan Item) {
	var subscribers []chan Item
	publishersCount := 0

	for {
		select {
		// Handle new items from publishers
		case item := <-pch:
			if item.Message == -1 {
				publishersCount--
				// Check if all publishers are done
				if publishersCount == 0 {
					// Close all subscriber channels
					for _, subCh := range subscribers {
						close(subCh)
					}
					return
				}
			} else {
				// Distribute item to all subscribers
				for _, subCh := range subscribers {
					subCh <- item
				}
			}

		// Handle new subscriber channels
		case newSub := <-sch:
			subscribers = append(subscribers, newSub)
			publishersCount++
		}
	}
}
func publisher(id string, ch chan Item) {
	println("publisher", id, "active")
	for m := 0; m < R; m++ {
		time.Sleep(time.Millisecond * time.Duration(rand.Int()%600)) // sleep between 0 and .6 seconds
		println("publisher", id, "publishing", m)
		ch <- Item{id, m}
	}
	ch <- Item{id, -1}
	println("publisher", id, "done")
}
func subscriber(id string, med chan chan Item) {
	println("subscriber", id, "subscribed")
	ch := make(chan Item, 3)
	med <- ch // asynchronous channel with capacity 3
	for item := range ch {
		println("    subscriber", id, "received", item.Message, "from", item.Publisher)
		time.Sleep(time.Millisecond * time.Duration(rand.Int()%200)) // sleep between 0 and .2 seconds
	}
	println("subscriber", id, "done")
}
func main() {
	publisherChannel := make(chan Item, 1)       // asynchronous channel with capacity 1
	subscriberChannel := make(chan chan Item, 1) // asynchronous channel with capacity 1
	go mediator(publisherChannel, subscriberChannel)
	for _, name := range []string{"A", "B", "C"} {
		go publisher(name, publisherChannel)
		time.Sleep(time.Millisecond * time.Duration(rand.Int()%200)) // sleep between 0 and .2 seconds
	}
	for _, name := range []string{"K", "L", "M"} {
		go subscriber(name, subscriberChannel)
		time.Sleep(time.Millisecond * time.Duration(rand.Int()%200)) // sleep between 0 and .2 seconds
	}
	time.Sleep(4 * time.Second)
}
