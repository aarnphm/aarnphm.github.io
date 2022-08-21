package footman

import (
	"math/rand"
	"time"
)

var (
	left, right         [5]chan bool
	ph                  [5]string
	enter, exit, permit chan bool
	log                 chan string
)

func footman() {
	seatedPhilosophers := 0
	for {
		select {
		case <-enter:
			if seatedPhilosophers < 4 {
				seatedPhilosophers++
				permit <- true
			} else {
				<-exit
				seatedPhilosophers--
				continue
			}
		case <-exit:
			seatedPhilosophers--
		}
	}
}

func runtimemonitor() {
	sitting := 0
	for m := range log {
		if m == "exit" {
			sitting -= 1
		} else {
			sitting += 1
			if sitting == 5 {
				panic("5 philosophers sitting")
			}
		}
	}
}

func philosopherState(i int, s string) {
	ph[i] = s
	println(ph[0], ph[1], ph[2], ph[3], ph[4])
	time.Sleep(time.Second * time.Duration(rand.Int()%3)) // eating or thinking 0..2 sec
}

func fork(i int) {
	for {
		select {
		case <-left[i]:
			<-left[i]
		case <-right[i]:
			<-right[i]
		}
	}
}

func philosopher(i int) {
	for {
		enter <- true
		<-permit
		log <- "enter"
		left[i] <- true
		right[(i+1)%5] <- true
		philosopherState(i, "eats  ")
		left[i] <- false
		right[(i+1)%5] <- false
		philosopherState(i, "thinks")
		log <- "exit"
		exit <- true
	}
}

func main() {
	for i := 0; i < 5; i++ {
		left[i], right[i], ph[i] = make(chan bool), make(chan bool), "thinks"
	}
	enter, exit, log = make(chan bool), make(chan bool), make(chan string)
	permit = make(chan bool)
	go footman()
	go runtimemonitor()
	for i := 0; i < 5; i++ {
		go fork(i)
		go philosopher(i)
	}
	time.Sleep(20 * time.Second)
}
