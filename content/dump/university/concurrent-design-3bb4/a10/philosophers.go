// A deadlock among philosophers is prevented if every philosopher picks up their lower-numbered fork first. The following program has a runtime monitor to check that,
// and the philosophers are instrumented with statements communicating their actions to the runtime monitor.
// The implementation of the philosophers is flawed as every philosopher picks
// up their left-hand fork (through that fork's right-hand channel) and then their right-hand fork (through that fork's left-hand channel).
// Hence, the runtime monitor will, at some point, flag that situation and terminate the program.
// Modify it such that, instead of picking up the left and then the right fork, philosophers pick up the lower-numbered fork and then
// the higher-numbered fork! After left[l] <- true add log <- PhilFork{ph, l} and after right[r] <- true add log <- PhilFork{ph, r}, like above. [2 points]
// package main
//
// import (
// 	"math/rand"
// 	"time"
// )
//
// type PhilFork struct {
// 	ph int
// 	f  int
// }
//
// var left, right [5]chan bool
// var ph [5]string
// var log chan PhilFork
//
// func philosopherState(i int, s string) {
// 	ph[i] = s
// 	println(ph[0], ph[1], ph[2], ph[3], ph[4])
// 	time.Sleep(time.Second * time.Duration(rand.Int()%3)) // eating or thinking 0..2 sec
// }
// func fork(f int) {
// 	for {
// 		select {
// 		case <-left[f]:
// 			<-left[f]
// 		case <-right[f]:
// 			<-right[f]
// 		}
// 	}
// }
// func runtimemonitor() {
// 	ff := []int{-1, -1, -1, -1, -1} // first picked up fork; -1 if no fork picked up
// 	for p := range log {
// 		if ff[p.ph] == -1 {
// 			ff[p.ph] = p.f
// 			// println(p.ph, "picking up", p.f, "first")
// 		} else {
// 			// println(p.ph, "picking up", p.f, "second")
// 			if ff[p.ph] < p.f {
// 				ff[p.ph] = -1
// 			} else {
// 				panic("higher-numbered fork picked first")
// 			}
// 		}
// 	}
// }
// func philosopher(ph int) {
// 	for {
// 		l, r := ph, (ph+1)%5
// 		if l < r {
// 			left[l] <- true
// 			log <- PhilFork{ph, l}
// 			right[r] <- true
// 			log <- PhilFork{ph, r}
// 		} else {
// 			right[r] <- true
// 			log <- PhilFork{ph, r}
// 			left[l] <- true
// 			log <- PhilFork{ph, l}
// 		}
// 		philosopherState(ph, "eats  ")
// 		left[l] <- false
// 		right[r] <- false
// 		philosopherState(ph, "thinks")
// 	}
// }
// func main() {
// 	log = make(chan PhilFork)
// 	go runtimemonitor()
// 	for i := 0; i < 5; i++ {
// 		left[i], right[i], ph[i] = make(chan bool), make(chan bool), "thinks"
// 	}
// 	for i := 0; i < 5; i++ {
// 		go fork(i)
// 		go philosopher(i)
// 	}
// 	time.Sleep(20 * time.Second)
// }
