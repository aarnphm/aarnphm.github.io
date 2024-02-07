package main

const N = 20

var inc, dec, done chan bool
var val chan int

func counter() {
	x := 0
	for {
		select {
		case <-inc:
			x += 1
		case <-dec:
			x -= 1
		case val <- x:
			return
		}
	}
}

func incN() {
	for i := 0; i < N; i++ {
		inc <- true
	}
	done <- true
}

func decN() {
	for i := 0; i < N; i++ {
		dec <- true
	}
	done <- true
}
func main() {
	done = make(chan bool)
	inc, dec, val = make(chan bool), make(chan bool), make(chan int)
	go counter()
	go incN()
	go decN()
	<-done
	<-done
	println("computed:", <-val)
	println("expected: 0")
}
