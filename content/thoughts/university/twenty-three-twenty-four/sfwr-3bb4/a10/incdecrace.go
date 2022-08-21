package main

const N = 1000

var x int
var done chan bool

func incN() {
	for i := 0; i < N; i++ {
		x += 1
	}
	done <- true
}
func decN() {
	for i := 0; i < N; i++ {
		x -= 1
	}
	done <- true
}
func main() {
	x, done = 0, make(chan bool)
	go incN()
	go decN()
	<-done
	<-done
	println("computed:", x)
	println("expected: 0")
}
