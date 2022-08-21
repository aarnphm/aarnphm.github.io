package main

import "fmt"

func main() {
	var x, y int
	done := make(chan bool)
	go func() { x = 1; done <- true }()
	go func() { y = 2; done <- true }()
	<-done
	<-done
	fmt.Println(x, y)
}
