// the grid contains paper rolls (@) and empty spaces (.). a forklift can access
// a roll if it has fewer than 4 adjacent rolls in the 8 surrounding cells
// (Moore neighborhood).
//
// part 1: single pass - count all rolls with <4 neighbors.
//
// part 2: iterative erosion - repeatedly remove all accessible rolls until
// none remain accessible. each round, collect positions to remove, then
// batch-remove them (order matters: we evaluate the whole grid state before
// modifying). accumulate total removed across all rounds.
//
// the erosion terminates when only dense clusters remain where every roll
// has 4+ neighbors, forming stable cores that forklifts cannot penetrate.
package main

import (
	"bufio"
	"fmt"
	"os"
)

var dirs = [][2]int{
	{-1, 0},
	{-1, 1},
	{0, 1},
	{1, 1},
	{1, 0},
	{1, -1},
	{0, -1},
	{-1, -1},
}

func adjacent(grid []string, r, c, rows, cols int) int {
	count := 0
	for _, d := range dirs {
		nr, nc := r+d[0], c+d[1]
		if nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == '@' {
			count++
		}
	}
	return count
}

func findAccessible(grid []string, rows, cols int) [][2]int {
	var accessible [][2]int
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if grid[r][c] == '@' && adjacent(grid, r, c, rows, cols) < 4 {
				accessible = append(accessible, [2]int{r, c})
			}
		}
	}
	return accessible
}

func p1(grid []string, rows, cols int) int {
	return len(findAccessible(grid, rows, cols))
}

func p2(grid []string, rows, cols int) int {
	// copy grid since we mutate it
	g := make([]string, len(grid))
	copy(g, grid)

	total := 0
	for {
		toRemove := findAccessible(g, rows, cols)
		if len(toRemove) == 0 {
			break
		}
		total += len(toRemove)
		for _, pos := range toRemove {
			row := []byte(g[pos[0]])
			row[pos[1]] = '.'
			g[pos[0]] = string(row)
		}
	}
	return total
}

func main() {
	file, err := os.Open("./d4.txt")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var grid []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		grid = append(grid, scanner.Text())
	}

	rows := len(grid)
	if rows == 0 {
		fmt.Println(0, 0)
		return
	}
	cols := len(grid[0])

	fmt.Println("p1:", p1(grid, rows, cols))
	fmt.Println("p2:", p2(grid, rows, cols))
}
