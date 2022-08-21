package main

import (
	"fmt"
	"net/rpc"
)

func printPuzzle(puzzle [][]int) {
	for _, row := range puzzle {
		for _, val := range row {
			fmt.Printf("%d ", val)
		}
		fmt.Println()
	}
}

// Checks if it's valid to place a number in the given position
func isValid(board [][]int, row, col, num int) bool {
	// Check row and column
	for x := 0; x < 9; x++ {
		if board[row][x] == num || board[x][col] == num {
			return false
		}
	}
	// Check 3x3 box
	startRow := row - row%3
	startCol := col - col%3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if board[i+startRow][j+startCol] == num {
				return false
			}
		}
	}
	return true
}

// Sudoku solver using backtracking
func solveSudoku(board [][]int) bool {
	for row := 0; row < 9; row++ {
		for col := 0; col < 9; col++ {
			if board[row][col] == 0 {
				for num := 1; num <= 9; num++ {
					if isValid(board, row, col, num) {
						board[row][col] = num
						if solveSudoku(board) {
							return true
						}
						board[row][col] = 0
					}
				}
				return false
			}
		}
	}
	return true
}

func main() {
	player, err := rpc.Dial("tcp", "localhost:8004")
	if err != nil {
		panic(err)
	}

	for i := 0; i < 2; i++ {
		var req int
		var puzzleResp PuzzleRequest
		err = player.Call("SudokuServer.RequestPuzzle", &req, &puzzleResp)
		if err != nil {
			panic(err)
		}

		fmt.Println("Puzzle received:")
		printPuzzle(puzzleResp.Puzzle)

		// Solve the Sudoku puzzle
		solveSudoku(puzzleResp.Puzzle)
		fmt.Println("Solved Puzzle:")
		printPuzzle(puzzleResp.Puzzle)

		var submitResp string
		solutionReq := PuzzleSolution{
			SessionID: puzzleResp.SessionID,
			Solution:  puzzleResp.Puzzle,
			Player:    "aarnphm",
		}
		err = player.Call("SudokuServer.SubmitSolution", &solutionReq, &submitResp)
		if err != nil {
			panic(err)
		}
		fmt.Println("Submit response:", submitResp)
	}

	// Viewing the leaderboard
	var leaderboard []LeaderboardEntry
	var req int
	err = player.Call("SudokuServer.GetLeaderboard", &req, &leaderboard)
	if err != nil {
		panic(err)
	}

	fmt.Println("Leaderboard:")
	for i, entry := range leaderboard {
		fmt.Printf("%d: %s - %.6d ms\n", i+1, entry.Player, entry.TimeTaken)
	}
}
