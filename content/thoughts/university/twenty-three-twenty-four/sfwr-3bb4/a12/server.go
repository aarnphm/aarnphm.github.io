package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"net/rpc"
	"os"
	"sort"
	"time"
)

type PuzzleSession struct {
	StartTime   time.Time
	PuzzleIndex int
}

type PuzzleRequest struct {
	SessionID int
	Puzzle    [][]int
}

type SudokuPuzzles struct {
	Unsolved [][][]int `json:"unsolved"`
	// ”' go formatting ”'
}

type PlayerSession struct {
	TotalTime     int64
	PuzzlesSolved int
}

type PuzzleSolution struct { // sent from player to server
	SessionID int
	Solution  [][]int
	Player    string
}

type LeaderboardEntry struct { // entry in leaderboard
	Player    string
	TimeTaken int64
}

type SudokuServer struct { // server state
	Puzzles         [][][]int
	Leaderboard     []LeaderboardEntry
	ActiveSessions  map[int]PuzzleSession
	player_sessions map[string]PlayerSession
}

// RequestPuzzle provides a new puzzle to the player
func (s *SudokuServer) RequestPuzzle(args *int, reply *PuzzleRequest) error {
	sessionID := rand.Intn(10000)
	puzzleIndex := rand.Intn(len(s.Puzzles))
	s.ActiveSessions[sessionID] = PuzzleSession{StartTime: time.Now(), PuzzleIndex: puzzleIndex}
	*reply = PuzzleRequest{SessionID: sessionID, Puzzle: s.Puzzles[puzzleIndex]}
	return nil
}

// SubmitSolution receives the solution from the player
func (s *SudokuServer) SubmitSolution(args *PuzzleSolution, reply *string) error {
	session, ok := s.ActiveSessions[args.SessionID]
	if !ok {
		*reply = "Session not found."
		return nil
	}
	if !isValidSudoku(args.Solution, s.Puzzles[session.PuzzleIndex]) {
		*reply = "Incorrect solution."
		return nil
	}
	timeTakenMillis := time.Since(session.StartTime).Milliseconds()
	playerSession := s.player_sessions[args.Player]
	playerSession.TotalTime += timeTakenMillis
	playerSession.PuzzlesSolved++
	s.player_sessions[args.Player] = playerSession
	if playerSession.PuzzlesSolved == 2 {
		s.updateLeaderboard(args.Player, playerSession.TotalTime)
	}
	*reply = fmt.Sprintf("Correct solution! Time taken: %d milliseconds", timeTakenMillis)
	return nil
}

// updateLeaderboard updates the leaderboard with the new entry
func (s *SudokuServer) updateLeaderboard(player string, timeTaken int64) {
	entry := LeaderboardEntry{Player: player, TimeTaken: timeTaken}
	s.Leaderboard = append(s.Leaderboard, entry)
	sort.Slice(s.Leaderboard, func(i, j int) bool {
		return s.Leaderboard[i].TimeTaken < s.Leaderboard[j].TimeTaken
	})
	if len(s.Leaderboard) > 100 {
		s.Leaderboard = s.Leaderboard[:100]
	}
}

// GetLeaderboard returns the current leaderboard
func (s *SudokuServer) GetLeaderboard(args *int, reply *[]LeaderboardEntry) error {
	*reply = s.Leaderboard
	return nil
}

func isValidSudoku(solution [][]int, puzzle [][]int) bool {
	// Check if each row, column, and 3x3 subgrid contains all numbers from 1 to 9
	for i := 0; i < 9; i++ {
		if !isValidRow(solution, i) || !isValidColumn(solution, i) || !isValidBox(solution, i) {
			return false
		}
	}
	// Check if the solution matches the non-zero values of the puzzle
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if puzzle[i][j] != 0 && puzzle[i][j] != solution[i][j] {
				return false
			}
		}
	}
	return true
}

func isValidRow(board [][]int, row int) bool {
	seen := make(map[int]bool)
	for i := 0; i < 9; i++ {
		num := board[row][i]
		if num != 0 {
			if seen[num] {
				return false
			}
			seen[num] = true
		}
	}
	return true
}

func isValidColumn(board [][]int, col int) bool {
	seen := make(map[int]bool)
	for i := 0; i < 9; i++ {
		num := board[i][col]
		if num != 0 {
			if seen[num] {
				return false
			}
			seen[num] = true
		}
	}
	return true
}

func isValidBox(board [][]int, box int) bool {
	seen := make(map[int]bool)
	startRow := (box / 3) * 3
	startCol := (box % 3) * 3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			num := board[startRow+i][startCol+j]
			if num != 0 {
				if seen[num] {
					return false
				}
				seen[num] = true
			}
		}
	}
	return true
}

func loadPuzzles(filename string) [][][]int {
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var puzzles SudokuPuzzles
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&puzzles)
	if err != nil {
		panic(err)
	}

	return puzzles.Unsolved
}

func main() {
	rand.Seed(time.Now().UnixNano())

	puzzles := loadPuzzles("sudoku.json")
	if puzzles == nil {
		panic("Failed to load puzzles")
	}

	srv := &SudokuServer{
		Puzzles:         puzzles,
		ActiveSessions:  make(map[int]PuzzleSession),
		player_sessions: make(map[string]PlayerSession),
	}
	rpc.Register(srv)

	listener, err := net.Listen("tcp", ":8004")
	if err != nil {
		panic(err)
	}

	fmt.Println("Sudoku Server running on port 8004")
	for {
		conn, err := listener.Accept()
		if err != nil {
			fmt.Println("Accept error:", err)
			continue
		}
		go rpc.ServeConn(conn)
	}
}
