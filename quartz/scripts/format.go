package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/schollz/progressbar/v3"
	"github.com/yosssi/gohtml"
)

type Stats struct {
	formatted uint64
	skipped   uint64
	errors    uint64
}

func (s *Stats) String() string {
	return fmt.Sprintf("\nSummary:\n✓ Formatted: %d files\n• Skipped: %d files\n✗ Errors: %d files\n",
		atomic.LoadUint64(&s.formatted),
		atomic.LoadUint64(&s.skipped),
		atomic.LoadUint64(&s.errors))
}

type FileTask struct {
	path     string
	content  []byte
	errChan  chan<- error
	statChan chan<- string
}

func worker(id int, tasks <-chan FileTask, wg *sync.WaitGroup) {
	defer wg.Done()

	for task := range tasks {
		formatted := formatHTML(task.content)
		if !bytes.Equal(formatted, task.content) {
			if err := os.WriteFile(task.path, formatted, 0644); err != nil {
				task.errChan <- fmt.Errorf("worker %d: failed to write %s: %v", id, task.path, err)
				task.statChan <- "error"
				continue
			}
			task.statChan <- "formatted"
		} else {
			task.statChan <- "skipped"
		}
	}
}

func formatHTML(content []byte) []byte {
	return gohtml.FormatBytes(content)
}

func findFiles(dir string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(path, ".html") {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

func main() {
	dir := flag.String("dir", "sites", "directory to format")
	numWorkers := flag.Int("workers", runtime.NumCPU(), "number of workers")
	flag.Parse()

	fmt.Printf("\n🚀 Starting format process...\n\n")
	fmt.Printf("🔍 Finding files to format...\n")

	files, err := findFiles(*dir)
	if err != nil {
		fmt.Printf("Error finding files: %v\n", err)
		os.Exit(1)
	}

	if len(files) == 0 {
		fmt.Println("No files found to format.")
		os.Exit(0)
	}

	fmt.Printf("\n📊 Found %d files, using %d workers\n\n", len(files), *numWorkers)

	// Create progress bar
	bar := progressbar.NewOptions(len(files),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionShowCount(),
		progressbar.OptionShowIts(),
		progressbar.OptionSetItsString("files"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[green]=[reset]",
			SaucerHead:    "[green]>[reset]",
			SaucerPadding: " ",
			BarStart:      "[",
			BarEnd:        "]",
		}))

	// Create channels
	tasks := make(chan FileTask, *numWorkers)
	errChan := make(chan error, len(files))
	statChan := make(chan string, len(files))

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < *numWorkers; i++ {
		wg.Add(1)
		go worker(i, tasks, &wg)
	}

	// Track stats
	stats := &Stats{}
	startTime := time.Now()

	// Start stats collector
	go func() {
		for stat := range statChan {
			switch stat {
			case "formatted":
				atomic.AddUint64(&stats.formatted, 1)
			case "skipped":
				atomic.AddUint64(&stats.skipped, 1)
			case "error":
				atomic.AddUint64(&stats.errors, 1)
			}
			bar.Add(1)
		}
	}()

	// Send tasks
	for _, file := range files {
		content, err := os.ReadFile(file)
		if err != nil {
			fmt.Printf("Error reading %s: %v\n", file, err)
			atomic.AddUint64(&stats.errors, 1)
			continue
		}

		tasks <- FileTask{
			path:     file,
			content:  content,
			errChan:  errChan,
			statChan: statChan,
		}
	}

	// Close tasks channel and wait for workers
	close(tasks)
	wg.Wait()

	// Close stat channel and wait for final updates
	close(statChan)
	close(errChan)

	// Print summary
	duration := time.Since(startTime).Seconds()
	fmt.Printf("\n%s", stats)
	fmt.Printf("\nTotal time: %.2fs\n", duration)

	// Print any errors that occurred
	for err := range errChan {
		fmt.Printf("Error: %v\n", err)
	}
}
