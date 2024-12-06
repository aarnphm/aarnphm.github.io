package main

import (
	"flag"
	"fmt"
	"log"
	"mime"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
)

var allowedExtensions = map[string]bool{
	".html":  true,
	".css":   true,
	".js":    true,
	".json":  true,
	".xml":   true,
	".mp4":   true,
	".webp":  true,
	".txt":   true,
	".md":    true,
	".jpg":   true,
	".jpeg":  true,
	".png":   true,
	".gif":   true,
	".svg":   true,
	".ico":   true,
	".pdf":   true,
	".woff":  true,
	".woff2": true,
	".go":    true,
	".py":    true,
}

func main() {
	port := flag.Int("port", 8080, "Port to serve on")
	dir := flag.String("dir", "public", "Directory to serve files from")
	flag.Parse()

	publicDir, err := filepath.Abs(*dir)
	if err != nil {
		log.Fatal(err)
	}

	if _, err := os.Stat(publicDir); os.IsNotExist(err) {
		log.Fatalf("Directory %s does not exist", publicDir)
	}

	// Register custom MIME types
	mime.AddExtensionType(".woff2", "font/woff2")
	mime.AddExtensionType(".woff", "font/woff")

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Clean the URL path
		urlPath := path.Clean(r.URL.Path)

		// Remove trailing slash except for root
		if urlPath != "/" {
			urlPath = strings.TrimSuffix(urlPath, "/")
		}

		// Construct full file path
		fullPath := filepath.Join(publicDir, filepath.FromSlash(urlPath))

		// Get file info
		fi, err := os.Stat(fullPath)
		if err != nil {
			if os.IsNotExist(err) {
				// Try adding .html extension
				htmlPath := fullPath + ".html"
				if fi, err = os.Stat(htmlPath); err == nil {
					fullPath = htmlPath
				} else {
					// Try index.html for directory-style URLs
					indexPath := filepath.Join(fullPath, "index.html")
					if fi, err = os.Stat(indexPath); err == nil {
						fullPath = indexPath
					} else {
						http.NotFound(w, r)
						return
					}
				}
			} else {
				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
				return
			}
		}

		// Check if path is a directory
		if fi.IsDir() {
			indexPath := filepath.Join(fullPath, "index.html")
			if _, err := os.Stat(indexPath); err == nil {
				fullPath = indexPath
			} else {
				http.NotFound(w, r)
				return
			}
		}

		// Verify file extension
		ext := strings.ToLower(filepath.Ext(fullPath))
		if !allowedExtensions[ext] {
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}

		// Set appropriate headers
		w.Header().Set("Cache-Control", "public, max-age=31536000")
		w.Header().Set("Vary", "Accept-Encoding")

		// Detect and set content type
		contentType := mime.TypeByExtension(ext)
		if contentType != "" {
			w.Header().Set("Content-Type", contentType)
		}

		// Serve the file
		http.ServeFile(w, r, fullPath)
	})

	addr := fmt.Sprintf(":%d", *port)
	log.Printf("Serving files from %s on http://localhost%s", publicDir, addr)
	if err := http.ListenAndServe(addr, handler); err != nil {
		log.Fatal(err)
	}
}
