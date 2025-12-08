import Foundation

func readInput(_ filename: String) -> [String] {
  let path = URL(fileURLWithPath: filename)
  guard let contents = try? String(contentsOf: path, encoding: .utf8) else {
    fatalError("could not read \(filename)")
  }
  return contents.split(separator: "\n", omittingEmptySubsequences: false)
    .map(String.init)
    .filter { !$0.isEmpty }
}

func readGrid(_ filename: String) -> [[Character]] {
  readInput(filename).map { Array($0) }
}

func p1(_ grid: [[Character]]) -> Int {
  0
}

func p2(_ grid: [[Character]]) -> Int {
  0
}

let grid = readGrid("d9.txt")
print("p1: \(p1(grid))")
print("p2: \(p2(grid))")
