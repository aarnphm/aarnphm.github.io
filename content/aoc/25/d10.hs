import Data.Array (Array, listArray, bounds, (!))
import qualified Data.Array as A

type Grid = Array (Int, Int) Char

readInput :: FilePath -> IO [String]
readInput = fmap lines . readFile

parseGrid :: [String] -> Grid
parseGrid rows =
  let h = length rows
      w = if null rows then 0 else length (head rows)
  in listArray ((0, 0), (h - 1, w - 1))
       [rows !! r !! c | r <- [0..h-1], c <- [0..w-1]]

gridBounds :: Grid -> ((Int, Int), (Int, Int))
gridBounds = bounds

p1 :: Grid -> Int
p1 _grid = 0

p2 :: Grid -> Int
p2 _grid = 0

main :: IO ()
main = do
  input <- readInput "d10.txt"
  let grid = parseGrid input
  putStrLn $ "p1: " ++ show (p1 grid)
  putStrLn $ "p2: " ++ show (p2 grid)
