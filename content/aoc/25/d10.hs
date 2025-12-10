module Main where

import Data.Bits (shiftL, xor, (.|.))
import qualified Data.IntSet as IS
import qualified Data.List as L
import Data.Ratio ((%), denominator)

type Mask = Int
type Button = [Int]
type Constraint = ([Rational], Rational) -- c · x <= b

parseLine :: String -> (String, [Button], [Int])
parseLine s = (patternStr, buttons, targets)
  where
    -- pattern between [ ]
    (patternStr, rest) = span (/= ']') . tail $ dropWhile (/= '[') s
    -- buttons between ( )
    buttons = goButtons rest
    goButtons :: String -> [Button]
    goButtons str =
      case dropWhile (/= '(') str of
        "" -> []
        (_ : xs) ->
          let (nums, rest) = span (/= ')') xs
           in parseInts nums : goButtons (drop 1 rest)
    -- targets between { }
    targets =
      let inside = takeWhile (/= '}') . tail $ dropWhile (/= '{') s
       in parseInts inside

parseInts :: String -> [Int]
parseInts = map read . words . map (\c -> if c == ',' then ' ' else c)

-- part 1: BFS over light states (<= 2^10 states)
-- XOR is communicative and associative, so we only really care about pressing the button 0 or 1 time.
-- hence this is a bitmask operations.
-- state space is 2^n possible light configurations (n = number of lights)
--
-- Given that we run from state 0 (all off), each edge would be pressing the button, which XORs the current state with given button's mask.
patternMask :: String -> Mask
patternMask = go 0 0
  where
    go _ acc [] = acc
    go i acc (c : cs) =
      let bit = if c == '#' then 1 else 0
       in go (i + 1) (acc .|. (bit `shiftL` i)) cs

buttonMask :: Button -> Mask
buttonMask = foldl (\m i -> m `xor` (1 `shiftL` i)) 0

minPressLights :: Mask -> [Mask] -> Int
minPressLights target masks = bfs (IS.singleton 0) [(0, 0)]
  where
    bfs _ [] = error "unreachable"
    bfs seen ((state, d) : q)
      | state == target = d
      | otherwise =
          let nexts =
                [ state' | m <- masks, let state' = state `xor` m, not (IS.member state' seen)
                ]
              seen' = foldl (flip IS.insert) seen nexts
              q' = q ++ zip nexts (repeat (d + 1))
           in bfs seen' q'

-- part 2: we can frame it as minimize sum(x_i) subject to Ax = b with x >= 0 integer
--
-- A[i][j] = 1 if button j affects counter i, else 0
-- b[j] = target value for counter i
-- x[j] = number of time to press button j
--
-- Here, we use Gauss elimination for RREF [A|b], such that each pivot row gives x_pivot = b_i - sum(coeff * x_free). If the system is fully determined then we have an unique solution, otherwise we brute-force the bounded space to find the miminum total presses.
gaussJordan :: [[Rational]] -> [Rational] -> ([[Rational]], [Rational], [Int])
gaussJordan a b = go 0 0 a b []
  where
    rows = length a
    cols = length (head a)

    go _ _ mat rhs pivots | null mat = ([], [], pivots)
    go r c mat rhs pivots
      | r >= rows || c >= cols = (mat, rhs, pivots)
      | otherwise =
          case findPivot r c mat of
            Nothing -> go r (c + 1) mat rhs pivots
            Just p ->
              let (mat1, rhs1) = swapRows r p mat rhs
                  pivotVal = (mat1 !! r) !! c
                  (mat2, rhs2) = normalizeRow r c pivotVal mat1 rhs1
                  (mat3, rhs3) = eliminate r c mat2 rhs2
                  (rest, restRhs, ps) = go (r + 1) (c + 1) mat3 rhs3 (pivots ++ [c])
               in (rest, restRhs, ps)

    findPivot r c mat = L.find (\i -> (mat !! i) !! c /= 0) [r .. rows - 1]
    swapRows i j mat rhs =
      let mat' = swap i j mat
          rhs' = swap i j rhs
       in (mat', rhs')
    swap i j xs =
      let xi = xs !! i
          xj = xs !! j
       in [ if k == i then xj else if k == j then xi else xs !! k | k <- [0 .. length xs - 1] ]
    normalizeRow r c pivot mat rhs =
      let factor = 1 / pivot
          mat' =
            [ if i == r
                then [ if j < c then mat !! i !! j else mat !! i !! j * factor | j <- [0 .. cols - 1] ]
                else mat !! i
              | i <- [0 .. rows - 1]
            ]
          rhs' = [ if i == r then rhs !! i * factor else rhs !! i | i <- [0 .. rows - 1] ]
       in (mat', rhs')
    eliminate r c mat rhs =
      let mat' =
            [ if i == r
                then mat !! i
                else
                  let factor = mat !! i !! c
                   in [ mat !! i !! j - factor * (mat !! r !! j) | j <- [0 .. cols - 1] ]
              | i <- [0 .. rows - 1]
            ]
          rhs' =
            [ if i == r
                then rhs !! i
                else rhs !! i - (mat !! i !! c) * (rhs !! r)
              | i <- [0 .. rows - 1]
            ]
       in (mat', rhs')

-- solve square system, returning Nothing if singular
solveSquare :: [[Rational]] -> [Rational] -> Maybe [Rational]
solveSquare mat rhs =
  let n = length mat
      (mat', rhs', _) = gaussJordan mat rhs
      leadCols = map (L.findIndex (/= 0)) mat'
   in if any (== Nothing) leadCols
        then Nothing
        else
          let xs = replicate n 0
              sol =
                foldr
                  ( \(i, mrow, bval) acc ->
                      case L.findIndex (/= 0) mrow of
                        Nothing -> acc
                        Just c ->
                          let rest = sum [mrow !! j * acc !! j | j <- [c + 1 .. n - 1]]
                              val = bval - rest
                           in take c acc ++ [val] ++ drop (c + 1) acc
                  )
                  xs
                  (zip3 [0 ..] mat' rhs')
           in Just sol

combinations :: Int -> [a] -> [[a]]
combinations 0 _ = [[]]
combinations _ [] = []
combinations k (x : xs) = map (x :) (combinations (k - 1) xs) ++ combinations k xs

boundsForFree :: [([Rational], Rational)] -> Int -> ([Rational], [Rational])
boundsForFree constraints f =
  let subsets = combinations f [0 .. length constraints - 1]
      initMin = replicate f (Nothing :: Maybe Rational)
      initMax = replicate f (0 :: Rational)
      (mn, mx) =
        foldl
          ( \(mins, maxs) subset ->
              let mat = [fst (constraints !! i) | i <- subset]
                  rhs = [snd (constraints !! i) | i <- subset]
               in case solveSquare mat rhs of
                    Nothing -> (mins, maxs)
                    Just sol ->
                      if feasible sol constraints
                        then
                          ( zipWith (\m v -> Just $ maybe v (min v) m) mins sol,
                            zipWith max maxs sol
                          )
                        else (mins, maxs)
          )
          (initMin, initMax)
          subsets
   in (map (maybe 0 id) mn, mx)
  where
    feasible sol cons = all (\(c, b) -> dot c sol <= b) cons
    dot v w = sum (zipWith (*) v w)

minPressJolts :: [[Int]] -> [Int] -> Int
minPressJolts buttons target =
  let m = length target
      n = length buttons
      a = [[if i `elem` btn then 1 else 0 | btn <- buttons] | i <- [0 .. m - 1]]
      aRat = map (map fromIntegral) a
      bRat = map fromIntegral target
      (aRed, bRed, pivots) = gaussJordan aRat bRat
      free = [c | c <- [0 .. n - 1], c `notElem` pivots]
      fCount = length free
      rowInfo =
        [ (bRed !! r, [aRed !! r !! f | f <- free])
          | (r, p) <- zip [0 ..] pivots,
            let _ = p
        ]
   in if fCount == 0
        then
          let pivotVals = map fst rowInfo
           in if all (\x -> x >= 0 && denominatorIsOne x) pivotVals
                then sum (map round pivotVals)
                else error "no feasible solution"
        else
          let constraints =
                [ (coeff, b) | (b, coeff) <- rowInfo ]
                  ++ [ (map (negAt j) [0 .. fCount - 1], 0) | j <- [0 .. fCount - 1] ]
              (mins, maxs) = boundsForFree constraints fCount
              bounds =
                [ (max 0 (ceilingR m'), floorR x) | (m', x) <- zip mins maxs ]
           in search bounds rowInfo

negAt :: Int -> Int -> Rational
negAt j idx = if idx == j then (-1) else 0

denominatorIsOne :: Rational -> Bool
denominatorIsOne x = denominator x == 1

ceilingR :: Rational -> Int
ceilingR = ceiling . fromRational

floorR :: Rational -> Int
floorR = floor . fromRational

search :: [(Int, Int)] -> [(Rational, [Rational])] -> Int
search bounds rows = go 0 []
  where
    fCount = length bounds
    go idx vals
      | idx == fCount =
          let pivots = map (\(b, coeff) -> b - sum (zipWith (*) coeff (map fromIntegral vals))) rows
           in if all (\v -> v >= 0 && denominatorIsOne v) pivots
                then sum vals + sum (map round pivots)
                else maxBound `div` 4
      | otherwise =
          let (lo, hi) = bounds !! idx
           in foldl
                (\acc v -> min acc (go (idx + 1) (vals ++ [v])))
                (maxBound `div` 4)
                [lo .. hi]

main :: IO ()
main = do
  raw <- readFile "d10.txt"
  let machines = lines raw
      parsed = map parseLine machines

      part1 =
        sum
          [ let target = patternMask pat
                masks = map buttonMask btns
             in minPressLights target masks
            | (pat, btns, _) <- parsed
          ]

      part2 =
        sum
          [ minPressJolts btns targets
            | (_, btns, targets) <- parsed
          ]

  putStrLn $ "p1: " ++ show part1
  putStrLn $ "p2: " ++ show part2
