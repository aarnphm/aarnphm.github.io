type Source = (Int, String)
type Result = Either (Int, String) Source

sourceError :: Int -> String -> Result
sourceError p m = Left (p, m)

expression :: Source -> Result
expression s = do
  (q, rest) <- term s
  case rest of
    '+':t -> expression (q + 1, t)
    '-':t -> expression (q + 1, t)
    _     -> Right (q, rest)

term :: Source -> Result
term s = do
  (q, rest) <- factor s
  case rest of
    '*':t -> term (q + 1, t)
    '/':t -> term (q + 1, t)
    _     -> Right (q, rest)

factor :: Source -> Result
factor (p, '(':t) = do
  (q, rest) <- expression (p + 1, t)
  case rest of
    ')':u -> Right (q + 1, u)
    _     -> sourceError q "missing )"
factor (p, id:'(':t) | id >= 'a' && id <= 'z' = do
  (r, rest) <- exprList (p + 2, t)
  case rest of
    ')':u -> Right (r + 1, u)
    _     -> sourceError r "missing )"
factor (p, id:t) | id >= 'a' && id <= 'z' = Right (p + 1, t)
factor (p, _:_) = sourceError p "id or ( expected"
factor (p, [])  = sourceError p "unexpected end"

exprList :: Source -> Result
exprList s = do
  (q, rest) <- expression s
  case rest of
    ',':t -> exprList (q + 1, t)
    _     -> Right (q, rest)

parse :: String -> IO ()
parse s =
  case expression (1, s) of
    Right (p, _) -> putStrLn (show (p - 1) ++ " characters parsed")
    Left (p, m)  -> putStrLn ("error at " ++ show p ++ ": " ++ m)

-- Examples:
-- parse "a+(b/c)"   -- 7 characters parsed
-- parse "a+b/c"     -- 5 characters parsed
-- parse "a*b*c"     -- 5 characters parsed
-- parse "a+(b"      -- error at 5: missing )
-- parse "a+("       -- error at 4: unexpected end
-- parse "1+*"       -- error at 1: id or ( expected
-- parse "a+"        -- error at 3: unexpected end
-- parse "ab"        -- 1 characters parsed
