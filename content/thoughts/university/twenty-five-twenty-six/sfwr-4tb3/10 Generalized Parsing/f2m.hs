type Source = (Int, String)

expression :: Source -> Maybe Source
expression s = do
  (q, rest) <- term s
  case rest of
    '+':t -> expression (q + 1, t)
    '-':t -> expression (q + 1, t)
    _     -> Just (q, rest)

term :: Source -> Maybe Source
term s = do
  (q, rest) <- factor s
  case rest of
    '*':t -> term (q + 1, t)
    '/':t -> term (q + 1, t)
    _     -> Just (q, rest)

factor :: Source -> Maybe Source
factor (p, '(':t) = do
  (q, rest) <- expression (p + 1, t)
  case rest of
    ')':u -> Just (q + 1, u)
    _     -> Nothing
factor (p, id:'(':t)
  | id >= 'a' && id <= 'z' = do
      (r, rest) <- exprList (p + 2, t)
      case rest of
        ')':u -> Just (r + 1, u)
        _     -> Nothing
factor (p, id:t)
  | id >= 'a' && id <= 'z' =
      Just (p + 1, t)
factor (_, _) = Nothing

exprList :: Source -> Maybe Source
exprList s = do
  (q, rest) <- expression s
  case rest of
    ',':t -> exprList (q + 1, t)
    _     -> Just (q, rest)

parse :: String -> IO ()
parse s =
  case expression (1, s) of
    Just (p, _) ->
      putStrLn (show (p - 1) ++ " characters parsed")
    Nothing ->
      putStrLn "parse error"

-- parse "a+(b/c)"   -- 7 characters parsed
-- parse "a+b/c"     -- 5 characters parsed
-- parse "a*b*c"     -- 5 characters parsed
-- parse "a+("       -- parse error
-- parse "1+*"       -- parse error
-- parse "a+"        -- parse error
-- parse "ab"        -- 1 characters parsed
