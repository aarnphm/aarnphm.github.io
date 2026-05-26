import Control.Exception (ErrorCall (..), evaluate, try)

type Source = (Int, String)

sourceError :: Int -> String -> a
sourceError p m = error ("error at " ++ show p ++ ": " ++ m)

expression :: Source -> Source
expression s =
  case term s of
    (q, '+':t) -> expression (q + 1, t)
    (q, '-':t) -> expression (q + 1, t)
    any        -> any

term :: Source -> Source
term s =
  case factor s of
    (q, '*':t) -> term (q + 1, t)
    (q, '/':t) -> term (q + 1, t)
    any        -> any

factor :: Source -> Source
factor (p, '(':t) =
  case expression (p + 1, t) of
    (q, ')':u) -> (q + 1, u)
    (q, _)     -> sourceError q "missing )"
factor (p, id:'(':t) | id >= 'a' && id <= 'z' =
  case exprList (p + 2, t) of
    (r, ')':u) -> (r + 1, u)
    (r, _)     -> sourceError r "missing )"
factor (p, id:t) | id >= 'a' && id <= 'z' = (p + 1, t)
factor (p, _ : _) = sourceError p "id or ( expected"
factor (p, [])    = sourceError p "unexpected end"

exprList :: Source -> Source
exprList s =
  case expression s of
    (q, ',':t) -> exprList (q + 1, t)
    any        -> any

parse :: String -> IO ()
parse s = do
  r <- try (evaluate (expression (1, s))) :: IO (Either ErrorCall Source)
  case r of
    Right (p, _)         -> putStrLn (show (p - 1) ++ " characters parsed")
    Left (ErrorCall msg) -> putStrLn msg

-- Examples:
-- parse "a+(b/c)"   -- 7 characters parsed
-- parse "a+b/c"     -- 5 characters parsed
-- parse "a*b*c"     -- 5 characters parsed
-- parse "a+(b"      -- error at 5: missing )
-- parse "a+("       -- error at 4: unexpected end
-- parse "1+*"       -- error at 1: id or ( expected
-- parse "a+"        -- error at 3: unexpected end
-- parse "ab"        -- 1 characters parsed
