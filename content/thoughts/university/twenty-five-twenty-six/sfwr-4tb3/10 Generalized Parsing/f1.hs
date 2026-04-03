-- AST
data Exp
  = Sum Exp Exp
  | Diff Exp Exp
  | Prod Exp Exp
  | Quot Exp Exp
  | Id Char
  | Fun Char [Exp]
  deriving (Eq, Show)

-- expression, term, factor build an AST and return (Exp, remainder)
expression :: String -> (Exp, String)
expression s = moreterms (term s)
  where
    moreterms :: (Exp, String) -> (Exp, String)
    moreterms (p, '+':t) =
      let (q, u) = term t in moreterms (Sum p q, u)
    moreterms (p, '-':t) =
      let (q, u) = term t in moreterms (Diff p q, u)
    moreterms any = any

term :: String -> (Exp, String)
term s = morefactors (factor s)
  where
    morefactors :: (Exp, String) -> (Exp, String)
    morefactors (p, '*':t) =
      let (q, u) = factor t in morefactors (Prod p q, u)
    morefactors (p, '/':t) =
      let (q, u) = factor t in morefactors (Quot p q, u)
    morefactors any = any

factor :: String -> (Exp, String)
factor ('(':t) =
  case expression t of
    (p, ')':u) -> (p, u)
    _          -> error "missing )"
factor (id:'(':t) | id >= 'a' && id <= 'z' =
  case exprList t of
    (ps, ')':u) -> (Fun id ps, u)
    _           -> error "missing )"
factor (id:t) | id >= 'a' && id <= 'z' = (Id id, t)
factor (_:_) = error "id or ( expected"
factor []    = error "unexpected end"

exprList :: String -> ([Exp], String)
exprList s =
  case expression s of
    (p, ',':u) -> let (q, n) = exprList u in (p:q, n)
    (p, t)     -> ([p], t)

-- Examples:
-- expression "a*(b+c)" -> (Prod (Id 'a') (Sum (Id 'b') (Id 'c')),"")
-- expression "a+f(x,y)" -> (Sum (Id 'a') (Fun 'f' [Id 'x',Id 'y']),"")
-- expression "a+(b/c)"  -> (Sum (Id 'a') (Quot (Id 'b') (Id 'c')),"")
