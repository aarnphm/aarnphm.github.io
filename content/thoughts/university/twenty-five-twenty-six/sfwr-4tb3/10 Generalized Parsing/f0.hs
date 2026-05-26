a :: String -> String
a ('a':t) =
  case a t of
    ('c':u) -> u
a ('b':t) = t
