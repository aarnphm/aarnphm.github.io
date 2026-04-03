s :: String -> String
s ('a':t) =
  case s t of
    ('c':u) -> u
s ('b':t) = t
