#!/usr/bin/env python3
"""
A2: Syntax Diagrams for Arithmetic Expressions

Grammar:
    expression  →  [ '+' | '–' ] term { ( '+' | '–' ) term }
    term        →  factor { ( '×' | '/' ) factor }
    factor      →  number | identifier | '(' expression ')'

Part 1: LaTeX with mdwtools syntax package
    - File: prettyprintgrammar.tex
    - Compile: pdflatex prettyprintgrammar.tex
    - Output: prettyprintgrammar.pdf

Part 3: RR Railroad Diagram Generator
    - Grammar: expression_grammar.ebnf (W3C EBNF format)
    - Generated: img/railroad_diagrams.xhtml (embedded SVGs)
    - Generated: img/diagram/*.svg (individual diagrams)

    To regenerate:
        java -jar rr-2.2-SNAPSHOT-java11/rr.war -out:img/railroad_diagrams.xhtml expression_grammar.ebnf
        java -jar rr-2.2-SNAPSHOT-java11/rr.war -noembedded -out:img/railroad.zip expression_grammar.ebnf

    Or use the web interface:
        java -jar rr-2.2-SNAPSHOT-java11/rr.war -gui
        Then open http://localhost:8080/
"""

import subprocess
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).parent


def compile_latex():
  tex_file = BASE_DIR / 'prettyprintgrammar.tex'
  print(f'Compiling {tex_file}...')
  result = subprocess.run(
    ['pdflatex', '-interaction=nonstopmode', str(tex_file)], cwd=BASE_DIR, capture_output=True, text=True
  )
  if result.returncode == 0:
    print(f'Success! Output: {BASE_DIR / "prettyprintgrammar.pdf"}')
  else:
    print(f'Error: {result.stderr}')
  return result.returncode == 0


def generate_railroad_diagrams():
  jar_file = BASE_DIR / 'rr-2.2-SNAPSHOT-java11' / 'rr.war'
  grammar_file = BASE_DIR / 'expression_grammar.ebnf'
  output_file = BASE_DIR / 'img' / 'railroad_diagrams.xhtml'

  print(f'Generating railroad diagrams from {grammar_file}...')
  result = subprocess.run(
    ['java', '-jar', str(jar_file), f'-out:{output_file}', str(grammar_file)], capture_output=True, text=True
  )
  if result.returncode == 0:
    print(f'Success! Output: {output_file}')
  else:
    print(f'Error: {result.stderr}')
  return result.returncode == 0


def start_rr_gui():
  jar_file = BASE_DIR / 'rr-2.2-SNAPSHOT-java11' / 'rr.war'
  print('Starting RR GUI server on http://localhost:8080/')
  print('Press Ctrl+C to stop.')
  subprocess.run(['java', '-jar', str(jar_file), '-gui'])


def open_outputs():
  pdf = BASE_DIR / 'prettyprintgrammar.pdf'
  xhtml = BASE_DIR / 'img' / 'railroad_diagrams.xhtml'

  if pdf.exists():
    print(f'Opening {pdf}')
    webbrowser.open(f'file://{pdf}')
  if xhtml.exists():
    print(f'Opening {xhtml}')
    webbrowser.open(f'file://{xhtml}')


def main():
  print('=' * 60)
  print('A2: Syntax Diagrams for Arithmetic Expressions')
  print('=' * 60)

  print('\n--- Grammar ---')
  print("expression  →  [ '+' | '–' ] term { ( '+' | '–' ) term }")
  print("term        →  factor { ( '×' | '/' ) factor }")
  print("factor      →  number | identifier | '(' expression ')'")

  print('\n--- Files ---')
  files = [
    ('LaTeX source', 'prettyprintgrammar.tex'),
    ('LaTeX PDF', 'prettyprintgrammar.pdf'),
    ('W3C EBNF grammar', 'expression_grammar.ebnf'),
    ('Railroad diagrams', 'img/railroad_diagrams.xhtml'),
    ('Expression SVG', 'img/diagram/expression.svg'),
    ('Term SVG', 'img/diagram/term.svg'),
    ('Factor SVG', 'img/diagram/factor.svg'),
  ]
  for desc, path in files:
    full_path = BASE_DIR / path
    status = '✓' if full_path.exists() else '✗'
    print(f'  {status} {desc}: {path}')

  print('\n--- Commands ---')
  print('  python a2.py compile   - Compile LaTeX to PDF')
  print('  python a2.py railroad  - Generate railroad diagrams')
  print('  python a2.py gui       - Start RR GUI server')
  print('  python a2.py open      - Open generated outputs')


if __name__ == '__main__':
  import sys

  if len(sys.argv) > 1:
    cmd = sys.argv[1]
    if cmd == 'compile':
      compile_latex()
    elif cmd == 'railroad':
      generate_railroad_diagrams()
    elif cmd == 'gui':
      start_rr_gui()
    elif cmd == 'open':
      open_outputs()
    else:
      print(f'Unknown command: {cmd}')
      main()
  else:
    main()
