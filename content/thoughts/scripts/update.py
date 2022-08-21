# /// script
# requires-python = ">=3.11"
# ///

import os, argparse, glob


# fmt: off
def update_index_md_with_pdfs(directory:str,index_file:str)->None:
  (pdf_files := [f for f in [os.path.basename(f) for f in glob.glob(os.path.join(directory, "*.pdf"))] if not f.endswith('.ignore.pdf')]).sort()
  with open(index_file,'r',encoding='utf-8') as file:lines=file.readlines()
  start=end=None
  for i, line in enumerate(lines):
    if 'update: start' in line: start = i + 1
    elif 'update: end' in line: end = i; break

  if start is None or end is None:raise ValueError('Could not find update section markers in the index.md file.')
  lines = lines[:start]+[f'- [[thoughts/{directory.split("/")[-1]}/{pdf}|{pdf[:-4]}]]\n' for pdf in pdf_files]+lines[end:]
  with open(index_file,'w',encoding='utf-8') as file:file.writelines(lines)
# fmt: on

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Update index.md with PDF files from a directory')
  parser.add_argument(
    '--directory',
    '-d',
    type=str,
    default=os.path.dirname(os.path.realpath(__file__)),
    help='Directory containing PDF files (default: current script directory)',
  )
  args = parser.parse_args()

  directory = args.directory
  index_file = os.path.join(directory, 'index.md')
  update_index_md_with_pdfs(directory, index_file)
