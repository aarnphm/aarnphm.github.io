# /// script
# requires-python = ">=3.11"
# ///

import os

def update_index_md_with_pdfs(directory:str,index_file:str)->None:
  (pdf_files:=[f for f in os.listdir(directory) if f.endswith('.pdf') and not f.endswith('.ignore.pdf')]).sort()
  with open(index_file,'r',encoding='utf-8') as file:lines=file.readlines()
  start=end=None
  for i, line in enumerate(lines):
    if 'update: start' in line: start = i + 1
    elif 'update: end' in line: end = i; break

  if start is None or end is None:raise ValueError('Could not find update section markers in the index.md file.')
  lines = lines[:start]+[f'- [[thoughts/papers/{pdf}|{pdf[:-4]}]]\n' for pdf in pdf_files]+lines[end:]
  with open(index_file,'w',encoding='utf-8') as file:file.writelines(lines)

if __name__=='__main__':update_index_md_with_pdfs((current_directory:=os.path.dirname(os.path.realpath(__file__))),os.path.join(current_directory,'index.md'))
