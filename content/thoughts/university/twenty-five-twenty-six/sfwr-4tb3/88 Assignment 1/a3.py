from pathlib import Path

import svgling
from nltk import Tree

tree1 = Tree('S', ['a', Tree('S', ['ε']), 'b', Tree('S', ['a', Tree('S', ['ε']), 'b', Tree('S', ['ε'])])])
tree2 = Tree('S', ['a', Tree('S', ['b', Tree('S', ['ε']), 'a', Tree('S', ['ε'])]), 'b', Tree('S', ['ε'])])


def save_tree(tree: Tree, output_path: Path) -> None:
  svg = svgling.draw_tree(tree)
  svg_content = svg._repr_svg_()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(svg_content)


if __name__ == '__main__':
  here = Path(__file__).parent
  save_tree(tree1, here / 'img' / 'parse_tree_a3_1.svg')
  save_tree(tree2, here / 'img' / 'parse_tree_a3_2.svg')
  print('saved parse_tree_a3_1.svg and parse_tree_a3_2.svg')
