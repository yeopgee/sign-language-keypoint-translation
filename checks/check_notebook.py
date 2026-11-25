"""Check Python/notebook syntax and shared definitions without running experiments.

Run from any directory with: python checks/check_notebook.py
CLI wrappers and exploratory branches differ intentionally; see the README.
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = {'model.py': ['Attention', 'Encoder', 'Decoder', 'Seq2Seq'], 'mask_text.py': ['grouped_region_mask_korean_text_final', 'apply_masking_to_json_dir'], 'train.py': ['LabelSmoothingLoss']}


def normalized(node):
    # Comments, formatting, and explanatory docstrings do not change behavior.
    node = ast.parse(ast.unparse(node)).body[0]
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and item.id == 'tv_models':
            item.id = 'models'  # The GAN module aliases torchvision.models.
        if hasattr(item, 'body') and isinstance(item.body, list):
            item.body = [statement for statement in item.body
                         if not (isinstance(statement, ast.Expr)
                                 and isinstance(statement.value, ast.Constant)
                                 and isinstance(statement.value.value, str))]
    return ast.dump(node, include_attributes=False)


def main():
    notebook = next((ROOT / 'notebooks').glob('*.ipynb'))
    cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
    definitions = {}
    code_cells = 0
    for index, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        tree = ast.parse(source, filename=f'{notebook.name}:cell{index}')
        code_cells += 1
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions.setdefault(node.name, []).append(normalized(node))
    checked = 0
    for path in ROOT.glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        names = SHARED.get(path.name, [])
        found = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for name in names:
            if normalized(found[name]) not in definitions.get(name, []):
                raise AssertionError(f'{path.name}:{name} differs from the notebook')
            checked += 1
    print(f'Checked all Python modules and {code_cells} notebook code cells; '
          f'{checked} shared definitions match.')


if __name__ == '__main__':
    main()
