"""Apply the two Korean text normalizations used before notebook masking."""

import argparse
import json
import re
from pathlib import Path


NORMALIZATION_DICT = {'주의 바랍니다': '유의 바랍니다'}


def replace_expressions(text):
    # Apply the notebook's common-expression replacement to both text fields.
    for original, replacement in NORMALIZATION_DICT.items():
        text = text.replace(original, replacement)
    return text


def normalize_directory(directory):
    count = 0
    for path in sorted(Path(directory).glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        for field in ('korean_text', 'masked_korean_text'):
            if field in data:
                data[field] = replace_expressions(data[field])
        if 'korean_text' in data:
            data['korean_text'] = re.sub(r'(금일|당일)', '오늘', data['korean_text'])
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_dirs', nargs='+', help='Normalized train and validation JSON directories.')
    args = parser.parse_args()
    for directory in args.json_dirs:
        print(f'{directory}: normalized text in {normalize_directory(directory)} files')


if __name__ == '__main__':
    main()
