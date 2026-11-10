"""Mask location and time slots in Korean text annotations."""
import argparse
import json
import re
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm

def grouped_region_mask_korean_text_final(text: str) -> Tuple[str, Dict[str, str]]:
    slot_values = {}
    REGION_END = r'(?=\s|[.,은는이가의에에서와]|[0-9]|$)'
    JOSA = r"(은|는|이|가|에|에서|로|를|과|와|도|까|까지|만|으로|요|도)"
    EXCLUDE_PATTERN = r'\b[가-힣]{1,10}(사고|중|발생)\b'
    EXCLUDE_FOR_STATION = r'(사거리|삼거리|오거리|도로|고가도로|역사|하상도로|지하차도|차도|우회도로)'

    patterns = [
        (r'([0-2]?[0-9]:[0-5][0-9])', lambda m: [("<시간>", m.group(1))]),
        (r'([0-9]{1,2}시\s?[0-9]{0,2}분?)', lambda m: [("<시간>", m.group(1))]),
        # ✅ 먼저 복합 지역 단어 (산 들어간 지역 방어용)
        (r'\b[가-힣]{2,10}(시|군|구|서구|북구|남구|동구)\b', lambda m: [("<지역>", m.group(0))]),
    
        # ✅ 산 (예외 단어 방어 포함)
        (r'(?<!부산)(?<!확산)(?<!등산)(?<!우산)(?<!횡산)([가-힣]{1,10}산)(?=\s|[.,은는이가의에에서와]|[0-9]|$)', 
         lambda m: [("<산>", m.group(1))]),
        (r'([가-힣0-9]{1,20}(사거리|삼거리|오거리|육거리)?\s?(고가도로|하상도로|지하차도|도로|차도|우회도로))',
         lambda m: [("<도로>", m.group(0))]),
        (r'([가-힣]{1,20}(사거리|삼거리|오거리|육거리))',
         lambda m: [("<도로>", m.group(1))]),
        (rf'([가-힣]{{1,10}}(읍|면|동|리))\s([가-힣]{{1,10}}(리)){REGION_END}',
         lambda m: [("<지역>", f"{m.group(1)} {m.group(3)}")]),
        (rf'([가-힣]{{1,10}}(시|도|군|구))\s?([가-힣]{{1,10}}(동|읍|면|리)){REGION_END}',
         lambda m: [("<지역>", f"{m.group(1)} {m.group(3)}")]),
        (r'\b([가-힣]{1,10}(시|도|군|구|읍|면|동|리))\b', lambda m: [("<지역>", m.group(1))]),
        (r'([가-힣]{2,10}[0-9]{1,2}가)', lambda m: [("<주소>", m.group(1))]),
        (r'([0-9]{1,5}-[0-9]{1,5}(번지)?)', lambda m: [("<주소>", m.group(1))]),
        (r'([가-힣]{1,20}(로|길)\s?[0-9]{1,4}(-[0-9]{1,4})?(번지)?)', lambda m: [("<주소>", m.group(1))]),
        (r'([0-9]{1,4}동)', lambda m: [("<주소>", m.group(1))]),
        (r'([가-힣A-Za-z0-9]{1,30}(아파트|오피스텔|주택|고시원|빌딩|건물|맨션|연립주택|사옥|하우스|타워|상가|점포|회관|센터|몰|모델하우스|아울렛|백화점))(?=\s|' + JOSA + r'|\b)',
         lambda m: [("<건물>", m.group(1))]),
        (r'\b(건물|아파트|빌딩|주택|하우스)\b', lambda m: [("<건물>", m.group(1))]),
        (r'([가-힣A-Za-z0-9]{1,30}(호텔|모텔|펜션))', lambda m: [("<숙박시설>", m.group(1))]),
        (r'([가-힣]{2,20}공원)', lambda m: [("<공원>", m.group(1))]),
        (r'([가-힣]{2,10}공고)', lambda m: [("<학교>", m.group(1))]),
        (rf'\b(?!안전?사고)([가-힣]{{2,10}}고)(?={JOSA})', lambda m: [("<학교>", m.group(1))]),
        (rf'\b((?!{EXCLUDE_FOR_STATION})[가-힣0-9]{{2,20}}역)(?=\s|' + JOSA + r'|\b)',
         lambda m: [("<역>", m.group(1))]),
        (EXCLUDE_PATTERN, lambda m: [("##EXCLUDE##", m.group(0))]),
    ]

    matches = []
    replaced_spans = []

    # ✅ 원본 text 기준으로 전체 매치 수집
    for pattern, slot_func in patterns:
        for match in re.finditer(pattern, text):
            start, end = match.span()
            if any(s < end and start < e for s, e in replaced_spans):
                continue
            for slot, value in slot_func(match):
                if slot != "##EXCLUDE##":
                    matches.append((start, end, slot, value))
                    replaced_spans.append((start, end))
                    slot_values[slot] = value  # ✅ 항상 마지막 slot_value만 저장됨
                    break

    # ✅ masked_text를 뒤에서 앞으로 치환하여 인덱스 보존
    masked_text = text
    for start, end, slot, _ in sorted(matches, reverse=True):
        masked_text = masked_text[:start] + slot + masked_text[end:]

    return masked_text, slot_values

def apply_masking_to_json_dir(json_dir: str, field="korean_text"):
    for file in tqdm(list(Path(json_dir).glob("*.json")), desc=f"📂 Masking in {json_dir}"):
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        if field in data:
            text = data[field].strip()
            masked_text, slot_values = grouped_region_mask_korean_text_final(text)
            data["masked_text"] = masked_text
            data["slot_values"] = slot_values

            with open(file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_dir')
    args = parser.parse_args()
    apply_masking_to_json_dir(args.json_dir)


if __name__ == '__main__':
    main()
