"""Merge source XML keypoints with JSON text annotations."""
import argparse
import json
import os
from glob import glob
from pathlib import Path
import xmltodict

POSE_KEEP_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]

def filter_pose_keypoints(points):
    keypoints = [list(map(float, p.split(','))) for p in points]
    return [coord for idx in POSE_KEEP_IDX if idx < len(keypoints) for coord in keypoints[idx][:2]]


def filter_hand_keypoints(points):
    keypoints = [list(map(float, p.split(','))) for p in points]
    return [coord for kpt in keypoints for coord in kpt[:2]]


def merge_single_sample_from_path(xml_path, json_path, save_dir):
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_data = xmltodict.parse(f.read())
    except Exception:
        print(f"[오류] XML 파싱 실패: {xml_path}")
        return False

    frames_dict = {}
    for track in xml_data["annotations"]["track"]:
        label = track["@label"]
        if label == "face_keypoints_2d":
            continue
        for key in track:
            if key not in ["@id", "@label"]:
                entries = track[key] if isinstance(track[key], list) else [track[key]]
                for entry in entries:
                    idx_f = int(entry["@frame"])
                    points = [p for p in entry["@points"].split(';') if p.strip()]
                    filtered_points = (
                        filter_pose_keypoints(points) if label == "pose_keypoints_2d"
                        else filter_hand_keypoints(points)
                    )
                    frames_dict.setdefault(idx_f, {})[label] = filtered_points

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        korean_text = js.get("korean_text", "")
        signer_id = js.get("translator", {}).get("id", None)
    except Exception:
        print(f"[오류] JSON 파싱 실패: {json_path}")
        return False

    base = os.path.splitext(os.path.basename(xml_path))[0]
    result = {
        "index": base,
        "id": signer_id,
        "korean_text": korean_text,
        "frames": []
    }

    prev_frame_vec = None
    for idx_f in sorted(frames_dict):
        frame_info = frames_dict[idx_f]
        frame_vec = frame_info.get("pose_keypoints_2d", []) + \
                    frame_info.get("hand_left_keypoints_2d", []) + \
                    frame_info.get("hand_right_keypoints_2d", [])
        if prev_frame_vec is not None and frame_vec == prev_frame_vec:
            continue
        prev_frame_vec = frame_vec
        result["frames"].append({
            "frame_idx": idx_f,
            "pose": frame_info.get("pose_keypoints_2d", []),
            "hand_left": frame_info.get("hand_left_keypoints_2d", []),
            "hand_right": frame_info.get("hand_right_keypoints_2d", [])
        })

    save_path = os.path.join(save_dir, f"{base}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml-root', required=True)
    parser.add_argument('--json-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    success = missing = 0
    for xml_path in sorted(glob(os.path.join(args.xml_root, '**', '*.xml'), recursive=True)):
        relative = os.path.relpath(os.path.dirname(xml_path), args.xml_root)
        json_name = os.path.basename(xml_path).replace('_F.xml', '.json')
        json_path = os.path.join(args.json_root, relative, json_name)
        if os.path.isfile(json_path):
            success += bool(merge_single_sample_from_path(xml_path, json_path, args.output_dir))
        else:
            missing += 1
    print(f'Merged {success} samples; {missing} annotation files missing.')


if __name__ == '__main__':
    main()
