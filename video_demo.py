"""Recreate the notebook's MediaPipe video-to-text demonstration."""

import argparse

import cv2
import mediapipe as mp
import numpy as np
import sentencepiece as spm
import torch

from model import Decoder, Encoder, Seq2Seq


POSE_KEEP_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18]


def extract_video_frames(video_path, fps=30):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps // fps) if original_fps > fps else 1
    frames = []
    count = 0
    while cap.isOpened():
        available, frame = cap.read()
        if not available:
            break
        if count % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count += 1
    cap.release()
    return frames


def extract_filtered_keypoints(frames):
    pose = mp.solutions.pose.Pose(static_image_mode=False)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
    results = []
    for frame in frames:
        coordinates = []
        pose_result = pose.process(frame)
        if pose_result.pose_landmarks:
            for index in POSE_KEEP_IDX:
                point = pose_result.pose_landmarks.landmark[index]
                coordinates.extend([point.x, point.y])
        else:
            coordinates.extend([0.0, 0.0] * len(POSE_KEEP_IDX))

        # Keep left and right hands in the same 108-coordinate order as training.
        hand_result = hands.process(frame)
        left_hand = [0.0, 0.0] * 21
        right_hand = [0.0, 0.0] * 21
        if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
            for index, landmarks in enumerate(hand_result.multi_hand_landmarks):
                label = hand_result.multi_handedness[index].classification[0].label
                points = [value for point in landmarks.landmark
                          for value in (point.x, point.y)]
                if label == 'Left':
                    left_hand = points
                elif label == 'Right':
                    right_hand = points
        coordinates.extend(left_hand)
        coordinates.extend(right_hand)
        results.append(coordinates)
    pose.close()
    hands.close()
    return np.asarray(results, dtype=np.float32)


def normalize_mediapipe_keypoints(keypoints, stat_path, image_width=1280,
                                   image_height=720):
    """Use the OpenPose training statistics, as in the notebook demonstration."""
    keypoints = np.copy(keypoints)
    keypoints[:, 0::2] *= image_width
    keypoints[:, 1::2] *= image_height
    stats = np.load(stat_path)
    pose = (keypoints[:, :24] - stats['pose_mu']) / (stats['pose_sd'] + 1e-8)
    left = ((keypoints[:, 24:66] - stats['left_min']) /
            (stats['left_max'] - stats['left_min'] + 1e-8) - 0.5)
    right = ((keypoints[:, 66:] - stats['right_min']) /
             (stats['right_max'] - stats['right_min'] + 1e-8) - 0.5)
    return np.concatenate([pose, left, right], axis=-1)


def predict_from_keypoints(model, keypoints, tokenizer, device, max_len=100):
    """Greedy decoding used for the notebook's video demonstration."""
    model.eval()
    x = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)
    decoder_input = torch.tensor([[tokenizer.bos_id()]], device=device)
    predicted = []
    with torch.no_grad():
        for _ in range(max_len):
            output = model(x, decoder_input, teacher_forcing_ratio=0.0)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            if next_token == tokenizer.eos_id():
                break
            predicted.append(next_token)
            decoder_input = torch.cat([decoder_input,
                                       torch.tensor([[next_token]], device=device)], dim=1)
    return tokenizer.decode(predicted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--normalization', required=True, help='normalization.npz from normalize_keypoints.py')
    parser.add_argument('--tokenizer', required=True, help='spm.model from prepare_arrays.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--image-width', type=int, default=1280)
    parser.add_argument('--image-height', type=int, default=720)
    args = parser.parse_args()

    frames = extract_video_frames(args.video, args.fps)
    if not frames:
        raise ValueError(f'No frames could be read from {args.video}')
    keypoints = extract_filtered_keypoints(frames)
    keypoints = normalize_mediapipe_keypoints(keypoints, args.normalization,
                                              args.image_width, args.image_height)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(Encoder(), Decoder(output_dim=tokenizer.vocab_size()), device).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(predict_from_keypoints(model, keypoints, tokenizer, device))


if __name__ == '__main__':
    main()
