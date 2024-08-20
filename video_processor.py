import glob
import json
from pathlib import Path

import cv2
import mediapipe as mp


def extract_landmarks(results):
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
    return landmarks


def process_video(video_path):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    video_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        landmarks = extract_landmarks(results)
        video_landmarks.append(landmarks)

    cap.release()
    return video_landmarks


def save_landmarks_to_json(videos, output_file):
    data = {}
    for video_path in videos:
        print(f"Processing video: {video_path}")
        video_landmarks = process_video(video_path)
        data[Path(video_path).stem] = video_landmarks

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    # List of videos to process
    videos = glob.glob('fencing_videos/*.mp4')

    # Output JSON file
    output_file = 'landmarks.json'

    # Process videos and save landmarks to JSON
    save_landmarks_to_json(videos, output_file)

    print("Landmarks saved to JSON file successfully.")
