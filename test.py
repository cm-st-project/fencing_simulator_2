from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame

# Initialize Pygame
pygame.init()

# Set up the display
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Fencing Simulator")

# Set up the clock for frame rate
clock = pygame.time.Clock()

video_path = 'fencing.mp4'
filename = Path(video_path).stem
position = filename.split('_')[-1]
# Load images
head_image = pygame.image.load(f'images/head-left.png')
head_image = pygame.transform.scale(head_image, (50, 50))  # Resize to desired dimensions

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Open video file
cap = cv2.VideoCapture(0)


def extract_landmarks(results):
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y])
    return np.array(landmarks)


def draw_painted_background(screen):
    # Fill the background with a light color to simulate a painted canvas
    screen.fill((230, 230, 230))  # Light gray background

    # Draw the fencing strip (a rectangle to represent the practice area)
    pygame.draw.rect(screen, (139, 69, 19), (50, 150, 700, 300))  # Brown filled rectangle
    pygame.draw.rect(screen, (255, 255, 255), (50, 150, 700, 300), 2)  # White border of the strip

    # Draw fencing lines (simulate the lines on the fencing strip)
    num_lines = 8
    line_spacing = 300 / (num_lines - 1)
    for i in range(num_lines):
        pygame.draw.line(screen, (255, 255, 255), (50, 150 + i * line_spacing), (750, 150 + i * line_spacing), 2)


def draw_pose_lines(screen, landmarks):
    if landmarks.size > 0:
        # Convert normalized coordinates to pixel coordinates
        landmarks_px = [(int(x * screen_width), int(y * screen_height)) for x, y in landmarks]

        # Define connections between landmarks
        connections = [
            (11, 12),  # Right shoulder to left shoulder
            (11, 13),  # Right shoulder to right elbow
            (13, 21),  # Right elbow to right wrist
            (12, 14),  # Left shoulder to left elbow
            (14, 16),  # Left elbow to left wrist
            (11, 23),  # Right shoulder to right hip
            (12, 24),  # Left shoulder to left hip
            (23, 25),  # Right hip to right knee
            (25, 27),  # Right knee to right ankle
            (24, 26),  # Left hip to left knee
            (26, 28),  # Left knee to left ankle
            (0, 12),  # Head to left shoulder
            (0, 11),  # Head to right shoulder
            (23, 24),  # Right hip to left hip
        ]

        # Draw lines between landmarks with white color and increased thickness
        for start_idx, end_idx in connections:
            start_pos = landmarks_px[start_idx]
            end_pos = landmarks_px[end_idx]
            pygame.draw.line(screen, (255, 255, 255), start_pos, end_pos, 6)  # White color and thickness of 6

            # Draw black circle at the wrist joint
            if start_idx == 14 or end_idx == 14:  # Right wrist
                pygame.draw.circle(screen, (255, 0, 0), start_pos, 7)  # Black circle
                pygame.draw.circle(screen, (0, 0, 0), end_pos, 7)
            else:
                pygame.draw.circle(screen, (255, 0, 0), start_pos, 7)  # Red circle at other joints
                pygame.draw.circle(screen, (255, 0, 0), end_pos, 7)

        # Fill the torso area with white
        right_shoulder = landmarks_px[11]
        left_shoulder = landmarks_px[12]
        right_hip = landmarks_px[23]
        left_hip = landmarks_px[24]
        torso_points = [right_shoulder, left_shoulder, left_hip, right_hip]
        pygame.draw.polygon(screen, (255, 255, 255), torso_points)

        # Draw head image
        head_x, head_y = landmarks_px[0]  # Head landmark
        head_rect = head_image.get_rect(center=(head_x, head_y))
        screen.blit(head_image, head_rect.topleft)

        # Draw the saber
        if len(landmarks_px) > 14:  # Ensure wrist landmarks are available
            right_wrist = np.array(landmarks_px[16])
            right_elbow = np.array(landmarks_px[14])
            right_shoulder = np.array(landmarks_px[12])

            # Calculate the angle of the saber based on the arm position
            delta_y = right_elbow[1] - right_shoulder[1]
            delta_x = right_elbow[0] - right_shoulder[0]
            angle = np.arctan2(delta_y, delta_x)  # Angle in radians

            # Draw a line extending from the right wrist to simulate the saber
            saber_length = 100
            saber_end_x = right_wrist[0] + saber_length * np.cos(angle)
            saber_end_y = right_wrist[1] - saber_length * np.sin(angle)
            saber_end = (int(saber_end_x), int(saber_end_y))
            pygame.draw.line(screen, (0, 0, 0), right_wrist, saber_end, 4)  # Black color and thickness of 4


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture and process frame for pose estimation
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    landmarks = extract_landmarks(results)

    # Draw the painted background
    draw_painted_background(screen)

    # Draw the pose lines, head image, and saber
    draw_pose_lines(screen, landmarks)

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(30)

pygame.quit()
cap.release()
cv2.destroyAllWindows()
