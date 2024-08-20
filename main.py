import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pygame

from video_processor import extract_landmarks

# Initialize Pygame
pygame.init()

# Set up the display
screen_width = 1200
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("FencingDynamics")

# Set up the clock for frame rate
clock = pygame.time.Clock()
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load JSON file with landmarks
with open('landmarks.json', 'r') as f:
    landmarks_data = json.load(f)


# Function to calculate the height of the person based on specific landmarks
def calculate_person_height(landmarks_px):
    shoulder_to_hip_distance = np.linalg.norm(
        np.array(landmarks_px[11]) - np.array(landmarks_px[23]))  # Right shoulder to right hip
    head_to_foot_distance = np.linalg.norm(
        np.array(landmarks_px[0]) - np.array(landmarks_px[27]))  # Head to right ankle
    return max(shoulder_to_hip_distance, head_to_foot_distance)


# Function to normalize landmarks to a default height and apply an offset
def normalize_landmarks(landmarks_px, default_height, start_x, start_y):
    person_height = calculate_person_height(landmarks_px)
    scale_factor = default_height / person_height

    return [(int(x * scale_factor) + start_x, int(y * scale_factor) + start_y) for x, y in landmarks_px]


# Function to draw pose lines
def draw_pose_lines(screen, landmarks, head_image, start_x=0, start_y=0):
    if landmarks:
        # Convert normalized coordinates to pixel coordinates
        landmarks_px = [(int(landmark['x'] * screen_width), int(landmark['y'] * screen_height)) for landmark in
                        landmarks]

        # Normalize the landmarks to a default height (e.g., 300 pixels) and apply the starting position offset
        default_height = 200
        landmarks_px = normalize_landmarks(landmarks_px, default_height, start_x, start_y)

        drawBody(landmarks_px, screen)
        drawHead(head_image, landmarks_px, screen)
        drawSaber(landmarks_px, screen)


def drawSaber(landmarks_px, screen):
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


def drawHead(head_image, landmarks_px, screen):
    # Draw head image
    head_x, head_y = landmarks_px[0]  # Head landmark
    head_rect = head_image.get_rect(center=(head_x, head_y))
    screen.blit(head_image, head_rect.topleft)


def drawBody(landmarks_px, screen):
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


# Function to draw the painted background
def draw_painted_background(screen):
    # Fill the background with a light color to simulate a painted canvas
    screen.fill((230, 230, 230))  # Light gray background

    # Draw the fencing strip (a rectangle to represent the practice area)
    pygame.draw.rect(screen, (139, 69, 19), (50, 150, screen_width - 100, 300))  # Brown filled rectangle
    pygame.draw.rect(screen, (255, 255, 255), (50, 150, screen_width - 100, 300), 2)  # White border of the strip

    # Draw fencing lines (simulate the lines on the fencing strip)
    num_lines = 8
    line_spacing = 300 / (num_lines - 1)
    for i in range(num_lines):
        pygame.draw.line(screen, (255, 255, 255), (50, 150 + i * line_spacing),
                         (screen_width - 50, 150 + i * line_spacing), 2)


# Main loop
def main_menu():
    running = True
    font = pygame.font.Font(None, 36)
    container_height = 200
    container_width = screen_width - 100
    container_y = 300
    scrollbar_width = 20
    scroll_y = 0

    while running:
        screen.fill((255, 255, 255))  # Clear the screen with black
        display_logo_name_main(font)
        display_selected_label_main(font)
        container_rect = draw_video_container(container_height, container_width, container_y)
        video_buttons = create_video_options(container_height, container_y, font, scroll_y)
        max_scroll_y = draw_scroll_bar_main(container_height, container_y, scroll_y, scrollbar_width, video_buttons)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if container_rect.collidepoint(mouse_pos):
                    for text_rect, video_name in video_buttons:
                        if text_rect.collidepoint(mouse_pos):
                            pose_estimation_screen(video_name)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:  # Scroll up
                scroll_y = max(0, scroll_y - 20)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:  # Scroll down
                scroll_y = min(max(0, max_scroll_y), scroll_y + 20)

        pygame.display.flip()
        clock.tick(30)


def display_logo_name_main(font):
    logo_image = pygame.image.load('logo.png')  # Load your app logo
    logo_image = pygame.transform.scale(logo_image, (150, 150))  # Resize to desired dimensions
    # Display the app name and logo
    app_name_surface = font.render('FencingDynamics', True, (0, 0, 0))
    app_name_rect = app_name_surface.get_rect(center=(screen_width // 2, 50))
    screen.blit(app_name_surface, app_name_rect.topleft)
    logo_rect = logo_image.get_rect(center=(screen_width // 2, 150))
    screen.blit(logo_image, logo_rect.topleft)


def draw_video_container(container_height, container_width, container_y):
    # Draw the container with a scrollbar
    container_rect = pygame.Rect(50, container_y, container_width, container_height)
    pygame.draw.rect(screen, (50, 50, 50), container_rect)  # Container background
    return container_rect


def display_selected_label_main(font):
    # Display the select label
    select_label = font.render('Select a Drill', True, (0, 0, 0))
    select_label_rect = select_label.get_rect(center=(screen_width // 5, 280))
    screen.blit(select_label, select_label_rect.topleft)


def create_video_options(container_height, container_y, font, scroll_y):
    # Create a list of buttons
    video_buttons = []
    y_offset = -scroll_y
    for video_name in landmarks_data.keys():
        text_surface = font.render(video_name, True, (255, 255, 255))
        text_rect = text_surface.get_rect(topleft=(60, y_offset + container_y))
        if text_rect.bottom > container_y and text_rect.top < container_y + container_height:
            screen.blit(text_surface, text_rect.topleft)
        video_buttons.append((text_rect, video_name))
        y_offset += 50
    return video_buttons


def draw_scroll_bar_main(container_height, container_y, scroll_y, scrollbar_width, video_buttons):
    # Draw scrollbar
    content_height = len(video_buttons) * 50  # Total height of content
    max_scroll_y = max(0, content_height - container_height)
    scrollbar_height = container_height * (container_height / max(1, content_height))
    scrollbar_x = screen_width - 50
    scrollbar_y = container_y + (scroll_y / max(1, max_scroll_y)) * (container_height - scrollbar_height)
    pygame.draw.rect(screen, (150, 150, 150), (scrollbar_x, scrollbar_y, scrollbar_width, scrollbar_height))
    return max_scroll_y


def pose_estimation_screen(video_name):
    landmarks = landmarks_data[video_name]
    position = Path(video_name).stem.split('_')[-1]
    # Load head image based on the position
    head_image = pygame.image.load(f'head-{position}.png')
    head_image = pygame.transform.scale(head_image, (50, 50))  # Resize to desired dimensions
    opposite_position = 'left' if position == 'right' else 'right'
    # Load the real-time head image
    realtime_head_image = pygame.image.load(f'head-{opposite_position}.png')
    # Resize the real-time head image to match the pre-recorded head image size
    realtime_head_image = pygame.transform.scale(realtime_head_image, head_image.get_size())

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Open the default camera

    back_button_color, back_button_rect, back_button_text = define_back_button_simulation_screen()

    frame_index = 0
    num_frames = len(landmarks)

    # Define size for the smaller video frame
    video_frame_width = screen_width // 5
    video_frame_height = screen_height // 5

    while frame_index < num_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                cap.release()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if back_button_rect.collidepoint(pygame.mouse.get_pos()):
                    cap.release()
                    return

        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Get the frame dimensions
        frame_resized = cv2.resize(frame, (video_frame_width, video_frame_height))

        # Convert resized frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))

        # Process the frame for real-time pose estimation
        results = pose.process(frame)
        realtime_landmarks = extract_landmarks(results)

        # Draw the painted background
        draw_painted_background(screen)



        # Display the video frame
        screen.blit(frame_surface, (screen_width - video_frame_width - 10, 10))  # Top-right corner

        if frame_index < num_frames:
            frame_landmarks = landmarks[frame_index]
            # Draw the pre-recorded pose lines, head image, and saber on the left side
            draw_pose_lines(screen, frame_landmarks, head_image, start_x=300, start_y=0)
            frame_index += 1
            # Draw the real-time pose lines, head image, and saber
            draw_pose_lines(screen, realtime_landmarks, realtime_head_image, start_x=100, start_y=100)

        # Draw the back button
        pygame.draw.rect(screen, back_button_color, back_button_rect)
        screen.blit(back_button_text, (back_button_rect.x + 10, back_button_rect.y + 5))

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(30)


def define_back_button_simulation_screen():
    # Define the back button
    back_button_rect = pygame.Rect(100, 10, 80, 30)
    back_button_color = (0, 0, 0)
    back_button_text = pygame.font.Font(None, 24).render(' Back', True, (255, 255, 255))
    return back_button_color, back_button_rect, back_button_text


if __name__ == "__main__":
    main_menu()
    pygame.quit()
