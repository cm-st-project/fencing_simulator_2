# FencingDynamics

**FencingDynamics** is a state-of-the-art simulator designed to help individuals practice and refine their fencing skills. Built using Python, Pygame, and MediaPipe, this application allows users to visualize and compare their fencing movements in real-time against pre-recorded sequences, providing a dynamic training experience.

## Features

- **Real-Time Pose Estimation**: Analyze your fencing movements in real-time using MediaPipe’s advanced pose detection.
- **Pre-Recorded Movement Comparison**: Compare your real-time movements against pre-recorded fencing sequences to improve technique.
- **Visual Feedback**: The simulator provides clear visual cues, drawing pose lines, landmarks, and other elements to help you assess your form.
- **User-Friendly Interface**: The Pygame-powered graphical interface is intuitive, making it easy to navigate through the simulation and practice sessions.

## Usage

1. **Run the Application**:
   ```bash
   python main.py
   ```

2. **Navigate the Interface**:
   - Upon starting, the main menu provides options to start a new simulation or exit the application.
   - During a simulation, your real-time fencing movements are displayed alongside pre-recorded movements for comparison.
   - Use the back button to return to the main menu from any simulation screen.

## Project Structure

- **main.py**: The main entry point of the application, responsible for initializing the graphical interface and integrating real-time pose estimation.
- **video_processor.py**: A utility module that processes video input, extracts pose landmarks, and prepares data for visualization.
- **landmarks.json**: A JSON file containing pre-recorded fencing movements that are used for comparison during simulations.
- **logo.png**: The logo of the application.

## Detailed File Descriptions

### `main.py`

The `main.py` file serves as the primary entry point for the FencingDynamics application. It is responsible for initializing the application, handling the user interface, and integrating various functionalities such as pose estimation, drawing landmarks, and managing real-time interactions. Below is a breakdown of its key components:

1. **Imports and Setup**:
   - The file imports necessary libraries including `pygame` for the graphical interface, `mediapipe` for pose estimation, and `cv2` for video processing.
   - It sets up the Pygame display environment, specifying the screen dimensions and initializing the main screen.

2. **MediaPipe Pose Initialization**:
   - The script initializes MediaPipe's pose detection model, which is later used to capture and process real-time pose data.

3. **Loading and Normalizing Landmarks**:
   - The file reads pre-recorded landmarks from a JSON file (`landmarks.json`). These landmarks represent key points on the human body, such as shoulders, hips, and ankles.
   - The script includes functions to calculate the height of the person based on the landmarks and normalize these landmarks to fit within the screen.

4. **Drawing Functions**:
   - `draw_pose_lines()`: This function draws lines connecting the detected landmarks to visualize the human pose. It takes into account the normalized landmarks and applies offsets to position the drawings correctly on the screen.
   - `draw_painted_background()`: A utility function to render a background that visually separates different parts of the simulation.

5. **Main Simulation Loop**:
   - The `main()` function contains the main loop where the simulation runs. It captures real-time video frames, processes them using MediaPipe to extract landmarks, and then renders these on the Pygame display.
   - The loop also handles the display of pre-recorded landmarks, allowing users to compare their real-time movements against a standard.

6. **Back Button and User Navigation**:
   - The file includes a back button that allows users to return to the main menu from the simulation screen. This button is drawn on the screen and is responsive to user clicks.

7. **Main Menu**:
   - The `main_menu()` function sets up the main menu interface, where users can start the simulation or exit the application. It’s the first screen users see when they run the application.

8. **Event Handling**:
   - The script manages various Pygame events such as mouse clicks and keyboard inputs. This allows for smooth interaction within the application, including navigation between screens and controlling the simulation.


### `video_processor.py`

The `video_processor.py` file is a utility module designed to handle video processing and pose extraction tasks. It works in conjunction with `main.py` to manage video input, process it using MediaPipe, and extract relevant data for use in the simulation. Here's a detailed breakdown of its functions:

1. **Imports and Setup**:
   - The file imports essential libraries such as `cv2` for video capture and processing, `mediapipe` for pose detection, and `json` for handling data storage.

2. **extract_landmarks()**:
   - This function is central to the video processing pipeline. It takes the results from MediaPipe's pose detection and extracts the landmarks (key points) from each frame.
   - The landmarks include coordinates (x, y, z) and visibility scores, which are then compiled into a list. This list is used in `main.py` to visualize the pose.

3. **process_video()**:
   - This function processes an entire video file, frame by frame. It opens the video using OpenCV, processes each frame through MediaPipe to detect the pose, and extracts landmarks.
   - The function can save the extracted landmarks into a JSON file for later use, allowing the application to load and compare movements during the simulation.

4. **Frame Handling and Storage**:
   - The script efficiently handles video frames, ensuring that each frame is correctly processed and that the landmarks are stored in a structured format.
   - It also includes mechanisms to handle different video formats and resolutions, ensuring compatibility with various input files.

5. **Error Handling**:
   - The file includes basic error handling to manage cases where pose landmarks might not be detected in a frame. This prevents the application from crashing and ensures smooth processing.

6. **Integration with main.py**:
   - The extracted landmarks from `video_processor.py` are passed back to `main.py` for visualization. This integration allows for real-time and pre-recorded data to be displayed side by side during the simulation.

`video_processor.py` acts as the backend processor, handling the heavy lifting of video analysis and pose extraction. By separating this functionality from `main.py`, the project maintains a clean and modular structure, making it easier to maintain and extend.


### `landmarks.json`

While not explicitly mentioned before, the `landmarks.json` file plays a crucial role in storing the pre-recorded landmarks data. Here's a brief explanation:

- **Purpose**:
  - The `landmarks.json` file contains a structured representation of pose landmarks that were extracted from a pre-recorded video. This data is used within the simulation to compare against real-time movements.

- **Data Structure**:
  - The file stores landmarks in a JSON format, with each frame’s landmarks represented as an array of objects. Each object contains x, y, z coordinates and a visibility score for a specific landmark (e.g., shoulder, hip, etc.).

- **Usage**:
  - During the simulation, the data from `landmarks.json` is loaded into memory and used to draw the pre-recorded pose lines on the screen. This allows users to visually compare their movements with a standard.


## Acknowledgments

- **MediaPipe**: For providing robust tools for real-time pose estimation.
- **Pygame**: For enabling the creation of an intuitive graphical interface.
- **OpenCV**: For handling video processing and integration.

