import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import copy

# Initialize color points storage for different colors
color_points = {
    "blue": [deque(maxlen=1024)],
    "green": [deque(maxlen=1024)],
    "red": [deque(maxlen=1024)],
    "yellow": [deque(maxlen=1024)],
}

# Track the current index for each color deque
color_indexes = {"blue": 0, "green": 0, "red": 0, "yellow": 0}

# Kernel for image processing (not used but kept for potential enhancements)
kernel = np.ones((5, 5), np.uint8)

# Define available colors and initialize the selected color
available_colors = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255)}
selected_color = "blue"

# Create a blank canvas
canvas_height, canvas_width = 480, 640
canvas = np.full((canvas_height, canvas_width, 3), (200, 200, 200), dtype=np.uint8)

def overlay_button(image, button, position):
    """Overlay a button with transparency onto the image."""
    if button.shape[2] == 4:  # Check if button has an alpha channel
        bgr_button = button[:, :, :3]  # Extract RGB channels
        Transparency_Level = 0.7
        alpha_channel = (button[:, :, 3] / 255.0) * Transparency_Level  # Normalize alpha values

        x, y = position
        h, w = bgr_button.shape[:2]

        # Blend the button with the region of interest (ROI)
        roi = image[y:y + h, x:x + w]
        for c in range(3):  # Blend each color channel
            roi[:, :, c] = (alpha_channel * bgr_button[:, :, c] + (1 - alpha_channel) * roi[:, :, c])

        # Place the blended ROI back
        image[y:y + h, x:x + w] = roi
    else:
        # If no alpha channel, directly overlay the button
        image[position[1]:position[1] + button.shape[0], position[0]:position[0] + button.shape[1]] = button

def redraw_canvas():
    """Clear and redraw the canvas with all points."""
    global canvas
    canvas[82:, :, :] = (200, 200, 200)  # Clear the drawing area
    for color, points in color_points.items():
        for point_group in points:
            for i in range(1, len(point_group)):
                if point_group[i - 1] and point_group[i]:
                    cv2.line(canvas, point_group[i - 1], point_group[i], available_colors[color], 2)

def draw_buttons(window):
    """Draw UI buttons on the given window."""
    button_positions = [
        ((20, 1), "CLEAR", clear_img),
        ((105, 1), "BLUE", blue_img),
        ((190, 1), "GREEN", green_img),
        ((275, 1), "RED", red_img),
        ((360, 1), "YELLOW", yellow_img),
        ((445, 1), "UNDO", undo_img),
        ((490, 1), "REDO", redo_img),
    ]
    for pos, label, button in button_positions:
        overlay_button(window, button, pos)

# Import and resize buttons
clear_img = cv2.imread('button_clear.png', cv2.IMREAD_UNCHANGED)
blue_img = cv2.imread('button_blue.png', cv2.IMREAD_UNCHANGED)
green_img = cv2.imread('button_green.png', cv2.IMREAD_UNCHANGED)
red_img = cv2.imread('button_red.png', cv2.IMREAD_UNCHANGED)
yellow_img = cv2.imread('button_yellow.png', cv2.IMREAD_UNCHANGED)
undo_img = cv2.imread('button_undo.png', cv2.IMREAD_UNCHANGED)
redo_img = cv2.imread('button_redo.png', cv2.IMREAD_UNCHANGED)

clear_img = cv2.resize(clear_img, (80, 80))
blue_img = cv2.resize(blue_img, (80, 80))
green_img = cv2.resize(green_img, (80, 80))
red_img = cv2.resize(red_img, (80, 80))
yellow_img = cv2.resize(yellow_img, (80, 80))
undo_img = cv2.resize(undo_img, (40, 40))
redo_img = cv2.resize(redo_img, (40, 40))

draw_buttons(canvas)

# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Setup webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, canvas_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, canvas_height)

# Undo and redo stacks
undo_stack, redo_stack = [], []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw UI buttons on the frame
    draw_buttons(frame)

    # Detect hands and landmarks
    results = hands_detector.process(frame_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append((int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])))

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Forefinger and thumb tips
        forefinger_tip = landmarks[8]
        thumb_tip = landmarks[4]
        cv2.circle(frame, forefinger_tip, 5, (0, 255, 0), -1)

        if thumb_tip[1] - forefinger_tip[1] < 30:  # Fingers close together
            for color in color_points.keys():
                color_points[color].append(deque(maxlen=1024))
                color_indexes[color] += 1

        elif forefinger_tip[1] <= 65:  # Check if hovering over control buttons
            if 20 <= forefinger_tip[0] <= 100:  # Clear button
                undo_stack.append((copy.deepcopy(color_points), copy.deepcopy(color_indexes)))
                redo_stack.clear()
                canvas[82:, :, :] = (200, 200, 200)
                for color in color_points.keys():
                    color_points[color] = [deque(maxlen=1024)]
                    color_indexes[color] = 0

            elif 105 <= forefinger_tip[0] <= 185:
                selected_color = "blue"
            elif 190 <= forefinger_tip[0] <= 270:
                selected_color = "green"
            elif 275 <= forefinger_tip[0] <= 355:
                selected_color = "red"
            elif 360 <= forefinger_tip[0] <= 440:
                selected_color = "yellow"

            elif 445 <= forefinger_tip[0] <= 485:  # Undo button
                if undo_stack:
                    redo_stack.append((copy.deepcopy(color_points), copy.deepcopy(color_indexes)))
                    color_points, color_indexes = undo_stack.pop()
                    redraw_canvas()

            elif 490 <= forefinger_tip[0] <= 535:  # Redo button
                if redo_stack:
                    undo_stack.append((copy.deepcopy(color_points), copy.deepcopy(color_indexes)))
                    color_points, color_indexes = redo_stack.pop()
                    redraw_canvas()

        else:  # Drawing on the canvas
            if forefinger_tip[1] > 82:  # Avoid control area
                undo_stack.append((copy.deepcopy(color_points), color_indexes.copy()))
                redo_stack.clear()
                if selected_color not in color_points:
                    color_points[selected_color] = [deque(maxlen=1024)]
                    color_indexes[selected_color] = 0
                if color_indexes[selected_color] >= len(color_points[selected_color]):
                    color_points[selected_color].append(deque(maxlen=1024))
                color_points[selected_color][color_indexes[selected_color]].appendleft(forefinger_tip)
                redraw_canvas()

    else:
        # Reset when no hand detected
        for color in color_points.keys():
            color_points[color].append(deque(maxlen=1024))
        color_indexes = {color: 0 for color in color_points}

    # Draw points on the frame and canvas
    for color, points in color_points.items():
        for point_group in points:
            for i in range(1, len(point_group)):
                if point_group[i - 1] and point_group[i]:
                    cv2.line(frame, point_group[i - 1], point_group[i], available_colors[color], 2)
                    cv2.line(canvas, point_group[i - 1], point_group[i], available_colors[color], 2)

    cv2.imshow("Virtual Paint", frame)
    cv2.imshow("Canvas", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key
        break

video_capture.release()
cv2.destroyAllWindows()