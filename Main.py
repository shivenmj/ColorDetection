import cv2
import numpy as np
from scipy.spatial import KDTree
import pyttsx3
from Utilpt2 import get_limits, get_color_bgr_values, get_color_objects, get_color_makeup

# Global variables
clicked_color = None
clicked_position = (-1, -1)
color_box_size = 150
show_objects = False
show_speaker = False
average_area_size = 15

# Extended list of colors and names
colors = get_color_bgr_values()

# Convert predefined colors to Lab space
lab_colors = {name: cv2.cvtColor(np.uint8([[value]]), cv2.COLOR_BGR2Lab)[0][0] for name, value in colors.items()}

# Create KDTree for fast nearest-neighbor lookup
lab_values = np.array(list(lab_colors.values()))
kdtree = KDTree(lab_values)

# Example objects associated with colors
objects = get_color_objects()

# Initialize text-to-speech engine
engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def get_color_name(bgr_color):
    clicked_lab = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2Lab)[0][0]
    _, index = kdtree.query(clicked_lab)
    color_name = list(lab_colors.keys())[index]
    return color_name


def get_color(event, _x, _y, flags, param):
    global clicked_color, clicked_position, show_objects, show_speaker
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= _x <= 80 and 90 <= _y <= 110:
            show_objects = not show_objects
            print("Objects text box clicked. Toggling show_objects to", show_objects)
        elif frame.shape[1] - 100 <= _x <= frame.shape[1] - 20 and frame.shape[0] - 50 <= _y <= frame.shape[0] - 30:
            show_speaker = not show_speaker
            print("Speaker text box clicked. Toggling show_speaker to", show_speaker)
            if show_speaker and clicked_color is not None:
                speak_color_info()
        else:
            clicked_position = (_x, _y)
            clicked_color = average_color(frame, clicked_position, average_area_size)
            show_objects = False
            show_speaker = False
            print("Color selected at position:", clicked_position, "Average color:", clicked_color)


def speak_color_info():
    if clicked_color is not None:
        color_name = get_color_name(clicked_color)
        color_makeup = get_color_makeup().get(color_name, [])
        darkness_level = calculate_darkness(clicked_color)

        text = f"The selected color is {color_name}. "
        if color_makeup:
            percentages = ', '.join(f'{makeup}' for makeup in color_makeup)
            text += f"It consists of {percentages}. "
        text += f"The darkness level is {darkness_level:.1f} out of 10."

        if color_name in objects:
            object_names = ', '.join(objects[color_name])
            text += f" It is commonly associated with {object_names}."

        speak(text)


def average_color(image, position, size):
    x, y = position
    height, width, _ = image.shape
    x_start = max(x - size, 0)
    x_end = min(x + size, width)
    y_start = max(y - size, 0)
    y_end = min(y + size, height)
    area = image[y_start:y_end, x_start:x_end]

    # Apply Gaussian blur to the selected area
    blurred_area = cv2.GaussianBlur(area, (5, 5), 0)

    average_color = cv2.mean(blurred_area)[:3]
    return np.array(average_color, dtype=np.uint8)


def increase_saturation(frame, saturation_scale=1.2):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation_scale
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_text_with_outline(image, text, position, font, scale, color, thickness, outline_color=(0, 0, 0),
                           outline_thickness=2):
    # Draw the outline
    cv2.putText(image, text, position, font, scale, outline_color, outline_thickness, lineType=cv2.LINE_AA)
    # Draw the text
    cv2.putText(image, text, position, font, scale, color, thickness, lineType=cv2.LINE_AA)


def draw_centered_text(image, text, font, scale, color, thickness, outline_color=(0, 0, 0), outline_thickness=2,
                       max_width=None):
    if max_width is not None:
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        while text_size[0] > max_width and scale > 0.1:
            scale -= 0.1
            text_size = cv2.getTextSize(text, font, scale, thickness)[0]

    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + 10
    draw_text_with_outline(image, text, (text_x, text_y), font, scale, color, thickness, outline_color,
                           outline_thickness)


def calculate_darkness(bgr_color):
    gray_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2GRAY)[0][0]
    return (255 - gray_color) / 255 * 10


cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', get_color)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Increase the saturation of the frame
        frame = increase_saturation(frame, saturation_scale=1.2)

        if clicked_color is not None:
            # Perform operations based on clicked color
            box_x = max(clicked_position[0] - 5, 0)
            box_y = max(clicked_position[1] - 5, 0)
            box_width = min(10, frame.shape[1] - box_x)
            box_height = min(10, frame.shape[0] - box_y)

            cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), 2)

            bgr_box = np.full((80, color_box_size, 3), clicked_color, dtype=np.uint8)
            color_name = get_color_name(clicked_color)
            draw_centered_text(bgr_box, color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                               max_width=color_box_size)

            color_makeup = get_color_makeup().get(color_name, [])
            y_offset = 60
            for makeup in color_makeup:
                text = f"{makeup}"
                draw_text_with_outline(bgr_box, text, (5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15

            frame[10:90, 10:10 + color_box_size] = bgr_box

            # Draw the "Objects" text box
            cv2.rectangle(frame, (10, 90), (80, 110), (255, 0, 0), -1)
            draw_text_with_outline(frame, "Objects", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Calculate and display darkness level
            darkness_level = calculate_darkness(clicked_color)
            darkness_text = "1 very light        10 very dark"
            line_start = (frame.shape[1] - 290, 20)
            line_end = (frame.shape[1] - 80, 20)
            cv2.line(frame, line_start, line_end, (255, 255, 255), 1)
            draw_text_with_outline(frame, darkness_text, (frame.shape[1] - 290, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                   (255, 255, 255), 1)
            cv2.circle(frame, (frame.shape[1] - 185, 50), 15, clicked_color.tolist(), -1)
            draw_text_with_outline(frame, f"{darkness_level:.1f}", (frame.shape[1] - 195, 55), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.4, (255, 255, 255), 1)

            # Draw the "Speaker" text box
            cv2.rectangle(frame, (frame.shape[1] - 100, frame.shape[0] - 50),
                          (frame.shape[1] - 20, frame.shape[0] - 30), (0, 0, 255), -1)
            draw_text_with_outline(frame, "Speaker", (frame.shape[1] - 95, frame.shape[0] - 35),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw the objects box if show_objects is True
        if show_objects and clicked_color is not None:
            objects_box = np.zeros((80, color_box_size, 3), dtype=np.uint8)
            color_name = get_color_name(clicked_color)
            if color_name in objects:
                object_names = objects[color_name]
                y_offset = 20
                for obj_name in object_names:
                    draw_text_with_outline(objects_box, obj_name, (5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                           (255, 255, 255), 1)
                    y_offset += 20
            frame[10:90, 200:200 + color_box_size] = objects_box

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
