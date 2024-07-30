import numpy as np
import cv2
import mediapipe as mp
import math

def mediapipe_landmarks(image_path, output):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Load an image
    image = cv2.imread(image_path)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Landmarks of interest
    want_landmark = [5, 9, 13, 17]
    position_landmark = {}
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks detected by MediaPipe
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Store the positions of landmarks of interest
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                if idx in want_landmark:
                    position_landmark[idx] = (cx, cy)
                    cv2.circle(image, (cx, cy), 5, (0, 255,0), cv2.FILLED)  # points of interest

            # Calculate the midpoint between landmarks 9 and 13
            if 9 in position_landmark and 13 in position_landmark and 17 in position_landmark:
                x0, y0 = position_landmark[5]
                x1, y1 = position_landmark[9]
                x2, y2 = position_landmark[13]
                x3, y3 = position_landmark[17]
                mid_x0, mid_y0 = (x1 + x2) // 2, (y1 + y2) // 2
                mid_x1, mid_y1 = (x2 + x3) // 2, (y2 + y3) // 2
                pos5 = [x0, y0]
                pos9 = [x1, y1]
                mid_pos1 = [mid_x0, mid_y0]
                mid_pos2 = [mid_x1, mid_y1]
                cv2.circle(image, (mid_x0, mid_y0), 5, (255, 0, 0), cv2.FILLED)  # midpoint
                cv2.circle(image, (mid_x1, mid_y1), 5, (255, 0, 0), cv2.FILLED)  # midpoint
                
    positions = [pos5, pos9, mid_pos1, mid_pos2]
    hands.close()
    
    # Save the output image
    cv2.imwrite(output, image)
    
    return positions

def heart_line_peak_point(coloring_path, output_path = False, print_coloring = False):
    
    # Read img
    img = cv2.imread(coloring_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    
    # Assign points as white pixels
    points = np.column_stack(np.where(binary == 255))
    points = points[points[:, 0].argsort()]
    
    # Divided line in to parts
    total_length = len(points)
    # part_length = total_length - 1
    
    # Assign tail and tip parts
    tail_part = points[:1] # white
    tip_part = points[1:] # blue
    
    img = cv2.imread(coloring_path)
    img_colorized = img.copy()
    
    # Colorizing tail and tip parts
    for part, color in zip([tail_part, tip_part], [(255, 255, 255), (255, 0, 0)]): 
        for y, x in part:
            img_colorized[y, x] = color
            
    white_pixel = np.where(img_colorized == 255)

    if len(white_pixel[0]) > 0:
        y = white_pixel[0][0]
        x = white_pixel[1][0]
        pixel_position = [x, y]
    else:
        pixel_position = [0, 0]
        print("No white pixel found in the image.")
    
    # For display colorized img
    if print_coloring == True:
        if output_path == False:
            print("Need output_path")
        else:
            cv2.imwrite(output_path, img_colorized)
    
    return pixel_position

def find_closest_position_index(pixel_position, positions):
    def euclidean_distance(pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    closest_index = -1
    min_distance = float('inf')

    for i, pos in enumerate(positions):
        distance = euclidean_distance(pixel_position, pos)
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index

def heart_line_type6(coloring_path, compare_path, output_path = False, print_coloring = False):
    
    # Read img
    img = cv2.imread(coloring_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    
    # Assign points as white pixels
    points = np.column_stack(np.where(binary == 255))
    points = points[points[:, 0].argsort()]
    
    # Divided line in to parts
    total_length = len(points)
    
    # Assign line part
    line_part = points[:total_length] # green
    
    img = cv2.imread(coloring_path)
    img_colorized = img.copy()
    
    # Colorizing tail and tip parts
    for part, color in zip([line_part], [(0, 255, 0)]): 
        for y, x in part:
            img_colorized[y, x] = color

    # For display colorized img
    if print_coloring == True:
        if output_path == False:
            print("Need output_path")
        else:
            cv2.imwrite(output_path, img_colorized)

    # Assign the life line for comparison
    image1 = img_colorized
    image2 = cv2.imread(compare_path)

    green_lower = np.array([0, 200, 0])
    green_upper = np.array([50, 255, 50])

    white_lower = np.array([200, 200, 200])
    white_upper = np.array([255, 255, 255])

    # Find the points of intersect/crossing
    green_mask = cv2.inRange(image1, green_lower, green_upper)
    white_mask = cv2.inRange(image2, white_lower, white_upper)

    green_coords = np.column_stack(np.where(green_mask > 0))
    white_coords = np.column_stack(np.where(white_mask > 0))

    # case1 = False
    # case2 = False
    # case3 = True
    cross_count = 0

    # Find out how many white pixels in compare_path line have intersect with blue or green pixels of coloring_path line
    for white_pixel in white_coords:
        if any(np.linalg.norm(green_pixel - white_pixel) <= 1 for green_pixel in green_coords):
            # case2 = True
            # case3 = False
            cross_count += 1

    return cross_count

# coloring_path = '/home/brownien/Work_Dan/SeqNetHand_v2/report/group1/IMG_FEMALE_0385/step7_crop_excess/3_IMG_FEMALE_0385_epochs_100.png'
# compare_path = '/home/brownien/Work_Dan/SeqNetHand_v2/report/group1/IMG_FEMALE_0385/step7_crop_excess/0_IMG_FEMALE_0385_epochs_100.png'
# output_path = False

# count_case1, count_case2 = fate_line_type(coloring_path, compare_path, output_path, print_coloring = False)

hand_input = "/home/brownien/Work_Dan/SeqNetHand_v2/for_playground/heart/hand4case/IMG_FEMALE_0121.png"
hand_output = "/home/brownien/Work_Dan/SeqNetHand_v2/for_playground/heart/hand4case/IMG_FEMALE_0121_out.png"

heart_input = "/home/brownien/Work_Dan/SeqNetHand_v2/for_playground/heart/case5/Layer2.png"
heart_output = "/home/brownien/Work_Dan/SeqNetHand_v2/for_playground/heart/case5/Layer2_out.png"

color = "/home/brownien/Work_Dan/SeqNetHand_v2/for_playground/heart/case5/Layer3.png"
color_out = "/home/brownien/Work_Dan/SeqNetHand_v2/for_playground/heart/case5/Layer3_out.png"

positions = mediapipe_landmarks(hand_input, hand_output)
pixel_position = heart_line_peak_point(heart_input, heart_output, print_coloring = True)

cross_count = heart_line_type6(coloring_path = color, compare_path = heart_input, output_path = color_out, print_coloring = True)

# print(f'positions: {positions}')
# print(f'pixel_position: {pixel_position}')

closest_index = find_closest_position_index(pixel_position, positions)
# print(f'The closest position to {pixel_position} is at index {closest_index}')
# print(f'index position: {positions[closest_index]}')

if cross_count > 0:
    print(f'case 5: crossing points count: {cross_count}')
    print(f'เส้นหัวใจสัมผัสเส้นชีวิต: เป็นผู้อกหักง่าย')
else:
    print(f'เส้นหัวใจไม่สัมผัสเส้นชีวิต')
    
    if closest_index == 0:
        print(f'case 1: The closest position to {pixel_position} is {positions[closest_index]} (index: {closest_index})')
        print(f'เส้นหัวใจเริ่มที่ใต้นิ้วชี้: ผู้ที่มีความพึงพอใจในชีวิตรัก')
    if closest_index == 1:
        print(f'case 2: The closest position to {pixel_position} is {positions[closest_index]} (index: {closest_index})')
        print(f'เส้นหัวใจเริ่มที่ใต้นิ้วกลาง: ผู้ที่เห็นแก่ตัวในความรัก')
    if closest_index == 2:
        print(f'case 3: The closest position to {pixel_position} is {positions[closest_index]} (index: {closest_index})')
        print(f'เส้นหัวใจเริ่มที่ใต้กึ่งกลางนิ้วทั้งสี่: เป็นผู้ตกหลุมรักง่าย')
    if closest_index == 3:
        print(f'case 4: The closest position to {pixel_position} is {positions[closest_index]} (index: {closest_index})')
        print(f'เส้นหัวใจตรงและสั้น: เป็นผู้ไม่สนใจเรื่องความรัก')