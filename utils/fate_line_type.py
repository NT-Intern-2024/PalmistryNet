import numpy as np
import cv2

# NOTE: Function for analyze type of the fate line
def fate_line_type(coloring_path, compare_path, output_path = False, print_coloring = False):
    
    # Read img
    img = cv2.imread(coloring_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    
    # Assign points as white pixels
    points = np.column_stack(np.where(binary == 255))
    points = points[points[:, 0].argsort()]
    
    # Divided line in to parts
    total_length = len(points)
    part_length = total_length // 5
    
    # tip_part = points[:part_length] + points[4*part_length:]
    # middle_part = points[part_length:4*part_length]
    
    # Assign tail and tip parts
    tail_part = points[:4*part_length] # green
    tip_part = points[4*part_length:] # blue
    
    img = cv2.imread(coloring_path)
    img_colorized = img.copy()
    
    # Colorizing tail and tip parts
    for part, color in zip([tail_part, tip_part], [(0, 255, 0), (255, 0, 0)]): 
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

    blue_lower = np.array([200, 0, 0])
    blue_upper = np.array([255, 50, 50])

    green_lower = np.array([0, 200, 0])
    green_upper = np.array([50, 255, 50])

    white_lower = np.array([200, 200, 200])
    white_upper = np.array([255, 255, 255])

    # Find the points of intersect/crossing
    blue_mask = cv2.inRange(image1, blue_lower, blue_upper)
    green_mask = cv2.inRange(image1, green_lower, green_upper)
    white_mask = cv2.inRange(image2, white_lower, white_upper)

    blue_coords = np.column_stack(np.where(blue_mask > 0))
    green_coords = np.column_stack(np.where(green_mask > 0))
    white_coords = np.column_stack(np.where(white_mask > 0))

    # case1 = False
    # case2 = False
    # case3 = True
    count_case1 = 0
    count_case2 = 0

    # Find out how many white pixels in compare_path line have intersect with blue or green pixels of coloring_path line
    for white_pixel in white_coords:
        if any(np.linalg.norm(blue_pixel - white_pixel) <= 1 for blue_pixel in blue_coords):
            # case1 = True
            # case3 = False
            count_case1 += 1

        if any(np.linalg.norm(green_pixel - white_pixel) <= 1 for green_pixel in green_coords):
            # case2 = True
            # case3 = False
            count_case2 += 1

    return count_case1, count_case2

coloring_path = '/home/brownien/Work_Dan/SeqNetHand_v2/report/group1/IMG_FEMALE_0385/step7_crop_excess/3_IMG_FEMALE_0385_epochs_100.png'
compare_path = '/home/brownien/Work_Dan/SeqNetHand_v2/report/group1/IMG_FEMALE_0385/step7_crop_excess/0_IMG_FEMALE_0385_epochs_100.png'
output_path = False

count_case1, count_case2 = fate_line_type(coloring_path, compare_path, output_path, print_coloring = False)

# Classify type of fate line base on intersect points
if count_case1 >= 1:
    print(f'case 1 tip(blue) = {count_case1} mid(green) = {count_case2}')
    print(f'เส้นวาสนาเชื่อมติดปลายเส้นชีวิต: เป็นผู้สร้างเนื้อสร้างตัวได้ด้วยตนเอง')
if count_case2 > 5 and count_case1 < 1:
    print(f'case 2 tip(blue) = {count_case1} mid(green) = {count_case2}')
    print(f'เส้นวาสนาติดเส้นชีวิตบริเวณกลางเส้น: เป็นผู้ยอมทิ้งสิ่งที่ตนสนใจ เพื่อความสุขของผู้อื่น')
if 0 < count_case2 <= 5 and count_case1 < 1:
    print(f'case 3 tip(blue) = {count_case1} mid(green) = {count_case2}')
    print(f'เส้นวาสนาเริ่มจากฐานนิ้วหัวแม่มือ: ได้รับความสนใจจากครอบครัว และคนใกล้ชิด')
if count_case2 == 0 and count_case1 == 0:
    print(f'case 4 tip(blue) = {count_case1} mid(green) = {count_case2}')
    print(f'เส้นวาสนาอยู่กลางฝ่ามือ: พึ่งดวงอย่างเดียวไม่ค่อยได้ หลายครั้งต้องพึ่งตัวเองด้วย')
# else:
#     print(f'.')
    