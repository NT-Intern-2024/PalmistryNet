import tensorflow as tf
from skimage import io, img_as_ubyte
import os,cv2,sys
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from PIL import Image
import mediapipe as mp

# NOTE: Add custom function in this file

WARP_SUCCESS = 1

def save_model(model, directory, filename='model.hdf5'):
    """
    Save the model to an HDF5 file in the specified directory.
    
    Parameters:
    model (tf.keras.Model): The Keras model to be saved.
    directory (str): The directory where the model file will be saved.
    filename (str): The name of the file where the model will be saved.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, filename)
    model.save(filepath)
    print(f"Model saved to {filepath}")

def data_generator_pred(data, batch_size=1):              
    """
    Yields the next training batch.
    data is an array  [[[frame1_filename,frame2_filename,…frame16_filename],label1], [[frame1_filename,frame2_filename,…frame16_filename],label2],……….].
    """
    num_samples = data.shape[0]

    while True:   
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = data[offset:offset+batch_size]
            # label = labels[offset:offset+batch_size]
            # Initialise X_train and y_train arrays for this batch
            X_train = []
            # y_train = []
            # For each example
            for i in range(0,batch_samples.shape[0]):
                X_train.append(batch_samples[i])
                # y_train.append(label[i])

            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            # X_train = np.rollaxis(X_train,1,4)
            # y_train = np.array(y_train)

            # yield the next training batch            
            yield X_train#, y_train
            
def data_generator(X, batch_size):
    while True:
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            yield X[start:end]

def tensor_slide(X, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(batch_size)

# NOTE: function for skelitonize the prediction lines
# def convert_to_skel(image_path, output_path, file_name):
def convert_to_skel(image_path):
    # Read the grayscale image
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if grayscale_image is None:
        raise ValueError("Image not loaded. Please check the image path.")
    
    # Apply thresholding to convert to black and white
    _, black_and_white_image = cv2.threshold(grayscale_image, 130, 255, cv2.THRESH_BINARY)
    
    binary_image = black_and_white_image // 255
    skeleton = skeletonize(binary_image)
    skel_img = skeleton.astype(np.uint8) * 255
    
    # cv2.imwrite(output_path  + "/" + file_name, skel_img)
    
    return skel_img

# NOTE: function for coloring the skel lines
# def colorize_lines(image_path, output_path, file_name, index):
def colorize_lines(image_path, index):
    # Define the colors for different indices
    colors = {
        0: (0, 255, 255),  # Yellow
        1: (0, 0, 255),    # Red
        2: (0, 255, 0),    # Green
        3: (255, 0, 0)     # Blue
    }
    
    # Read the grayscale image
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if image is loaded properly
    if grayscale_image is None:
        raise ValueError("Image not loaded. Please check the image path.")
    
    # Get the color for the given index
    color = colors.get(index, (255, 255, 255))  # Default to white if index is not in the dictionary
    
    # Create an empty color image
    color_image = np.zeros((grayscale_image.shape[0], grayscale_image.shape[1], 3), dtype=np.uint8)
    
    # Colorize the lines
    color_image[grayscale_image == 255] = color  # Apply the color to the white lines (255)
    
    # Save the output image
    # cv2.imwrite(output_path + "/" + f"i{index}_" + file_name, color_image)
    
    return color_image

# NOTE: function for merging colorize line
# def merge_images(image_paths, output_path, file_name):
def merge_images(image_paths):
    # Read the first image to get the size
    base_image = cv2.imread(image_paths[0])
    if base_image is None:
        raise ValueError("Image not loaded. Please check the image path: " + image_paths[0])
    
    # Create an empty image with black background
    height, width, _ = base_image.shape
    final_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Define the order of layering (furthest to closest)
    order = [3, 2, 1, 0]

    for index in order:
        # Read the image for the current index
        image = cv2.imread(image_paths[index])
        if image is None:
            raise ValueError("Image not loaded. Please check the image path: " + image_paths[index])

        # Merge the images: keep everything that's not black
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0
        final_image[mask] = image[mask]

    # Save the final merged image
    # cv2.imwrite(output_path +  + "/" + f"final_" + file_name, final_image)
    
    return final_image

def merge_images4poly(image_paths, base_lay):
    # Read the first image to get the size
    base_image = cv2.imread(base_lay)
    if base_image is None:
        raise ValueError("Image not loaded. Please check the image path: " + base_lay)
    
    # Create an empty image with black background
    final_image = base_image.copy()

    # Define the order of layering (furthest to closest)
    order = [3, 2, 1, 0]

    for index in order:
        # Read the image for the current index
        image = cv2.imread(image_paths[index])
        if image is None:
            raise ValueError("Image not loaded. Please check the image path: " + image_paths[index])

        # Merge the images: keep everything that's not black
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0
        final_image[mask] = image[mask]

    # Save the final merged image
    # cv2.imwrite(output_path +  + "/" + f"final_" + file_name, final_image)
    
    return final_image

# NOTE: use for crop poly lines /w highlight
# def crop_excess_line(image_path, output_path):
def crop_excess_line(image_path, idx):
    # Read the image
    image = cv2.imread(image_path)
    
    # Define color ranges for color and white in RGB
    
    # yellow
    if idx == 0:
        color_lower = np.array([0, 200, 200])
        color_upper = np.array([100, 255, 255])
    
    # red
    elif idx == 1:
        color_lower = np.array([0, 0, 200])
        color_upper = np.array([100, 100, 255])
    
    # green
    elif idx == 2:        
        color_lower = np.array([0, 200, 0])
        color_upper = np.array([100, 255, 100])
    
    # blue
    elif idx == 3:        
        color_lower = np.array([200, 0, 0])
        color_upper = np.array([255, 100, 100])    
     
    white_lower = np.array([200, 200, 200])
    white_upper = np.array([255, 255, 255])
    
    # Create masks for color and white
    color_mask = cv2.inRange(image, color_lower, color_upper)
    white_mask = cv2.inRange(image, white_lower, white_upper)
    
    # Find contours of the color highlight
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask for the result
    result_mask = np.zeros_like(white_mask)
    
    # Draw the color highlight on the result mask
    cv2.drawContours(result_mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # Bitwise-AND the result mask with the white mask to crop out the white line outside the color highlight
    cropped_white_mask = cv2.bitwise_and(white_mask, result_mask)
    
    # Create an output image where only the cropped white line is visible
    output_image = cv2.bitwise_and(image, image, mask=cropped_white_mask)
    
    return output_image
    
    # Save the output image
    # cv2.imwrite(output_path, output_image)

"""
def warp_image_og(path_to_image, path_to_warped_image):
        # 7 landmark points (normalized)
    pts_index = list(range(21))
    pts_target_normalized = np.float32([[1-0.48203104734420776, 0.9063420295715332],
                                        [1-0.6043621301651001, 0.8119394183158875],
                                        [1-0.6763232946395874, 0.6790258884429932],
                                        [1-0.7340714335441589, 0.5716733932495117],
                                        [1-0.7896472215652466, 0.5098430514335632],
                                        [1-0.5655680298805237, 0.5117031931877136],
                                        [1-0.5979393720626831, 0.36575648188591003],
                                        [1-0.6135331392288208, 0.2713503837585449],
                                        [1-0.6196483373641968, 0.19251111149787903],
                                        [1-0.4928809702396393, 0.4982593059539795],
                                        [1-0.4899863600730896, 0.3213786780834198],
                                        [1-0.4894656836986542, 0.21283167600631714],
                                        [1-0.48334982991218567, 0.12900274991989136],
                                        [1-0.4258815348148346, 0.5180916786193848],
                                        [1-0.4033462107181549, 0.3581996262073517],
                                        [1-0.3938145041465759, 0.2616880536079407],
                                        [1-0.38608720898628235, 0.1775170862674713],
                                        [1-0.36368662118911743, 0.5642163157463074],
                                        [1-0.33553171157836914, 0.44737303256988525],
                                        [1-0.3209102153778076, 0.3749568462371826],
                                        [1-0.31213682889938354, 0.3026996850967407]])

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # 1. Extract 21 landmark points
        image = cv2.flip(cv2.imread(path_to_image), 1)
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # MyDebug: check image
        # cv2.imshow("Debug image", image)
        # cv2.waitKey()

        results = hands.process(image2)
        
        image_height, image_width, _ = image.shape
        if results.multi_hand_landmarks is None:
            print("Hand not detect")
            return None
        else:
            hand_landmarks = results.multi_hand_landmarks[0]
            # 2. Align images
            pts = np.float32([[hand_landmarks.landmark[i].x*image_width,
                            hand_landmarks.landmark[i].y*image_height] for i in pts_index])
            pts_target = np.float32([[x*image_width, y*image_height] for x,y in pts_target_normalized])
            M, mask = cv2.findHomography(pts, pts_target, cv2.RANSAC,5.0)
            warped_image = cv2.warpPerspective(image, M, (image_width, image_height), borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(path_to_warped_image, warped_image)
            return WARP_SUCCESS
"""

def warp_image(path_to_image, path_to_warped_image):
        # 7 landmark points (normalized)
    pts_index = list(range(21))
    pts_target_normalized = np.float32([[0.528295755, 0.861742795],
                                        [0.350052714, 0.764109135],
                                        [0.233854324, 0.641931593],
                                        [0.15225932, 0.548922539],
                                        [0.0873371661, 0.48465997],
                                        [0.409291625, 0.47470203],
                                        [0.351743162, 0.330576],
                                        [0.321447194, 0.238182724],
                                        [0.301581979, 0.16162923],
                                        [0.525921345, 0.464105815],
                                        [0.508178294, 0.296378523],
                                        [0.499403298, 0.18857044],
                                        [0.493700236, 0.103198588],
                                        [0.633148551, 0.491282433],
                                        [0.656118453, 0.33559373],
                                        [0.663122892, 0.238813519],
                                        [0.669961691, 0.156807095],
                                        [0.732504964, 0.549353838],
                                        [0.800100386, 0.450199068],
                                        [0.837219238, 0.383892775],
                                        [0.874538898, 0.32089889]])

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # 1. Extract 21 landmark points
        image = cv2.flip(cv2.imread(path_to_image), 1)
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # MyDebug: check image
        # cv2.imshow("Debug image", image)
        # cv2.waitKey()

        results = hands.process(image2)
        
        image_height, image_width, _ = image.shape
        if results.multi_hand_landmarks is None:
            print("Hand not detect")
            return None
        else:
            hand_landmarks = results.multi_hand_landmarks[0]
            # 2. Align images
            pts = np.float32([[hand_landmarks.landmark[i].x*image_width,
                            hand_landmarks.landmark[i].y*image_height] for i in pts_index])
            pts_target = np.float32([[x*image_width, y*image_height] for x,y in pts_target_normalized])
            M, mask = cv2.findHomography(pts, pts_target, cv2.RANSAC,5.0)
            warped_image = cv2.warpPerspective(image, M, (image_width, image_height), borderMode=cv2.BORDER_REPLICATE)
            cv2.imwrite(path_to_warped_image, warped_image)
            return WARP_SUCCESS

def warp(path_to_input_image, path_to_warped_image):
    if path_to_input_image[-4:] in ['heic', 'HEIC']:
        path_to_input_image = path_to_input_image[:-4] + 'png'
    warp_result = warp_image(path_to_input_image, path_to_warped_image)
    if warp_result is None:
        return None
    else:
        print("warp: else case")
        return WARP_SUCCESS

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def create_circular_mask(h, w, center, radius):
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = center
    cv2.circle(mask, (cx, cy), radius, (255, 255, 255), thickness=cv2.FILLED)
    return mask

def crop_fingers(image_path, output_path):
    want_landmark = [1, 0, 5, 6, 9, 10, 13, 14, 17, 18]
    # Initialize Mediapipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.8)
    mp_drawing = mp.solutions.drawing_utils

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return
    
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe
    results = hands.process(image_rgb)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for the palm
            palm_center = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]  # Center of the palm
            
            # Convert normalized coordinates to pixel coordinates
            palm_center_x = int(palm_center.x * image.shape[1])
            palm_center_y = int(palm_center.y * image.shape[1])  # Use image height for y
            
            # Calculate the radius of the circle based on maximum distance to red points
            max_distance = 0
            for idx in want_landmark:
                point = (int(hand_landmarks.landmark[idx].x * image.shape[1]), 
                         int(hand_landmarks.landmark[idx].y * image.shape[0]))
                dist = distance((palm_center_x, palm_center_y), point)
                if dist > max_distance:
                    max_distance = dist
            
            # Set a smaller radius value (e.g., half of max_distance)
            radius = int(max_distance * 0.75)
            
            # Create a circular mask for the hand region
            mask = create_circular_mask(image.shape[0], image.shape[1], (palm_center_y, palm_center_x), radius)
            
            # Remove fingers
            palm_only = np.bitwise_and(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

            print(f"hand_landmarks: {hand_landmarks}")
            
            # Save the palm-only image as PNG
            cv2.imwrite(output_path, palm_only)

            # print(f"Saved palm-only image: {output_path}")
    else:
        print("No hands detected.")

    # Release resources
    hands.close()

"""
# NOTE: MON ADD (not using at the moment)
def ske_color(image,layer,filename,epochs):
    #os.makedirs(os.path.join('output','templine_ske_color',filename), exist_ok=True)
    #output = os.path.join('output',filename,'templine_ske_color',f'{layer}_{filename}_epoch{epochs}.png')
    #os.path.join('output', filename,'templine_ske_color',filename)
    bgr_list = [
        (0,255, 255),  # yellow : layer0 วาสนา
        (255,0,0),  # Blue : layer1 สติปัญญา
        (0,0,255),  # Red : layer2 เส้นจิตใจ
        (255,0,255),  #magenta : layer3 เส้นชีวิต
    ]
    
    white = np.array([255, 255, 255])
    mask = np.all(image == white, axis=-1)
    
    if layer == 0:
    #print(layer,bgr_list[layer])
        image[mask] = bgr_list[layer]
        
        #all_color_ske_layer.append(os.path.join('output',filename,'templine_ske_color',f'{layer}_{filename}_epoch{epochs}.png'))
        cv2.imwrite(os.path.join('output',filename,'templine_ske_color',f'{layer}_{filename}_epochs_{epochs}.png'),image)
    elif layer == 1:
        image[mask] = bgr_list[layer]
        #all_color_ske_layer.append(os.path.join('output',filename,'templine_ske_color',f'{layer}_{filename}_epoch{epochs}.png'))
        cv2.imwrite(os.path.join('output',filename,'templine_ske_color',f'{layer}_{filename}_epochs_{epochs}.png'), image)
    elif layer == 2:
        image[mask] = bgr_list[layer]
        #all_color_ske_layer.append(os.path.join('output',filename,'templine_ske_color',f'{layer}_{filename}_epoch{epochs}.png'))
        cv2.imwrite(os.path.join('output',filename,'templine_ske_color',f'{layer}_{filename}_epochs_{epochs}.png'), image)
    elif layer == 3:
        
        image[mask] = bgr_list[layer]
        #all_color_ske_layer.append(os.path.join('output',filename,'templine_ske_color',f'{layer}_{filename}_epoch{epochs}.png'))
        cv2.imwrite(os.path.join('output',filename,'templine_ske_color',f'{layer}_{filename}_epochs_{epochs}.png'), image)
        all_color_ske_layer = []
        for i in range(4):
            all_color_ske_layer.append(os.path.join('output',filename,'templine_ske_color',f'{i}_{filename}_epochs_{epochs}.png'))
        if len(all_color_ske_layer) >=4:     
            return all_color_ske_layer

# NOTE: MON ADD (not using at the moment)
def line_before_skel(png_name, filename, epochs):
    os.makedirs(os.path.join('output', filename,'templine_ske_color'), exist_ok=True)
    
    input_path = os.path.join('output',filename,'templine_before_sk')
    
    for layer in png_name:
        # Construct the full path for the input image
        image_path = os.path.join(input_path, layer)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if the image was loaded properly
        if img is None:
            print(f"Error: Image '{image_path}' not loaded correctly")
            continue
        
        # Apply a binary threshold to detect the white line
        _, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask with the same dimensions as the image, initialized to black
        mask = np.zeros_like(img)

        # Draw the detected contours on the mask
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

        # Use the mask to keep only the white line in the image
        result = cv2.bitwise_and(binary, mask)

        # Convert result to binary (0 and 1) for skeletonization
        binary_result = (result // 255).astype(np.uint8)

        # Skeletonize the result
        skeleton = skeletonize(binary_result)  # Skeletonize expects binary image with 0 and 1

        # Convert the skeleton to 255 scale for visualization
        skeleton = (skeleton * 255).astype(np.uint8)
        
        
        # Convert the binary image to a 3-channel (RGB) image
        rgb_image_ske = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
        
        layer = int(layer[0])

        result_ske_color = ske_color(rgb_image_ske,layer,filename,epochs)
        
        # if isinstance(result_ske_color, list):
        #     if len(result_ske_color) >=4:
        #         print('before enter merge_ske_color func')
        #         merge_ske_color(result_ske_color,filename)
        
# NOTE: MON ADD (not using at the moment)
def merge_ske_color(result_ske_color,filename):
    os.makedirs(os.path.join('output',filename,'final_report_result'), exist_ok=True)
    print('enter  merge_ske_color')
    output = os.path.join('output',filename,'final_report_result')
    # Read the first image to get the size
    png_files = [file for file in os.listdir(f'output/{filename}/templine_ske_color') if file.endswith('.png')]

    if len(png_files) == 4:
        print("There are exactly 4 PNG files in the 'test' folder.")
        base_image = cv2.imread(result_ske_color[0])
        # if base_image is None:
        #     raise ValueError("Image not loaded. Please check the image path: " + result_ske_color[0])
        
        # Create an empty image with black background
        height, width, _ = base_image.shape
        final_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Define the order of layering (furthest to closest)
        order = [3, 2, 1, 0]

        for index in order:
            # Read the image for the current index
            image = cv2.imread(result_ske_color[index])
            # if image is None:
            #     raise ValueError("Image not loaded. Please check the image path: " + result_ske_color[index])

            # Merge the images: keep everything that's not black
            mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0
            final_image[mask] = image[mask]
        cv2.imwrite(f'{output}/result.png', final_image)
        print('final_image',final_image)
        
    else:
        print(f"There are {len(png_files)} PNG files in the 'test' folder.")
        return 

    
    #return final_image


# NOTE : THIRD ADD detect only palm not detect finger ################################################################################
# detect plam as circle 
def create_circular_mask(h, w, center, radius):
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = center
    cv2.circle(mask, (cx, cy), radius, (255, 255, 255), thickness=cv2.FILLED)
    return mask

def detect_palm_folder(folder_path, output_folder):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Initialize Mediapipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image '{image_path}'")
            continue
        
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe
        results = hands.process(image_rgb)

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks for the palm
                # Use landmarks with decimal values (Dacimal)
                palm_center = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]  # Center of the palm
                
                # Convert normalized coordinates to pixel coordinates
                palm_center_x = int(palm_center.x * image.shape[1])
                palm_center_y = int(palm_center.y * image.shape[1])  # Use image height for y
                
                # Calculate the radius of the circle 
                # use a fixed radius of 100 pixels
                radius = 100
                
                # Create a circular mask for the hand region
                mask = create_circular_mask(image.shape[0], image.shape[1], (palm_center_y, palm_center_x), radius)
                
                # remove fingers
                palm_only = np.bitwise_and(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

                # Save the palm-only image as PNG
                output_filename = os.path.splitext(image_file)[0] + '_palm_only.png'
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, palm_only)

                print(f"Saved palm-only image: {output_filename}")

    # Release resources
    hands.close()
 ##################################################################################################################################################
# if __name__ == "__main__":
#     # Specify the path to your image folder and output folder
#     folder_path = 'test'
#     output_folder = 'output_palm_only'

#     # Detect palm in images within the folder and save palm-only images as PNG
#     detect_palm_folder(folder_path, output_folder)
"""