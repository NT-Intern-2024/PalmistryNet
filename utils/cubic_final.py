import os
import cv2
import numpy as np
from scipy.optimize import curve_fit

def cubic_function(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def format_cubic_equation(coefficients):
    a, b, c, d = coefficients
    equation = f'y = {a:.4f}x^3 + {b:.4f}x^2 + {c:.4f}x + {d:.4f}'
    return equation

def perform_cubic_regression(image_paths, output_folder, color_thick=1, poly_thick=1, save_coef=False):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Define filename to color mapping
    filename_to_color = {
        '0': 'yellow',
        '1': 'red',
        '2': 'green',
        '3': 'blue'
    }

    for image_path in image_paths:
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Get the first character of the filename
        first_char = filename[0]

        # Get color based on the first character
        color = filename_to_color.get(first_char, 'cyan')  # Default to cyan if filename doesn't match

        # Read the image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        result_image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to RGB for color drawing

        if image is None:
            print(f"Error: Unable to read image '{image_path}'")
            continue

        # Perform adaptive thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area to ensure we get the largest four
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

        # Convert color name to BGR tuple
        color_dict = {
            'yellow': (0, 255, 255),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'cyan': (255, 255, 0)  # Default color if not matched
        }
        line_color = color_dict.get(color, (255, 255, 0))  # Default to cyan if color not found

        # Iterate over each contour (line) found
        for i, contour in enumerate(contours):
            # Convert contour points to x, y values
            x_values = contour[:, 0, 0]  # x coordinates
            y_values = contour[:, 0, 1]  # y coordinates

            height, width = image.shape
            max_image_size = max(height, width)

            # Perform cubic regression with switched axes
            popt, _ = curve_fit(cubic_function, y_values, x_values)

            # Print the coefficients of the cubic regression function
            if save_coef == True:           
                print(f"{image_path} - Coefficients (a, b, c, d):", popt)

            # Format the cubic regression equation
            equation = format_cubic_equation(popt)
            
            if save_coef == True:
                print(f"{image_path} - Fitted Cubic Equation:", equation)

            # Generate points for the fitted cubic curve with switched axes
            fit_y = np.linspace(0, max_image_size, 100)
            fit_x = cubic_function(fit_y, *popt)

            # Draw the original data points as a line on the result image
            for j in range(len(x_values) - 1):
                pt1 = (int(x_values[j]), int(y_values[j]))
                pt2 = (int(x_values[j + 1]), int(y_values[j + 1]))
                cv2.line(result_image_rgb, pt1, pt2, line_color, color_thick) # NOTE for edit original line 

            # Draw the fitted cubic curve on the result image with switched axes
            for j in range(len(fit_x) - 1):
                pt1 = (int(fit_x[j]), int(fit_y[j]))
                pt2 = (int(fit_x[j + 1]), int(fit_y[j + 1]))
                cv2.line(result_image_rgb, pt1, pt2, (255, 255, 255), poly_thick) # NOTE for edit regression line 

            # Save the result image with the cubic fit
            output_image_path = os.path.join(output_folder, f'{filename}.png')
            cv2.imwrite(output_image_path, result_image_rgb)

            # print(f"Saved result image with cubic fit for Line {i+1} of {image_path}: {output_image_path}")

def get_image_paths_from_folder(folder_path):
    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)
    # Filter the list to include only PNG files
    image_paths = [os.path.join(folder_path, f) for f in all_files if f.endswith('.png')]
    return image_paths

"""
input_folder_path = './Predict_line_by_third_v2/data'  # Replace with the path to your folder containing the images
output_folder_path = './Predict_line_by_third_v2/output_cubic_v4'  # Replace with the path to your output folder

# input_folder_path = 'data'  # Replace with the path to your folder containing the images
# output_folder_path = 'output_palm'  # Replace with the path to your output folder

image_paths = get_image_paths_from_folder(input_folder_path)
perform_cubic_regression(image_paths, output_folder_path)
"""