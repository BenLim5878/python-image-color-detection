import cv2
import numpy as np
from image_processing import *

# Parameters
TARGET_IMAGE_SIZE = 540
OBJECT_UPP_BOUND_SCREEN_SIZE = 0.8
OBJECT_LOW_BOUND_SIZE = 800
SQUARE_WIDTH_HEIGHT_THRESHOLD = 0.15
CIRCLE_WIDTH_HEIGHT_THRESHOLD = 0.15
CIRCLE_RATIO_MATCH_THRESHOLD = 0.15

# Macros
def load_image(image_directory,image_file_name):
    return cv2.imread(f'{image_directory}/{image_file_name}')


def preproces_image(img):
    # Scale image
    img = scale(img, TARGET_IMAGE_SIZE)
    # Denoise image
    img = denoise(img, 15)
    return img

def find_num_object(img):
    # Gray image
    img_gray = gray(img)
    # Threshold
    threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Find contour
    contours = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    high_bound = (img.shape[0] * img.shape[1]) * OBJECT_UPP_BOUND_SCREEN_SIZE
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > OBJECT_LOW_BOUND_SIZE and area < high_bound:
            filtered_contours.append(cnt)
    # Count number of oject
    num_object = 0
    for cnt in filtered_contours:
        num_object += 1
    return num_object

def process(image_directory, image_file_name, target_color_range_1, target_color_range_2):
    # Load image
    img = load_image(image_directory,image_file_name)
    # Preprocess image
    img = preproces_image(img)
    # Get HSV value of the image
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, target_color_range_1, target_color_range_2)
    # Masking the image
    img_masked = cv2.bitwise_and(img, img, mask = mask)
    # Find number of object
    total_object = find_num_object(img_masked)
    cv2.imshow(image_file_name, img_masked)
    print(f'\n\n\n------ {image_file_name} stats ------')
    print(f'The number of blue objects in the scene is : {total_object}')
    print(f'------ end of {image_file_name} stats ------')
    cv2.waitKey(0)