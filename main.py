import cv2
import numpy as np
from image_processing import *

# Parameters
TARGET_IMAGE_SIZE = 540
OBJECT_UPP_BOUND_SCREEN_SIZE = 0.8
OBJECT_LOW_BOUND_SIZE = 5000
DEFAULT_COLOR_RANGE_1 = (71, 104, 0)
DEFAULT_COLOR_RANGE_2 = (130, 255, 255)


# Macros
def load_image(image_directory,image_file_name):
    return cv2.imread(f'{image_directory}/{image_file_name}')


def preproces_image(img):
    # Scale image
    img = scale(img, TARGET_IMAGE_SIZE)
    # Add border
    img = addborder(img, 100)
    # Denoise image
    img = denoise(img, 15)
    return img

def process_mask(img, img_mask):
    # Gray image
    img_gray = gray(img_mask)

    # Threshold
    threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contour
    contours = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    high_bound = (img.shape[0] * img.shape[1]) * OBJECT_UPP_BOUND_SCREEN_SIZE
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        cv2.fillPoly(threshold, [cnt], color=(0, 0, 0))
        if area > OBJECT_LOW_BOUND_SIZE and area < high_bound:
            filtered_contours.append(cnt)

    # Fill contour
    for cnt in filtered_contours:
        cv2.fillPoly(threshold, [cnt], color=(255, 255, 255))

    # Mask original image with processed threshold mask
    img_final = cv2.bitwise_and(img, img, mask=threshold)

    # Highlight object
    num_object = 0
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_final, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_final, f'Road sign {num_object + 1}', (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        num_object += 1
    return img_final

def enable_debug():
    cv2.namedWindow("HSV Lower Boundary Slider")
    cv2.namedWindow("HSV Upper Boundary Slider")
    cv2.createTrackbar("Lower Hue", "HSV Lower Boundary Slider", DEFAULT_COLOR_RANGE_1[0], 255, lambda x: x)
    cv2.createTrackbar("Lower Saturation", "HSV Lower Boundary Slider", DEFAULT_COLOR_RANGE_1[1], 255, lambda x:x)
    cv2.createTrackbar("Lower Value", "HSV Lower Boundary Slider", DEFAULT_COLOR_RANGE_1[2], 255, lambda x:x)
    cv2.createTrackbar("Upper Hue", "HSV Upper Boundary Slider", DEFAULT_COLOR_RANGE_2[0], 255, lambda x:x)
    cv2.createTrackbar("Upper Saturation", "HSV Upper Boundary Slider", DEFAULT_COLOR_RANGE_2[1], 255, lambda x:x)
    cv2.createTrackbar("Upper Value", "HSV Upper Boundary Slider", DEFAULT_COLOR_RANGE_2[2], 255, lambda x:x)

def run_debug(image_file_name,img):
    while (True):
        lower_hue = cv2.getTrackbarPos("Lower Hue", "HSV Lower Boundary Slider")
        lower_saturation = cv2.getTrackbarPos("Lower Saturation", "HSV Lower Boundary Slider")
        lower_value = cv2.getTrackbarPos("Lower Value", "HSV Lower Boundary Slider")
        upper_hue = cv2.getTrackbarPos("Upper Hue", "HSV Upper Boundary Slider")
        upper_saturation = cv2.getTrackbarPos("Upper Saturation", "HSV Upper Boundary Slider")
        upper_value = cv2.getTrackbarPos("Upper Value", "HSV Upper Boundary Slider")
        lower = np.array([lower_hue, lower_saturation, lower_value])
        upper = np.array([upper_hue, upper_saturation, upper_value])

        # Create and apply mask
        img_masked = create_apply_mask(img, lower, upper)
        # Process the image with mask
        img_final = process_mask(img, img_masked)

        cv2.putText(img_final, f'Debug mode', (0, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.putText(img_final, f'Press Esc Twice to Exit...', (0, 35), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv2.imshow(image_file_name, img_final)

        # for button pressing and changing
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break


def create_apply_mask(img,lower, upper):
    # Get HSV value of the image
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Morphological Open
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Morphological Close
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
    # Masking the image
    img_masked = cv2.bitwise_and(img, img, mask=mask_close)
    return img_masked



def main(image_directory, image_file_name, is_debug_mode):
    # Load image
    img = load_image(image_directory,image_file_name)
    # Preprocess image
    img = preproces_image(img)

    if (is_debug_mode):
        enable_debug()
        run_debug(image_file_name,img)
        return

    # Create and apply mask
    mask = create_apply_mask(img, DEFAULT_COLOR_RANGE_1, DEFAULT_COLOR_RANGE_2)
    # Process the image with mask
    img_final = process_mask(img, mask)

    cv2.imshow(image_file_name, img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
