import cv2
import numpy as np
from pathlib import Path

def is_inverted(img):
    """
    Using gradient from border to the mid to detect, wether the image iss inverted
    we could import this function also from the other class
    """
    h, w = img.shape
    
    # Boundaries
    by, bx = max(1, int(h * 0.05)), max(1, int(w * 0.05))
    top = img[0:by, :]
    bottom = img[h-by:h, :]
    left = img[:, 0:bx]
    right = img[:, w-bx:w]
    
    # Center
    cy, cx = h // 2, w // 2
    dy, dx = int(h * 0.2), int(w * 0.2)
    center = img[cy-dy:cy+dy, cx-dx:cx+dx]
    
    edge_mean = np.mean([np.mean(top), np.mean(bottom), np.mean(left), np.mean(right)])
    center_mean = np.mean(center)
    
    return edge_mean > center_mean

def find_joint_space_y(img_gray):
    """
    Find y coordinate of joint space using vertical sobel of the image 
    maximizing sum over rows
    """
    h, w = img_gray.shape
    
    # gaussian (smooth highfrequent edges)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Sobel (dy=1)
    s_v = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
    
    # constrain search area horizontal
    x_start, x_end = int(w * 0.10), int(w * 0.90)
    center_roi = s_v[:, x_start:x_end]
    
    # blur edges / compute row mean (eigentlich muss man das nicht normalisieren)
    roi_blurred = cv2.GaussianBlur(center_roi, (5, 5), 0)
    row_means = np.mean(roi_blurred, axis=1)
    
    # Sconstrain search area vertical
    # search_top = int(h * 0.25)
    # search_bottom = int(h * 0.75)

    # Changed:
    search_top = int(h * 0.40)
    search_bottom = int(h * 0.82)
    
    # find max
    local_max_index = np.argmax(row_means[search_top:search_bottom])
    line_y = search_top + local_max_index
    
    return line_y

def suppress_metal_artifacts(img, threshold=235):
    """
    Suppress very bright metal-like pixels before ROI extraction.
    """
    metal_mask = img > threshold

    if not np.any(metal_mask):
        return img

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    metal_mask = cv2.dilate(
        metal_mask.astype(np.uint8),
        kernel,
        iterations=2,
    ).astype(bool)

    img_no_metal = img.copy()

    non_metal_pixels = img[~metal_mask]
    replacement_value = (
        np.median(non_metal_pixels)
        if non_metal_pixels.size > 0
        else np.median(img)
    )

    img_no_metal[metal_mask] = replacement_value

    return img_no_metal

def get_roi_mask(image_input):
    """
    Image as path or image
    
    """
    # 1. Eingabe verarbeiten (Pfad vs. Array)
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Bild konnte nicht geladen werden: {image_input}")
    elif isinstance(image_input, np.ndarray):
        # Falls das Bild farbig übergeben wird, in Graustufen umwandeln
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input.copy()
    else:
        raise TypeError("Die Eingabe muss ein Dateipfad (str/Path) oder ein Numpy-Array sein.")

    h, w = img.shape

    # Inversion if neccessary
    if is_inverted(img):
        img = cv2.bitwise_not(img)

    # # Find join space y
    # line_y = find_joint_space_y(img)

    # # Pipeline Preprocess 
    # processed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # processed = cv2.equalizeHist(processed)
    # _, mask = cv2.threshold(processed, 85, 255, cv2.THRESH_BINARY)

    #Changed:
    # Suppress very bright metal/implant-like pixels before ROI generation.
    img_for_roi = suppress_metal_artifacts(img)

    # Find joint-space line after metal suppression.
    line_y = find_joint_space_y(img_for_roi)

    # Constrain ROI search to a band around the detected joint-space line.
    y_top = max(0, int(line_y - 0.22 * h))
    y_bottom = min(h, int(line_y + 0.22 * h))
    x_left = int(0.10 * w)
    x_right = int(0.90 * w)

    search_window = np.zeros_like(img_for_roi, dtype=np.uint8)
    search_window[y_top:y_bottom, x_left:x_right] = 1

    # Pipeline preprocessing.
    processed = cv2.normalize(
        img_for_roi,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    processed = cv2.equalizeHist(processed)
    _, mask = cv2.threshold(processed, 85, 255, cv2.THRESH_BINARY)

    # Restrict thresholded mask to the joint-space search window.
    mask = (mask * search_window).astype(np.uint8)

    #end change

    # morph operations
    r = int(h * 0.04)
    d = 2 * r + 1 
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if contours:
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     temp_mask = np.zeros_like(mask)
    #     cv2.drawContours(temp_mask, [largest_contour], -1, 255, -1)
    #     mask = temp_mask
    # else:
    #     mask = np.zeros_like(mask)

    #changed:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 0.01 * h * w:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        contour_center_y = y + ch / 2

        if y_top <= contour_center_y <= y_bottom:
            valid_contours.append(contour)

    if valid_contours:
        largest_contour = max(valid_contours, key=cv2.contourArea)
        temp_mask = np.zeros_like(mask)
        cv2.drawContours(temp_mask, [largest_contour], -1, 255, -1)
        mask = temp_mask
    else:
        # Safer fallback: use the joint-space search window instead of selecting bolts.
        mask = (search_window * 255).astype(np.uint8)

    kw = max(1, int(w * 0.10))
    kh = max(1, int(h * 0.20))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # dynamical crop
    y_top = max(0, int(line_y - 0.30 * h))
    y_bottom = min(h, int(line_y + 0.20 * h))
    mask[:y_top, :] = 0
    mask[y_bottom:, :] = 0

    # MASK FORMAT HEEEEERE
    binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    
    return binary_mask