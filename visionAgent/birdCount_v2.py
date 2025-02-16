import os
import numpy as np
from vision_agent.tools import *
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

import os
import numpy as np
from typing import *
from pillow_heif import register_heif_opener
from vision_agent.tools import *

def count_birds_in_image(image_path: str) -> int:
    """
    Count the number of birds in an image, excluding detections with confidence less than 50%.
    
    Parameters:
        image_path (str): Path to the input image file
        
    Returns:
        int: Number of birds detected with confidence >= 50%
        
    Notes:
        - The function subdivides the image into four overlapping sections for better detection
        - Saves the annotated image as 'detected_birds.jpg' in the current directory
        - Filters out detections with confidence < 50%
    """
    # Load the image
    image = load_image(image_path)
    height, width, _ = image.shape
    
    def subdivide_image(img: np.ndarray):
        h, w, _ = img.shape
        mid_h = h // 2
        mid_w = w // 2
        overlap_h = int(mid_h * 0.1)
        overlap_w = int(mid_w * 0.1)
        top_left = img[:mid_h + overlap_h, :mid_w + overlap_w, :]
        top_right = img[:mid_h + overlap_h, mid_w - overlap_w:, :]
        bottom_left = img[mid_h - overlap_h:, :mid_w + overlap_w, :]
        bottom_right = img[mid_h - overlap_h:, mid_w - overlap_w:, :]
        return [top_left, top_right, bottom_left, bottom_right]

    def bounding_box_match(b1, b2, iou_threshold=0.1):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        if x2 < x1 or y2 < y1:
            return False
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        return iou >= iou_threshold

    def merge_bounding_box_list(detections):
        merged_detections = []
        for det in detections:
            matched_index = None
            for i, existing_det in enumerate(merged_detections):
                if bounding_box_match(det["bbox"], existing_det["bbox"]):
                    matched_index = i
                    break
            if matched_index is not None:
                if det["score"] > merged_detections[matched_index]["score"]:
                    merged_detections[matched_index] = det
            else:
                merged_detections.append(det)
        return merged_detections

    # Process each subdivided section
    subdivided_images = subdivide_image(image)
    all_detections = []

    for idx, sub_img in enumerate(subdivided_images):
        detections = owlv2_object_detection("bird, duck", sub_img)
        sub_h, sub_w, _ = sub_img.shape

        # Filter out detections with confidence < 50%
        detections = [d for d in detections if d["score"] >= 0.5]

        # Convert normalized coords to pixel coords
        unnormalized = []
        for d in detections:
            unnormalized.append({
                "label": d["label"],
                "score": d["score"],
                "bbox": [
                    d["bbox"][0] * sub_w,
                    d["bbox"][1] * sub_h,
                    d["bbox"][2] * sub_w,
                    d["bbox"][3] * sub_h
                ]
            })

        # Calculate offset for mapping back to original image
        offset_x = (idx % 2) * (width // 2 - int(width // 2 * 0.1))
        offset_y = (idx // 2) * (height // 2 - int(height // 2 * 0.1))

        # Map coordinates back to original image space
        for det in unnormalized:
            all_detections.append({
                "label": det["label"],
                "score": det["score"],
                "bbox": [
                    (det["bbox"][0] + offset_x) / width,
                    (det["bbox"][1] + offset_y) / height,
                    (det["bbox"][2] + offset_x) / width,
                    (det["bbox"][3] + offset_y) / height
                ]
            })

    # Merge overlapping detections
    final_detections = merge_bounding_box_list(all_detections)
    
    # Save annotated image
    annotated_image = overlay_bounding_boxes(image, final_detections)
    save_image(annotated_image, "detected_birds.jpg")

    return len(final_detections)
