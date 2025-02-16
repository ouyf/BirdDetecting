import os
import numpy as np
from vision_agent.tools import *
from typing import *
from pillow_heif import register_heif_opener
register_heif_opener()
import vision_agent as va
from vision_agent.tools import register_tool

def count_birds_in_image(image_path: str) -> int:
    """
    Count the number of birds in an image.

    Steps:
    1. Load the image using load_image.
    2. Subdivide the image into four overlapping sections.
    3. Perform detection on each section using the prompt 'bird, duck'.
    4. Merge bounding boxes from all sections to remove duplicates.
    5. Overlay the bounding boxes and save the resulting image.
    6. Return the total number of detected birds as the final solution.
    """

    import numpy as np

    # 1) Load the image
    image = load_image(image_path)
    height, width, _ = image.shape

    # Function to subdivide image
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

    # 2) Subdivide image
    subdivided_images = subdivide_image(image)

    # IoU helper for merging bounding boxes
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

    # Merge bounding boxes from all subdivided sections
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

    all_detections = []

    # 3) Perform detection on each subdivided section
    for idx, sub_img in enumerate(subdivided_images):
        detections = countgd_object_detection("bird, duck", sub_img)
        sub_h, sub_w, _ = sub_img.shape

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

        # Calculate offset to map subdivided coords back to original
        offset_x = (idx % 2) * (width // 2 - int(width // 2 * 0.1))
        offset_y = (idx // 2) * (height // 2 - int(height // 2 * 0.1))

        # Offset to original image space and normalize
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

    # 4) Merge bounding box results
    final_detections = merge_bounding_box_list(all_detections)

    # 5) Overlay bounding boxes and save the result
    annotated_image = overlay_bounding_boxes(image, final_detections)
    save_image(annotated_image, "detected_birds.jpg")

    # 6) Return the total number of detected birds
    return len(final_detections)
