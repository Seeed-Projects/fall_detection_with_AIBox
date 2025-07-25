import cv2
import numpy as np
from common.toolbox import id_to_color 
import paho.mqtt.client as mqtt
import time
import json

# MQTT configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "security/fall_alert"

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

def publish_fall_alert(box):
    alert = {
        "timestamp": int(time.time()),
        "event": "fall_down",
        "box": {
            "xmin": int(box[0]),
            "ymin": int(box[1]),
            "xmax": int(box[2]),
            "ymax": int(box[3])
        }
    }
    mqtt_client.publish(MQTT_TOPIC, json.dumps(alert))

def inference_result_handler(original_frame, infer_results, labels, config_data, tracker=None):
    """
    Processes inference results and draw detections (with optional tracking).

    Args:
        infer_results (list): Raw output from the model.
        original_frame (np.ndarray): Original image frame.
        labels (list): List of class labels.
        enable_tracking (bool): Whether tracking is enabled.
        tracker (BYTETracker, optional): ByteTrack tracker instance.

    Returns:
        np.ndarray: Frame with detections or tracks drawn.
    """
    detections = extract_detections(original_frame, infer_results, config_data)  #should return dict with boxes, classes, scores
    frame_with_detections = draw_detections(detections, original_frame, labels, tracker=tracker)
    return frame_with_detections


def draw_detection(image: np.ndarray, box: list, labels: list, score: float, color: tuple, track=False, fall_down=False):
    ymin, xmin, ymax, xmax = map(int, box)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    top_text = f"{labels[0]}: {score:.1f}%" if not track or len(labels) == 2 else f"{score:.1f}%"

    if fall_down:
        bottom_text = "fall_down"
        bottom_font_scale = 1  
        bottom_thickness = 3

        pos = (xmax - 100, ymax-10)  
        text_color = (0, 0, 255)   
        border_color = (0, 0, 0)   
    else:
        if track and len(labels) == 2:
            bottom_text = labels[1]
        elif track:
            bottom_text = labels[0]
        else:
            bottom_text = None
        bottom_font_scale = 0.5
        bottom_thickness = 1
        pos = (xmax - 70, ymax - 6)
        text_color = (255, 255, 255)  
        border_color = (0, 0, 0)      

    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, border_color, 2, cv2.LINE_AA)
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if bottom_text:
        cv2.putText(image, bottom_text, pos, font, bottom_font_scale, border_color, bottom_thickness, cv2.LINE_AA)
        cv2.putText(image, bottom_text, pos, font, bottom_font_scale, text_color, bottom_thickness - 1, cv2.LINE_AA)



def denormalize_and_rm_pad(box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
    """
    Denormalize bounding box coordinates and remove padding.

    Args:
        box (list): Normalized bounding box coordinates.
        size (int): Size to scale the coordinates.
        padding_length (int): Length of padding to remove.
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.

    Returns:
        list: Denormalized bounding box coordinates with padding removed.
    """
    for i, x in enumerate(box):
        box[i] = int(x * size)
        if (input_width != size) and (i % 2 != 0):
            box[i] -= padding_length
        if (input_height != size) and (i % 2 == 0):
            box[i] -= padding_length

    return box

def is_fall_down(box, y_thresh, aspect_ratio_thresh=0.8):
    xmin, ymin, xmax, ymax = map(int, box)
    center_y = (ymin + ymax) // 2
    width = xmax - xmin
    height = ymax - ymin
    aspect_ratio = width / (height + 1e-5)
    print(f"center_y: {center_y}, y_thresh: {y_thresh}, aspect_ratio: {aspect_ratio}, aspect_ratio_thresh: {aspect_ratio_thresh}")

    return center_y > y_thresh and aspect_ratio < aspect_ratio_thresh


def extract_detections(image: np.ndarray, detections: list, config_data) -> dict:
    """
    Extract detections from the input data.

    Args:
        image (np.ndarray): Image to draw on.
        detections (list): Raw detections from the model.
        config_data (Dict): Loaded JSON config containing post-processing metadata.

    Returns:
        dict: Filtered detection results containing 'detection_boxes', 'detection_classes', 'detection_scores', and 'num_detections'.
    """

    visualization_params = config_data["visualization_params"]
    score_threshold = visualization_params.get("score_thres", 0.5)
    max_boxes = visualization_params.get("max_boxes_to_draw", 50)


    #values used for scaling coords and removing padding
    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []

    for class_id, detection in enumerate(detections):
        for det in detection:
            bbox, score = det[:4], det[4]
            if score >= score_threshold and class_id == 0:  # class_id == 0 means background, skip it
                denorm_bbox = denormalize_and_rm_pad(bbox, size, padding_length, img_height, img_width)
                all_detections.append((score, class_id, denorm_bbox))

    #sort all detections by score descending
    all_detections.sort(reverse=True, key=lambda x: x[0])

    #take top max_boxes
    top_detections = all_detections[:max_boxes]

    scores, class_ids, boxes = zip(*top_detections) if top_detections else ([], [], [])

    return {
        'detection_boxes': list(boxes),
        'detection_classes': list(class_ids),
        'detection_scores': list(scores),
        'num_detections': len(top_detections)
    }


def draw_detections(detections: dict, img_out: np.ndarray, labels, tracker=None):
    boxes = detections["detection_boxes"]
    scores = detections["detection_scores"]
    num_detections = detections["num_detections"]
    classes = detections["detection_classes"]

    fall_down_y_thresh = img_out.shape[0] * 0.5  
    aspect_ratio_thresh = 2  

    if tracker:
        pass
    else:
        for idx in range(num_detections):
            class_id = classes[idx]
            box = boxes[idx]
            score = scores[idx]
            
            if class_id == 0 and is_fall_down(box, fall_down_y_thresh, aspect_ratio_thresh):
                color = (0, 0, 255)  
                publish_fall_alert(box)
                draw_detection(img_out, box, [labels[class_id], "fall_down"], score * 100.0, color, fall_down=True)
            else:
                color = tuple(id_to_color(class_id).tolist())
                draw_detection(img_out, box, [labels[class_id]], score * 100.0, color)

    return img_out


def find_best_matching_detection_index(track_box, detection_boxes):
    """
    Finds the index of the detection box with the highest IoU relative to the given tracking box.

    Args:
        track_box (list or tuple): The tracking box in [x_min, y_min, x_max, y_max] format.
        detection_boxes (list): List of detection boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
        int or None: Index of the best matching detection, or None if no match is found.
    """
    best_iou = 0
    best_idx = -1

    for i, det_box in enumerate(detection_boxes):
        iou = compute_iou(track_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return best_idx if best_idx != -1 else None


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU measures the overlap between two boxes:
        IoU = (area of intersection) / (area of union)
    Values range from 0 (no overlap) to 1 (perfect overlap).

    Args:
        boxA (list or tuple): [x_min, y_min, x_max, y_max]
        boxB (list or tuple): [x_min, y_min, x_max, y_max]

    Returns:
        float: IoU value between 0 and 1.
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-5)
