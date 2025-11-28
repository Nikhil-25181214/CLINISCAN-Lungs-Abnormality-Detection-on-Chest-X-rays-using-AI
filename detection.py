import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ---------------------------------------------------
# LOAD FASTER R-CNN OFFLINE
# ---------------------------------------------------
def load_frcnn():
    model_path = "models/faster_rcnn_best.pth"

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    num_classes = 9   # 8 classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


frcnn = load_frcnn()

# Your class names EXACTLY in the order the model was trained
CLASS_NAMES = [
    "background", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax"
]


# ---------------------------------------------------
# DETECTION FUNCTION
# ---------------------------------------------------
def detect_image(image_np):

    img = Image.fromarray(image_np).convert("RGB")
    img_tensor = F.to_tensor(img)

    with torch.no_grad():
        out = frcnn([img_tensor])[0]

    boxes = out["boxes"]
    scores = out["scores"]
    labels = out["labels"]

    # -----------------------------
    # CONF THRESHOLD
    # -----------------------------
    conf_mask = scores > 0.50
    boxes = boxes[conf_mask]
    scores = scores[conf_mask]
    labels = labels[conf_mask]

    # -----------------------------
    # NMS
    # -----------------------------
    if len(boxes) > 0:
        keep = nms(boxes, scores, iou_threshold=0.4)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    # -----------------------------
    # DRAW & BUILD RETURN RESULT
    # -----------------------------
    draw = ImageDraw.Draw(img)
    predictions = []

    for box, score, label_id in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()

        class_name = CLASS_NAMES[label_id]

        predictions.append((class_name, float(score)))

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, y1), f"{class_name} {score:.2f}", fill="yellow")

    return img, predictions
