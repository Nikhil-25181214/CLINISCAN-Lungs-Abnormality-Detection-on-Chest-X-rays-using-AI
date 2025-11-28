import torch
import torch.nn as nn
from torchvision import models, transforms

# -------------------------------------------------------------
# CLASS NAMES (15 classes)
# -------------------------------------------------------------
CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltrate",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Pleural Thickening",
    "Normal"
]


# -------------------------------------------------------------
# BUILD RESNET50 EXACTLY LIKE TRAINING
# -------------------------------------------------------------
def create_resnet50(num_classes=15):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(in_features, num_classes))
    return model


# -------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------
def load_classification_model():
    model = create_resnet50(15)

    state = torch.load(
        r"A:\HRITHIK\models\resnet50_best.pth",
        map_location="cpu"
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()
    return model


clf_model = load_classification_model()


# -------------------------------------------------------------
# TRANSFORM
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -------------------------------------------------------------
# CLASSIFICATION FUNCTION (2 arguments!)
# -------------------------------------------------------------
def classify_image(image, class_names):
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = clf_model(img_tensor)
        probs = torch.sigmoid(logits)[0]

    results = [
        (class_names[i], float(probs[i]))
        for i in range(len(class_names))
    ]
    return sorted(results, key=lambda x: x[1], reverse=True)
