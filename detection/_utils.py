import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
from datasets import COCOFB_INSTANCE_CATEGORY_NAMES as coco_names

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
# define the torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, scores, and 
    class labels. 
    """
    # transform the image to tensor
    image = transform(image).to(device)
    # add a batch dimension
    image = image.unsqueeze(0) 
    # get the predictions on the image
    with torch.no_grad():
        outputs = model(image) 
    # print(outputs)
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    scores = pred_scores[pred_scores >= detection_threshold]
    labels = outputs[0]['labels'][:len(boxes)]
    # get all the predicited class names
    pred_classes = [coco_names[i] for i in labels.cpu().numpy()]
    return boxes, pred_classes, scores, labels

def draw_boxes(boxes, classes, scores, labels, image):
    """
    Draws the bounding box around a detected object.
    """
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, f"{classes[i]}:{scores[i]*100.:.2f}%", (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image