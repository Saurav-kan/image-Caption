import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from transformers import BlipProcessor, BlipForConditionalGeneration

# Define COCO instance category names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
      'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' 
]

# load up our caption+detector models
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )  # captioner model

    detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
    detection_model.eval() 

    return processor, caption_model, detection_model


@st.cache_resource
# cache so the program doesn't need to constantly reload
def get_models():
    return load_models()


@st.cache_resource
# caching the transform too
def get_transform():
    return T.Compose([
        T.ToTensor(),   # convert PIL to tensor
    ])


def predict_caption(image: Image.Image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)  # generate caption tokens
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption  # return text


def predict_detection(image: Image.Image, model, transform, threshold=0.5):
    tensor = transform(image)
    with torch.no_grad():
        outputs = model([tensor])[0]  # pass list for faster rcnn
    boxes  = outputs['boxes']
    labels = outputs['labels']
    scores = outputs['scores']
    # filter by score threshold
    keep = scores >= threshold
    return boxes[keep], labels[keep], scores[keep]


def draw_boxes(image: Image.Image, boxes, labels, scores, label_map):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, lbl, scr in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{label_map.get(lbl.item(), lbl.item())}: {scr:.2f}"
        draw.text((x1, y1), text, fill="red", font=font)
    return image   # annotated image


def main():
    st.title("🌕🔥🌕Streamlit Image Insight")
    st.write("Upload an image to get a caption and object detection overlays.")
    # load models and transform only once

    processor, caption_model, detection_model = get_models()
    transform = get_transform()

    # only allow image uploads
    uploaded = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # generate caption   
        with st.spinner("Creating caption..."):
            caption = predict_caption(image, processor, caption_model)
        st.markdown(f"**Caption:** {caption}")
        with st.spinner("Looking for any objects..."):
            boxes, labels, scores = predict_detection(image, detection_model, transform)
        # Use coco labels or objects
        label_map = {i: name for i, name in enumerate(COCO_INSTANCE_CATEGORY_NAMES)}
        result_img = image.copy()
        result_img = draw_boxes(result_img, boxes, labels, scores, label_map)
        st.image(result_img, caption="Detection Results", use_column_width=True)


if __name__ == "__main__":
    main()
