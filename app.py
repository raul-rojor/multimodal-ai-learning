import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from tinyclip_improved15 import TinyCLIPImproved

st.set_page_config(page_title="TinyCLIP Image Classifier", layout="centered")

st.title("TinyCLIP — Zero-Shot Image Classifier")
st.markdown("Upload an image and get a zero-shot prediction using a CLIP-style model trained on 5,000 COCO images.")

# Load model
@st.cache_resource
def load_model():
    model = TinyCLIPImproved()
    model.load_state_dict(torch.load('./checkpoints/improved_model.pt', map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_tokenizer_and_classes():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
    prompts = [f"a photo of a {c}" for c in classes]
    tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    return tokenizer, classes, tokens

model = load_model()
tokenizer, classes, tokens = load_tokenizer_and_classes()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    from PIL import ImageOps
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img = transform(image).unsqueeze(0)

        with torch.no_grad():
            text_embs = model.text_encoder(tokens['input_ids'], tokens['attention_mask'])
            img_emb = model.image_encoder(img)
            sim = img_emb @ text_embs.T
            probs = torch.softmax(sim, dim=1)
            top_idx = probs.argmax().item()
            confidence = probs[0][top_idx].item() * 100

    st.success(f"**Prediction:** {classes[top_idx]} ({confidence:.1f}%)")

    # Show top-5
    top5 = probs[0].topk(5)
    st.markdown("**Top 5 predictions:**")
    for prob, idx in zip(top5.values, top5.indices):
        st.write(f"- {classes[idx.item()]}: {prob.item()*100:.1f}%")