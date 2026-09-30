import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Pneumonia AI Diagnostic", layout="centered")
st.title("🏥 نظام الذكاء الاصطناعي الطبي لتشخيص التهاب الرئة")

MODEL_PATH = "pneumonia_resnet18.pth"
# رابط الملف المباشر الخاص بحسابك على Hugging Face
MODEL_URL = "https://huggingface.co/eldhrawy/pneumonia-detector/resolve/main/pneumonia_resnet18.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("جاري تحميل أوزان النموذج من Hugging Face..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    class_names = ['NORMAL', 'PNEUMONIA']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device, class_names

model, device, class_names = load_model()

# فئة Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, class_idx, output

grad_cam = GradCAM(model, model.layer4)

uploaded_file = st.file_uploader("اختر صورة الأشعة السينية (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    orig_img = Image.open(uploaded_file).convert('RGB')
    resized_orig = orig_img.resize((224, 224))

    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform_eval(orig_img).unsqueeze(0).to(device)

    with st.spinner('جاري التحليل وتوليد الخريطة الحرارية...'):
        heatmap, pred_idx, output = grad_cam.generate_heatmap(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        confidence = probs[pred_idx].item() * 100
        pred_class = class_names[pred_idx]

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        resized_orig_np = np.array(resized_orig)
        overlay = cv2.addWeighted(resized_orig_np, 0.55, heatmap_colored, 0.45, 0)

    st.success("تم التحليل بنجاح!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(orig_img, caption="الصورة الأصلية", use_container_width=True)
    with col2:
        st.image(overlay, caption="الخريطة الحرارية (Grad-CAM)", use_container_width=True)

    st.metric(label="التشخيص النهائي", value=pred_class)
    st.metric(label="نسبة التأكد (Confidence)", value=f"{confidence:.2f}%")
