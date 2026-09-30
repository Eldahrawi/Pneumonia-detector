import os
import urllib.request
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="تشخيص التهاب الرئة بالذكاء الاصطناعي",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 نظام الذكاء الاصطناعي الطبي لتشخيص التهاب الرئة")
st.write("قم بتحميل صورة الأشعة السينية (X-Ray) للصدر للحصول على التقييم التشخيصي.")

DEVICE = torch.device('cpu')
MODEL_PATH = "pneumonia_resnet18.pth"
MODEL_URL = "https://huggingface.co/eldhrawy/pneumonia-detector/resolve/main/pneumonia_resnet18.pth"

# 2. تحسين دالة التنزيل والتحميل مع تقليل استهلاك الذاكرة
@st.cache_resource(show_spinner=False)
def load_model_and_assets():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    # بناء النموذج وتعديل الطبقة الأخيرة مباشرة
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # تحميل الأوزان وتصفيتها لتفادي استهلاك الذاكرة
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model_dict = model.state_dict()
    
    pretrained_dict = {
        k: v for k, v in state_dict.items() 
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model.to(DEVICE)
    model.eval()
    
    class_names = ["Normal", "Pneumonia"]
    return model, class_names

# تحميل النموذج
with st.spinner("جاري تهيئة النظام..."):
    try:
        model, class_names = load_model_and_assets()
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل النموذج: {e}")
        st.stop()

# 3. تحويلات الصورة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. واجهة الرفع والتحليل
uploaded_file = st.file_uploader("اختر صورة الأشعة (JPG / PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="صورة الأشعة المرفوعة", use_container_width=True)
    
    if st.button("بدء التشخيص", type="primary"):
        with st.spinner("جاري تحليل الصورة..."):
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                
            predicted_class = class_names[predicted.item()]
            score = confidence.item() * 100

        st.subheader("نتيجة التحليل:")
        if predicted_class == "Pneumonia":
            st.error(f"⚠️ **النتيجة: احتمال وجود التهاب رئوي (Pneumonia)**\n\nنسبة التأكد: **{score:.2f}%**")
        else:
            st.success(f"✅ **النتيجة: الأشعة سليمة (Normal)**\n\nنسبة التأكد: **{score:.2f}%**")
