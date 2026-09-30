import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# ضبط إعدادات الصفحة
st.set_page_config(
    page_title="تشخيص التهاب الرئة بالذكاء الاصطناعي",
    page_icon="🏥",
    layout="centered"
)

# عنوان التطبيق
st.title("🏥 نظام الذكاء الاصطناعي الطبي لتشخيص التهاب الرئة")
st.write("قم بتحميل صورة الأشعة السينية (X-Ray) للصدر للحصول على التقييم التشخيصي.")

# تحديد الجهاز دائماً كـ CPU لتفادي مشاكل السيرفرات السحابية
DEVICE = torch.device('cpu')
MODEL_PATH = "best_model.pth"

# دالة تحميل النموذج مع التخزين المؤقت لتسريع الأداء
@st.cache_resource
def load_model_and_assets():
    # 1. إعادة بناء هيكل النموذج (ResNet18)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # 2. تحميل الأوزان بشكل آمن وتوجيهها إلى CPU
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
    else:
        st.error(f"لم يتم العثور على ملف النموذج: {MODEL_PATH}")
        st.stop()
        
    model.to(DEVICE)
    model.eval()
    
    class_names = ["Normal", "Pneumonia"]
    return model, class_names

# تحميل الموديل
try:
    model, class_names = load_model_and_assets()
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل النموذج: {e}")
    st.stop()

# تحويلات الصورة المطابقة لمعالجة ResNet18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# واجهة رفع الملف
uploaded_file = st.file_uploader("اختر صورة الأشعة (JPG / PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="صورة الأشعة المرفوعة", use_column_width=True)
    
    # زر التشخيص
    if st.button("تشخيص الصورة"):
        with st.spinner("جاري تحليل الصورة بواسطة النموذج..."):
            # تجهيز الصورة للموديل
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            # إجراء التنبؤ
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                
            predicted_class = class_names[predicted.item()]
            score = confidence.item() * 100

        # عرض النتيجة
        st.subheader("نتيجة التحليل:")
        if predicted_class == "Pneumonia":
            st.error(f"⚠️ **النتيجة: احتمال وجود التهاب رئوي (Pneumonia)**\n\nنسبة التأكد: **{score:.2f}%**")
        else:
            st.success(f"✅ **النتيجة: الأشعة سليمة (Normal)**\n\nنسبة التأكد: **{score:.2f}%**")
