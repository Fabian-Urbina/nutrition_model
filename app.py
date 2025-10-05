import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io, joblib
from huggingface_hub import hf_hub_download
from model.nutrition_cnn import NutritionCNN

# Configuración
st.set_page_config(page_title="Macronutrient Estimator", page_icon="🍽️", layout="centered")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Descargar modelo y scaler
repo_id = "fito-zenden/nutrition_model_01"
model_path = hf_hub_download(repo_id=repo_id, filename="nutrition_model_best.pth")
scaler_path = hf_hub_download(repo_id=repo_id, filename="nutrition_scaler.pkl")

# Cargar modelo
model = NutritionCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.eval()

# Cargar scaler
scaler = joblib.load(scaler_path)

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("🍽️ Estimador de Macronutrientes")
st.write("Sube una imagen de comida y el modelo estimará sus valores nutricionales.")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_dict = model(input_tensor)
        output_tensor = torch.cat(list(output_dict.values()), dim=1)
        output_np = scaler.inverse_transform(output_tensor.cpu().numpy())

    # Mostrar resultados
    labels = ["Calorías", "Masa (g)", "Grasa (g)", "Carbohidratos (g)", "Proteínas (g)"]
    results = {labels[i]: round(float(output_np[0, i]), 2) for i in range(5)}

    st.subheader("🔍 Predicción:")
    st.table(results)
