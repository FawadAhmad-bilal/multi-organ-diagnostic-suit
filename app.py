
import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from keras.applications.vgg16 import preprocess_input as vgg_preprocess


from grad_cam import make_gradcam_heatmap, overlay_heatmap

# ===========================================================================
# 1) MODEL LOGIC — same as before, nothing here changes prediction behavior
# ===========================================================================

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# last_conv_layer: the exact layer name Grad-CAM will hook into. These were
# read directly out of each .keras file's config.json, so they're accurate
# for chest / malaria / skin_cancer / vgg. The eye_prediction entry assumes
# a standard ResNet50 top (same as chest_model) since that file was too
# large to inspect here — confirm the layer name matches by running
# `model.summary()` once and checking the last activation before the
# GlobalAveragePooling2D layer; edit LAST_CONV_EYE below if it differs.
LAST_CONV_EYE = "conv5_block3_out"

MODEL_CONFIGS = {
    "Chest X-ray (COVID-19 Radiography)": {
        "path": os.path.join(MODEL_DIR, "chest_model.keras"),
        "img_size": (224, 224),
        "class_names": ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"],
        "preprocess": resnet_preprocess,
        "last_conv_layer": "conv5_block3_out",
        "icon": "🫁",
    },
    "Malaria Cell Image": {
        "path": os.path.join(MODEL_DIR, "malaria_model.keras"),
        "img_size": (128, 128),
        "class_names": ["Parasitized", "Uninfected"],
        "preprocess": mobilenet_preprocess,
        "last_conv_layer": "out_relu",
        "icon": "🦟",
    },
    "Skin Lesion (HAM10000)": {
        "path": os.path.join(MODEL_DIR, "skin_cancer.keras"),
        "img_size": (128, 128),
        "class_names": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "preprocess": efficientnet_preprocess,
        "last_conv_layer": "top_activation",
        "icon": "🩹",
    },
    "Brain Tumor MRI": {
        "path": os.path.join(MODEL_DIR, "vgg_model.keras"),
        "img_size": (128, 128),
        "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
        "preprocess": vgg_preprocess,
        "last_conv_layer": "block5_conv3",
        "icon": "🧠",
    },
    "Eye Fundus (Diabetic Retinopathy)": {
        "path": os.path.join(MODEL_DIR, "eye_prediction.keras"),
        "img_size": (224, 224),
        "class_names": ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative DR"],
        "preprocess": resnet_preprocess,
        "last_conv_layer": LAST_CONV_EYE,
        "icon": "👁️",
    },
}

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi (common mole)",
    "vasc": "Vascular lesions",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return tf.keras.models.load_model(path)


def preprocess_image(pil_img, img_size, preprocess_fn):
    """Resize + apply the SAME preprocessing the model saw during training.
    Returns both the model-ready batch and the plain 0-255 array used for
    Grad-CAM's visual overlay (they must NOT be the same array — the overlay
    needs raw pixel values, not ImageNet-normalized ones)."""
    resized = pil_img.convert("RGB").resize(img_size)
    raw_array = np.array(resized).astype("float32")
    batch = np.expand_dims(raw_array.copy(), axis=0)
    batch = preprocess_fn(batch)
    return batch, raw_array


def run_prediction(config, pil_img):
    """Pure logic: no st.* calls in here, so it can't break from UI edits."""
    model = load_model(config["path"])
    batch, raw_array = preprocess_image(pil_img, config["img_size"], config["preprocess"])
    preds = model.predict(batch, verbose=0)[0]
    pred_index = int(np.argmax(preds))
    return {
        "model": model,
        "batch": batch,
        "raw_array": raw_array,
        "preds": preds,
        "pred_index": pred_index,
        "pred_class": config["class_names"][pred_index],
        "confidence": float(preds[pred_index]) * 100,
    }


# ===========================================================================
# 2) UI / PRESENTATION LAYER — everything below is purely visual
# ===========================================================================

# ---- EDIT THESE TWO LINES FOR THE FOOTER --------------------------------
YOUR_NAME = "Fawad Ahmad"
GITHUB_URL = "https://github.com/FawadAhmad-bilal"
# --------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg:        #0b1220;
    --panel:     #111a2c;
    --panel-alt: #16213a;
    --border:    #22304a;
    --accent:    #2dd4bf;
    --accent-2:  #38bdf8;
    --text:      #e6edf3;
    --muted:     #8a97ab;
    --danger:    #f87171;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* Page background */
.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #12213a 0%, var(--bg) 45%) fixed;
    color: var(--text);
}

/* Hide Streamlit's default chrome so the app doesn't look like a default demo.
   IMPORTANT: the header must stay visible (not visibility:hidden) because
   the sidebar's collapse/expand arrow lives inside it in current Streamlit
   versions. Hiding the whole header hides that arrow too, with no way to
   bring a collapsed sidebar back. Instead, make it blend into the page:
   transparent background, no border, and hide only the hamburger menu icon
   and the "Made with Streamlit" footer text specifically. */
#MainMenu, footer { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1150px; }

/* ---------------- Sidebar ---------------- */
section[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--text); }

/* ---------------- Hero header ---------------- */
.hero {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 28px 32px;
    background: linear-gradient(135deg, var(--panel) 0%, var(--panel-alt) 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    margin-bottom: 28px;
}
.hero-icon {
    font-size: 2.6rem;
    line-height: 1;
    filter: drop-shadow(0 0 12px rgba(45, 212, 191, 0.35));
}
.hero-title {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, var(--text) 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-tagline {
    margin: 4px 0 0 0;
    color: var(--muted);
    font-size: 0.95rem;
}

/* ---------------- Section / card ---------------- */
.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 22px;
}
.card h3 {
    margin-top: 0;
    font-size: 1.05rem;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-label {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem;
    color: var(--muted);
    font-weight: 600;
}

/* ---------------- Upload widget ---------------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel-alt) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; }

/* ---------------- Result card ---------------- */
.result-card {
    background: linear-gradient(135deg, rgba(45,212,191,0.10) 0%, rgba(56,189,248,0.06) 100%);
    border: 1px solid rgba(45,212,191,0.35);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 22px;
}
.result-class {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--accent);
    margin: 0;
}
.result-confidence {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 2px 0 0 0;
}
.result-desc { color: var(--text); opacity: 0.85; margin-top: 8px; font-size: 0.9rem; }

/* ---------------- Custom probability bars ---------------- */
.prob-row { margin-bottom: 10px; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: var(--text);
    margin-bottom: 4px;
}
.prob-track {
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, var(--accent-2), var(--accent));
}
.prob-fill.top { background: linear-gradient(90deg, var(--accent), #6ee7d8); }

/* ---------------- Disclaimer ---------------- */
.disclaimer {
    border: 1px solid rgba(248,113,113,0.35);
    background: rgba(248,113,113,0.06);
    border-radius: 12px;
    padding: 12px 16px;
    color: var(--danger);
    font-size: 0.82rem;
}

/* ---------------- Footer ---------------- */
.app-footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
    padding-top: 18px;
    border-top: 1px solid var(--border);
    margin-top: 8px;
}
.app-footer a { color: var(--accent-2); text-decoration: none; }
.app-footer a:hover { text-decoration: underline; }
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-icon">🩺</div>
            <div>
                <p class="hero-title">Multi-Organ Diagnostic Suite</p>
                <p class="hero-tagline">
                    Five transfer-learning models, one dashboard — with Grad-CAM
                    explainability for every prediction.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Returns (module_name, config, show_gradcam)."""
    st.sidebar.markdown("### 🧭 Diagnostic Module")
    module = st.sidebar.radio(
        "Choose an organ / scan type",
        list(MODEL_CONFIGS.keys()),
        format_func=lambda name: f"{MODEL_CONFIGS[name]['icon']}  {name}",
        label_visibility="collapsed",
    )
    config = MODEL_CONFIGS[module]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div class="card-label">Model info</div>
        <div style="color: var(--text); font-size: 0.9rem; line-height:1.7;">
            Input size: <b>{config['img_size'][0]}×{config['img_size'][1]}</b><br>
            Classes: <b>{len(config['class_names'])}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM heatmap", value=True)
    return module, config, show_gradcam


def render_upload_card(module, config):
    st.markdown(
        f"""
        <div class="card">
            <h3>{config['icon']} Upload — {module}</h3>
            <p style="color: var(--muted); font-size: 0.88rem; margin-top: -6px;">
                JPG or PNG. The image is resized to
                {config['img_size'][0]}×{config['img_size'][1]} and preprocessed
                exactly as during training before prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key=module,
        label_visibility="collapsed",
    )


def render_images(pil_img, overlaid_img, show_gradcam):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>🖼️ Uploaded Image</h3>', unsafe_allow_html=True)
        st.image(pil_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h3>🔥 Grad-CAM Focus Map</h3>', unsafe_allow_html=True)
        if show_gradcam:
            st.image(overlaid_img, use_container_width=True)
        else:
            st.write("Enable Grad-CAM in the sidebar to see the focus map.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_result_card(result, config):
    pred_class = result["pred_class"]
    confidence = result["confidence"]
    short_code = pred_class.split(" ")[0].lower()
    description = CLASS_DESCRIPTIONS.get(short_code, "")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-label">Prediction</div>
            <p class="result-class">{pred_class}</p>
            <p class="result-confidence">{confidence:.2f}% confidence</p>
            {f'<p class="result-desc">{description}</p>' if description else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(result, config):
    st.markdown('<div class="card"><h3>📊 Class Probabilities</h3>', unsafe_allow_html=True)
    top_index = result["pred_index"]
    rows = ""
    for i, (name, p) in enumerate(zip(config["class_names"], result["preds"])):
        pct = float(p) * 100
        fill_class = "prob-fill top" if i == top_index else "prob-fill"
        rows += f"""
        <div class="prob-row">
            <div class="prob-label"><span>{name}</span><span>{pct:.1f}%</span></div>
            <div class="prob-track"><div class="{fill_class}" style="width:{pct:.1f}%;"></div></div>
        </div>
        """
    st.markdown(rows + "</div>", unsafe_allow_html=True)


def render_footer():
    st.markdown(
        f"""
        <div class="app-footer">
            Built by {YOUR_NAME} &nbsp;·&nbsp;
            <a href="{GITHUB_URL}" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Multi-Organ Diagnostic Suite",
        page_icon="🩺",
        layout="wide",
    )
    inject_css()
    render_hero()

    module, config, show_gradcam = render_sidebar()
    uploaded_file = render_upload_card(module, config)

    if uploaded_file is None:
        st.info("⬆️ Upload an image above to run a prediction.")
        render_footer()
        return

    if not os.path.exists(config["path"]):
        st.error(
            f"Model file not found at `{config['path']}`. "
            "Place all .keras files in the same folder as app.py."
        )
        render_footer()
        return

    pil_img = Image.open(uploaded_file)
    result = run_prediction(config, pil_img)

    overlaid_img = None
    if show_gradcam:
        heatmap, _ = make_gradcam_heatmap(
            result["batch"], result["model"], config["last_conv_layer"],
            pred_index=result["pred_index"],
        )
        overlaid_img = overlay_heatmap(result["raw_array"], heatmap)

    render_images(pil_img, overlaid_img, show_gradcam)
    render_result_card(result, config)
    render_probability_bars(result, config)
    render_footer()


if __name__ == "__main__":
    main()