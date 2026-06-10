import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess, decode_predictions

# ── Konfigurasi Halaman Streamlit ──────────────────────────────────
st.set_page_config(
    page_title="FishCheck — Kesegaran Ikan Bandeng",
    page_icon="🐟",
    layout="wide"
)

# ── Pengaturan Model ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 60.0

# ── CSS Premium & Modern (Tema Gelap Aqua Glassmorphism) ───────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    /* Disable scrollbar visual representation to prevent iframe layout shifting */
    ::-webkit-scrollbar {
        display: none;
    }
    html {
        -ms-overflow-style: none;  /* IE and Edge */
        scrollbar-width: none;  /* Firefox */
    }
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    .stApp { 
        background: linear-gradient(135deg, #070e14 0%, #0d1e26 50%, #112733 100%); 
        min-height: 100vh; 
        color: #e2f1f7;
    }
    
    /* Hero Banner */
    .hero {
        background: linear-gradient(135deg, rgba(27, 122, 156, 0.2) 0%, rgba(16, 74, 97, 0.15) 50%, rgba(8, 43, 58, 0.2) 100%);
        border-radius: 24px; 
        padding: 45px 30px; 
        text-align: center;
        margin-bottom: 35px; 
        border: 1px solid rgba(176, 228, 245, 0.12);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }
    .hero:hover {
        border-color: rgba(176, 228, 245, 0.25);
        box-shadow: 0 16px 45px rgba(27, 122, 156, 0.15);
    }
    .hero h1 { 
        color: #ffffff; 
        font-size: 3.2rem; 
        font-weight: 700; 
        margin: 0; 
        letter-spacing: -0.5px;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
    }
    .hero p { 
        color: #aae0f2; 
        font-size: 1.25rem; 
        margin-top: 12px; 
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block; 
        background: linear-gradient(90deg, rgba(27, 122, 156, 0.25), rgba(16, 74, 97, 0.25));
        color: #5ce1e6; 
        padding: 6px 18px; 
        border-radius: 50px;
        font-size: 0.85rem; 
        margin-bottom: 20px;
        font-weight: 600;
        letter-spacing: 0.8px;
        border: 1px solid rgba(92, 225, 230, 0.25);
    }
    
    /* Result Cards */
    .result-fresh {
        background: linear-gradient(135deg, rgba(8, 77, 43, 0.45) 0%, rgba(16, 117, 66, 0.35) 100%);
        border: 2px solid #2ecc71; 
        border-radius: 20px; 
        padding: 35px;
        text-align: center; 
        box-shadow: 0 10px 35px rgba(46, 204, 113, 0.15);
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.5s ease;
    }
    .result-fresh h2 { color: #2ecc71; font-size: 2.3rem; margin: 0; font-weight: 700; letter-spacing: 0.5px; }
    .result-fresh h3 { color: #ffffff; font-size: 1.35rem; margin: 12px 0 6px 0; }
    .result-fresh p  { color: #c4f9da; margin: 0; font-size: 1rem; }
    
    .result-moderate {
        background: linear-gradient(135deg, rgba(92, 65, 9, 0.45) 0%, rgba(140, 99, 17, 0.35) 100%);
        border: 2px solid #f39c12; 
        border-radius: 20px; 
        padding: 35px;
        text-align: center; 
        box-shadow: 0 10px 35px rgba(243, 156, 18, 0.15);
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.5s ease;
    }
    .result-moderate h2 { color: #f39c12; font-size: 2.3rem; margin: 0; font-weight: 700; letter-spacing: 0.5px; }
    .result-moderate h3 { color: #ffffff; font-size: 1.35rem; margin: 12px 0 6px 0; }
    .result-moderate p  { color: #fef1d2; margin: 0; font-size: 1rem; }
    
    .result-stale {
        background: linear-gradient(135deg, rgba(84, 8, 8, 0.45) 0%, rgba(130, 16, 16, 0.35) 100%);
        border: 2px solid #e74c3c; 
        border-radius: 20px; 
        padding: 35px;
        text-align: center; 
        box-shadow: 0 10px 35px rgba(231, 76, 60, 0.15);
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.5s ease;
    }
    .result-stale h2 { color: #e74c3c; font-size: 2.3rem; margin: 0; font-weight: 700; letter-spacing: 0.5px; }
    .result-stale h3 { color: #ffffff; font-size: 1.35rem; margin: 12px 0 6px 0; }
    .result-stale p  { color: #fddcd9; margin: 0; font-size: 1rem; }
    
    .result-uncertain {
        background: linear-gradient(135deg, rgba(61, 74, 82, 0.45) 0%, rgba(86, 103, 114, 0.35) 100%);
        border: 2px solid #bdc3c7; 
        border-radius: 20px; 
        padding: 35px;
        text-align: center; 
        box-shadow: 0 10px 35px rgba(189, 195, 199, 0.15);
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.5s ease;
    }
    .result-uncertain h2 { color: #bdc3c7; font-size: 2.3rem; margin: 0; font-weight: 700; letter-spacing: 0.5px; }
    .result-uncertain h3 { color: #ffffff; font-size: 1.35rem; margin: 12px 0 6px 0; }
    .result-uncertain p  { color: #f4f6f7; margin: 0; font-size: 1rem; }
    
    /* Custom Progress bar styling */
    .score-container {
        background: rgba(255, 255, 255, 0.02); 
        border-radius: 16px;
        padding: 16px 20px; 
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 16px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .score-container:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(27, 122, 156, 0.25);
        transform: translateX(4px);
    }
    .score-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .score-label { 
        color: #aae0f2; 
        font-size: 0.95rem; 
        font-weight: 500; 
    }
    .score-value { 
        font-size: 1.35rem; 
        font-weight: 700; 
        color: #ffffff;
    }
    .progress-bar-bg {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        height: 10px;
        width: 100%;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .fill-fresh {
        background: linear-gradient(90deg, #107542, #2ecc71);
        box-shadow: 0 0 12px rgba(46, 204, 113, 0.4);
    }
    .fill-moderate {
        background: linear-gradient(90deg, #8c6311, #f39c12);
        box-shadow: 0 0 12px rgba(243, 156, 18, 0.4);
    }
    .fill-rotten {
        background: linear-gradient(90deg, #821010, #e74c3c);
        box-shadow: 0 0 12px rgba(231, 76, 60, 0.4);
    }
    
    /* Other Elements */
    .section-title {
        color: #aae0f2; 
        font-size: 1.15rem; 
        font-weight: 600;
        text-transform: uppercase; 
        letter-spacing: 1.5px; 
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card {
        background: rgba(255, 255, 255, 0.02); 
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px; 
        padding: 24px; 
        backdrop-filter: blur(10px); 
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }
    .card:hover {
        border-color: rgba(27, 122, 156, 0.15);
    }
    .footer {
        text-align: center; 
        color: rgba(170, 224, 242, 0.35); 
        font-size: 0.85rem;
        padding: 30px 20px; 
        border-top: 1px solid rgba(255, 255, 255, 0.04); 
        margin-top: 50px;
        line-height: 1.5;
    }
    
    /* Custom Styling for streamlit file uploader */
    div[data-testid="stFileUploader"] {
        background: rgba(27, 122, 156, 0.05); 
        border-radius: 16px; 
        padding: 20px;
        border: 2px dashed rgba(170, 224, 242, 0.2) !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        background: rgba(27, 122, 156, 0.08); 
        border-color: rgba(92, 225, 230, 0.4) !important;
    }
    
    /* Animation Keyframes */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Holographic Visual Scanner */
    .scanner-box {
        position: relative; 
        display: inline-block;
        width: 170px; 
        height: 100px;
        border: 2px dashed rgba(92, 225, 230, 0.3);
        border-radius: 16px; 
        margin: 20px auto; 
        overflow: hidden;
        background: rgba(27, 122, 156, 0.06);
    }
    .scanner-line {
        position: absolute; 
        top: 0; 
        left: 0; 
        width: 100%; 
        height: 4px;
        background: linear-gradient(90deg, transparent, #2ecc71, transparent); 
        box-shadow: 0 0 12px #2ecc71;
        animation: scan 2.5s ease-in-out infinite;
    }
    .fish-icon {
        font-size: 55px; 
        position: absolute; 
        top: 50%; 
        left: 50%;
        transform: translate(-50%, -50%); 
        opacity: 0.95;
        animation: pulse 2.2s ease-in-out infinite;
    }
    @keyframes scan { 
        0% { top: -5%; } 
        50% { top: 105%; } 
        100% { top: -5%; } 
    }
    @keyframes pulse {
        0% { transform: translate(-50%, -50%) scale(0.96); opacity: 0.65; }
        50% { transform: translate(-50%, -50%) scale(1.06); opacity: 1; }
        100% { transform: translate(-50%, -50%) scale(0.96); opacity: 0.65; }
    }
    
    /* Responsive Media Queries */
    @media (max-width: 768px) {
        .hero { padding: 30px 15px; border-radius: 18px; margin-bottom: 25px; }
        .hero h1 { font-size: 2.1rem; }
        .hero p { font-size: 1.05rem; }
        .result-fresh, .result-moderate, .result-stale, .result-uncertain { padding: 25px 15px; }
        .result-fresh h2, .result-moderate h2, .result-stale h2, .result-uncertain h2 { font-size: 1.8rem; }
        .score-value { font-size: 1.15rem; }
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model Utama & Validator ───────────────────────────────────
@st.cache_resource
def load_main_model():
    return tf.keras.models.load_model('model_efficientnet_bandeng.keras')

@st.cache_resource
def load_validator():
    return MobileNetV2(weights='imagenet')

try:
    model = load_main_model()
    validator_model = load_validator()
except Exception as e:
    st.error(f"Gagal memuat file model: {e}. Pastikan file 'model_efficientnet_bandeng.keras' berada di folder proyek.")

# ── Fungsi Bantuan (Preprocessing & Validasi) ──────────────────────
def preprocess_for_freshness(img_pil):
    img = img_pil.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = efficientnet_preprocess(img_array)
    return img_array

def check_is_fish(img_pil):
    img = img_pil.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = mobilenet_preprocess(img_array)
    
    preds = validator_model.predict(img_array, verbose=0)
    decoded = decode_predictions(preds, top=10)[0]
    
    fish_keywords = ['fish', 'tench', 'shark', 'ray', 'sturgeon', 'gar', 'eel', 'coho']
    
    for _, label, _ in decoded:
        label_lower = label.lower()
        for kw in fish_keywords:
            if kw in label_lower:
                return True, decoded
    return False, decoded

# ── HERO SECTION ───────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🤖 POWERED BY EFFICIENTNETB0 · DUAL-STAGE SYSTEM</div>
    <h1>🐟 FishCheck</h1>
    <p>Sistem Deteksi Tingkat Kesegaran Ikan Bandeng (Milkfish) Berbasis Computer Vision</p>
</div>
""", unsafe_allow_html=True)

# ── Inisialisasi Session State untuk Riwayat Pengujian ──────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Upload Gambar ──────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload foto ikan Bandeng (JPG / PNG)",
    type=['jpg', 'jpeg', 'png'],
    label_visibility="collapsed"
)

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    with st.spinner('🛡️ Memvalidasi jenis objek dalam gambar...'):
        is_fish, top_preds = check_is_fish(img)
        
    if not is_fish:
        st.markdown(f"""
        <div class="result-stale" style="margin-top: 20px; margin-bottom: 25px;">
            <h2>🚫 OBJEK DITOLAK</h2>
            <h3>Gambar Ini Diduga Bukan Ikan</h3>
            <p>Sistem menyaring gambar ini karena teridentifikasi sebagai objek non-target.</p>
            <p style="font-size:0.85rem; margin-top:12px; color:#fccac6; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 8px;">
                (Prediksi AI mendeteksi objek: {', '.join([p[1].replace('_', ' ') for p in top_preds[:3]])})
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_img, _ = st.columns([1, 1])
        with col_img:
            st.markdown('<p class="section-title">📷 Foto Yang Diunggah</p>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        st.stop()
    
    # Menjalankan Preprocessing & Prediksi
    img_array = preprocess_for_freshness(img)
    
    with st.spinner('🔍 Menganalisis kondisi kesegaran ikan...'):
        predictions = model.predict(img_array, verbose=0)[0]
    
    p_busuk = float(predictions[0])
    p_kurang = float(predictions[1])
    p_segar = float(predictions[2])
    
    pred_idx = np.argmax(predictions)
    confidence = predictions[pred_idx] * 100
    
    classes = ['Busuk', 'Kurang Segar', 'Segar']
    pred_label = classes[pred_idx]
    
    is_uncertain = confidence < CONFIDENCE_THRESHOLD
    
    # Simpan ke riwayat pengujian
    st.session_state.history.append({
        'name': uploaded_file.name,
        'label': pred_label,
        'confidence': confidence,
        'uncertain': is_uncertain
    })
    
    # ── Tampilan Hasil Analisis ─────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<p class="section-title">📷 Foto Ikan Anda</p>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        
    with col2:
        st.markdown('<p class="section-title">📊 Probabilitas Kategori</p>', unsafe_allow_html=True)
        
        # Kelas Segar
        st.markdown(f"""
        <div class="score-container">
            <div class="score-header">
                <span class="score-label">🟢 Segar (Fresh)</span>
                <span class="score-value" style="color: #2ecc71;">{p_segar*100:.1f}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill fill-fresh" style="width: {p_segar*100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Kelas Kurang Segar
        st.markdown(f"""
        <div class="score-container">
            <div class="score-header">
                <span class="score-label">🟡 Kurang Segar (Moderate)</span>
                <span class="score-value" style="color: #f39c12;">{p_kurang*100:.1f}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill fill-moderate" style="width: {p_kurang*100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Kelas Busuk
        st.markdown(f"""
        <div class="score-container">
            <div class="score-header">
                <span class="score-label">🔴 Busuk (Rotten)</span>
                <span class="score-value" style="color: #e74c3c;">{p_busuk*100:.1f}%</span>
            </div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill fill-rotten" style="width: {p_busuk*100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Panel Kesimpulan Prediksi ────────────────────────────────────
    if is_uncertain:
        st.markdown(f"""
        <div class="result-uncertain">
            <h2>⚠️ PREDIKSI TIDAK PASTI</h2>
            <h3>Keyakinan Tertinggi: {confidence:.1f}% (Kecenderungan: {pred_label})</h3>
            <p>Model mendeteksi hasil dengan keraguan tinggi (di bawah {CONFIDENCE_THRESHOLD}%).<br>
            Sistem merekomendasikan untuk melakukan pemeriksaan fisik secara manual menggunakan panduan di bawah.</p>
        </div>
        """, unsafe_allow_html=True)
    elif pred_label == 'Segar':
        st.markdown(f"""
        <div class="result-fresh">
            <h2>✅ IKAN SEGAR</h2>
            <h3>Tingkat Keyakinan: {confidence:.1f}%</h3>
            <p>Ikan Bandeng ini memiliki kualitas fisik yang sangat prima, aman, dan layak dikonsumsi.</p>
        </div>
        """, unsafe_allow_html=True)
    elif pred_label == 'Kurang Segar':
        st.markdown(f"""
        <div class="result-moderate">
            <h2>🟡 IKAN KURANG SEGAR</h2>
            <h3>Tingkat Keyakinan: {confidence:.1f}%</h3>
            <p>Ikan Bandeng menunjukkan tanda-tanda awal penurunan kesegaran. Layak dikonsumsi dengan catatan segera diolah.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-stale">
            <h2>❌ IKAN BUSUK / TIDAK LAYAK</h2>
            <h3>Tingkat Keyakinan: {confidence:.1f}%</h3>
            <p>Ikan Bandeng ini sudah mengalami pembusukan lanjut. Sangat tidak disarankan untuk dikonsumsi demi kesehatan.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Panduan Verifikasi Manual Ikan Bandeng ───────────────────────
    with st.expander("🔍 Panduan Fisik Manual Kesegaran Ikan Bandeng"):
        st.markdown("""
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px;">
            <div style="flex: 1; min-width: 280px; background: rgba(46,204,113,0.05); border: 1px solid rgba(46,204,113,0.25); border-radius: 12px; padding: 20px;">
                <h4 style="color: #2ecc71; margin-top: 0; display: flex; align-items: center; gap: 8px;">🟢 Ciri Ikan Bandeng Segar</h4>
                <ul style="color: #b6f5d2; padding-left: 20px; margin: 0; line-height: 1.7; font-size: 0.95rem;">
                    <li><b>Mata:</b> Cembung, jernih, bersih, pupil hitam pekat menonjol.</li>
                    <li><b>Insang:</b> Berwarna merah cerah atau merah darah segar tanpa lendir keruh.</li>
                    <li><b>Sisik:</b> Melekat sangat kuat, berkilau keperakan khas bandeng, tidak mudah lepas.</li>
                    <li><b>Daging:</b> Sangat elastis, jika ditekan dengan jari akan kembali ke bentuk awal dengan cepat.</li>
                    <li><b>Aroma:</b> Berbau khas air payau segar, tidak menyengat.</li>
                </ul>
            </div>
            <div style="flex: 1; min-width: 280px; background: rgba(231,76,60,0.05); border: 1px solid rgba(231,76,60,0.25); border-radius: 12px; padding: 20px;">
                <h4 style="color: #e74c3c; margin-top: 0; display: flex; align-items: center; gap: 8px;">🔴 Ciri Ikan Bandeng Busuk</h4>
                <ul style="color: #fccac6; padding-left: 20px; margin: 0; line-height: 1.7; font-size: 0.95rem;">
                    <li><b>Mata:</b> Cekung, keruh kemerahan, tertutup selaput keputihan.</li>
                    <li><b>Insang:</b> Berwarna coklat gelap atau abu-abu pucat, berlendir tebal dan bau asam.</li>
                    <li><b>Sisik:</b> Kusam, rapuh, dan sangat mudah lepas/terkelupas saat dipegang.</li>
                    <li><b>Daging:</b> Lembek dan berair, meninggalkan bekas lekukan saat ditekan jari.</li>
                    <li><b>Aroma:</b> Berbau busuk asam, tengik, atau amis pekat yang menyengat.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Tampilan Awal Sebelum Upload (Animasi Hologram Scanner) ───────
    st.markdown("""
    <div style="text-align: center; background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.04); border-radius: 20px; padding: 40px; margin-top: 15px; backdrop-filter: blur(10px);">
        <div class="scanner-box">
            <div class="fish-icon">🐟</div>
            <div class="scanner-line"></div>
        </div>
        <p style="color: #aae0f2; font-size: 1.1rem; font-weight: 500; margin-top: 18px; letter-spacing: 0.3px;">
            📸 Unggah foto ikan Bandeng utuh (*full body*) untuk menganalisis kesegaran
        </p>
        <p style="color: rgba(170, 224, 242, 0.5); font-size: 0.85rem; margin-top: 5px;">
            Format yang didukung: JPG, JPEG, PNG
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Riwayat Pengujian Terkini ──────────────────────────────────────
if st.session_state.history:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">🕘 Riwayat Pengujian Terkini</p>', unsafe_allow_html=True)
    for h in reversed(st.session_state.history[-5:]):
        if h['uncertain']:
            icon = "⚠️"
            status = f"Tidak Pasti (Arah: {h['label']})"
            badge_color = "#bdc3c7"
        elif h['label'] == 'Segar':
            icon = "🟢"
            status = "Segar"
            badge_color = "#2ecc71"
        elif h['label'] == 'Kurang Segar':
            icon = "🟡"
            status = "Kurang Segar"
            badge_color = "#f39c12"
        else:
            icon = "🔴"
            status = "Busuk"
            badge_color = "#e74c3c"
            
        st.markdown(f"""
        <div class="card" style="padding:16px 20px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span>{icon}</span>
                <span style="font-weight: 600; color: #ffffff;">{h['name']}</span>
            </div>
            <div style="background: rgba(0,0,0,0.25); padding: 5px 12px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); font-size: 0.85rem;">
                Status: <span style="font-weight: 700; color: {badge_color};">{status}</span> ({h['confidence']:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🐟 FishCheck · Klasifikasi Kesegaran Ikan Bandeng (3 Kelas) · EfficientNetB0 Transfer Learning<br>
    Mata Kuliah Pengolahan Citra Digital · Teknik Informatika UMRAH · Tahun Ajaran 2025/2026
</div>
""", unsafe_allow_html=True)
