import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import json
import joblib
import hdbscan
from math import pi
from scipy.stats import mode
from sklearn.metrics.pairwise import euclidean_distances

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Zonasi Agroekologi Pamekasan",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konfigurasi Plotly Dark Theme
import plotly.io as pio
pio.templates.default = "plotly_dark"

# CSS untuk styling custom - Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles - Dark Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #0d1117;
        color: #c9d1d9;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1600px;
        background: #0d1117;
    }
    
    /* Modern Header - Dark Theme Gradient */
    .main-header {
        background: linear-gradient(135deg, #1a2332 0%, #2d3b4e 50%, #3a4d63 100%);
        color: #e6edf3;
        padding: 2.5rem 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        line-height: 1.2;
        color: #e6edf3;
    }
    
    .main-subtitle {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 400;
        line-height: 1.5;
        color: rgba(201,209,217,0.8);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e6edf3;
        margin: 2rem 0 1.25rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #30363d;
    }
    
    /* Info Box - Dark Theme */
    .info-box {
        background: #161b22;
        border-left: 3px solid #58a6ff;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.25rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        border: 1px solid #30363d;
    }
    
    .info-box h4 {
        color: #e6edf3;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .info-box p {
        color: #8b949e;
        line-height: 1.6;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Metric Cards */
    .metric-box {
        background: #161b22;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #30363d;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3);
        transition: all 0.2s ease;
    }
    
    .metric-box:hover {
        border-color: #58a6ff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    
    /* Zona Badges */
    .zona-badge {
        display: inline-block;
        padding: 0.4rem 0.875rem;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.85rem;
        margin: 0.25rem;
        background: #161b22;
        border: 1px solid #30363d;
        color: #c9d1d9;
        transition: all 0.2s;
    }
    
    .zona-badge:hover {
        border-color: #58a6ff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: #161b22;
        border-top: 1px solid #30363d;
        border-radius: 8px;
        text-align: center;
        color: #8b949e;
        font-size: 0.85rem;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #30363d;
        border-radius: 6px;
        margin: 0.5rem 0;
        background: #161b22;
    }
    
    /* Streamlit Components Styling */
    .stButton > button {
        background: #238636;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.625rem 1.5rem;
        font-weight: 500;
        font-size: 0.9rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #2ea043;
        box-shadow: 0 4px 12px rgba(35,134,54,0.4);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: #161b22;
        padding: 0.375rem;
        border-radius: 6px;
        border: 1px solid #30363d;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        font-size: 0.9rem;
        color: #8b949e;
        border: none;
        background: transparent;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #21262d;
        color: #c9d1d9;
    }
    
    .stTabs [aria-selected="true"] {
        background: #238636;
        color: white !important;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 600;
        color: #e6edf3;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 500;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #0d1117;
        padding-top: 1.5rem;
        border-right: 1px solid #21262d;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        display: none;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        gap: 0.5rem;
        padding: 0;
    }
    
    /* Hide radio button circles */
    [data-testid="stSidebar"] [data-baseweb="radio"] input[type="radio"] {
        display: none;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] > div {
        background: transparent;
        padding: 0.875rem 1.125rem;
        border-radius: 8px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 0.375rem;
        border: 1px solid transparent;
        color: #8b949e;
        cursor: pointer;
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: -0.01em;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] > div:hover {
        background: #161b22;
        border-color: #30363d;
        color: #e6edf3;
        transform: translateX(2px);
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        border-color: #238636;
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(35, 134, 54, 0.3);
        transform: scale(1.02);
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"]:hover {
        background: linear-gradient(135deg, #2ea043 0%, #238636 100%);
        transform: scale(1.02) translateX(2px);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-weight: 500;
        color: #c9d1d9;
        transition: all 0.2s;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #58a6ff;
        background: #21262d;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border: 1px solid #30363d;
        border-radius: 6px;
        overflow: hidden;
        background: #161b22;
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        border-radius: 6px;
        border: 1px solid #30363d;
        background: #161b22;
        color: #c9d1d9;
        transition: all 0.2s;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #58a6ff;
    }
    
    /* Card Container */
    .card {
        background: #161b22;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #30363d;
        box-shadow: 0 1px 2px rgba(0,0,0,0.3);
        margin-bottom: 1.25rem;
        transition: all 0.2s;
    }
    
    .card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        border-color: #58a6ff;
    }
    
    /* Additional Dark Theme Overrides */
    .stMarkdown, .stText {
        color: #c9d1d9;
    }
    
    /* Input fields */
    input, textarea, select {
        background: #0d1117 !important;
        color: #c9d1d9 !important;
        border-color: #30363d !important;
    }
    
    /* Code blocks */
    code {
        background: #161b22;
        color: #79c0ff;
        padding: 0.2em 0.4em;
        border-radius: 3px;
    }
    
    /* Alert/Message Boxes */
    .stAlert {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
    }
    
    [data-baseweb="notification"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    
    [data-baseweb="notification"] > div {
        color: #c9d1d9 !important;
    }
    
    /* Success box */
    .stSuccess, [data-baseweb="notification"][kind="success"] {
        background: #1c3d2b !important;
        border-color: #238636 !important;
    }
    
    /* Info box */
    .stInfo, [data-baseweb="notification"][kind="info"] {
        background: #1a2734 !important;
        border-color: #58a6ff !important;
    }
    
    /* Warning box */
    .stWarning, [data-baseweb="notification"][kind="warning"] {
        background: #3d3420 !important;
        border-color: #ffc107 !important;
    }
    
    /* Error box */
    .stError, [data-baseweb="notification"][kind="error"] {
        background: #3d1f1f !important;
        border-color: #e53e3e !important;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk load data
@st.cache_data
def load_data():
    """Load semua data yang diperlukan"""
    df_hasil = pd.read_csv('Hasil_Zonasi_Agroekologi_Pamekasan.csv')
    df_clean = pd.read_csv('Tanah-Iklim_2014-2024_clean.csv')
    
    with open('best_parameters.json', 'r') as f:
        best_params = json.load(f)
    
    return df_hasil, df_clean, best_params

@st.cache_resource
def load_model():
    """Load model dan scaler"""
    model = joblib.load('model_hdbscan_pamekasan.pkl')
    scaler = joblib.load('scaler_zonasi.pkl')
    return model, scaler

@st.cache_data
def load_tanaman_data():
    """Load data tanaman dari Dinas Pertanian"""
    try:
        df_tanaman = pd.read_csv('tanaman_madura_kondisi_tumbuh.csv')
        return df_tanaman
    except:
        return None

def cek_kesesuaian_tanaman(zona_char, tanaman_row):
    """Menghitung skor kesesuaian tanaman dengan zona (0-100)"""
    skor = 0
    max_skor = 0
    keterangan = []
    
    # 1. Cek pH (bobot 25)
    max_skor += 25
    pH_zona = zona_char['pH_mean']
    pH_range = tanaman_row['pH Tanah'].replace(',', '.').split('-')
    pH_min = float(pH_range[0].strip())
    pH_max = float(pH_range[1].strip())
    
    if pH_min <= pH_zona <= pH_max:
        skor += 25
        keterangan.append(f"pH sesuai ({pH_zona:.1f})")
    elif pH_zona < pH_min:
        gap = pH_min - pH_zona
        if gap <= 0.5:
            skor += 15
            keterangan.append(f"pH sedikit rendah (perlu kapur ringan)")
        else:
            skor += 5
            keterangan.append(f"pH terlalu rendah (perlu pengapuran)")
    else:
        keterangan.append(f"pH terlalu tinggi")
    
    # 2. Cek Nitrogen (bobot 20)
    max_skor += 20
    N_zona = zona_char['n_mean']
    N_min = int(tanaman_row['N (Nitrogen)'].replace('ppm', '').strip())
    
    if N_zona >= N_min:
        skor += 20
        keterangan.append(f"N cukup ({N_zona:.0f} ppm)")
    elif N_zona >= N_min * 0.8:
        skor += 15
        keterangan.append(f"N cukup dengan pemupukan ringan")
    else:
        skor += 5
        keterangan.append(f"N kurang (perlu pemupukan intensif)")
    
    # 3. Cek C-Organik (bobot 15)
    max_skor += 15
    C_zona = zona_char['c_org_mean']
    C_range = tanaman_row['C - Organik (%)'].replace('%', '').replace(',', '.').split('-')
    C_min = float(C_range[0].strip())
    C_max = float(C_range[1].strip())
    
    if C_min <= C_zona <= C_max:
        skor += 15
        keterangan.append(f"C-Organik optimal")
    elif C_zona < C_min:
        skor += 8
        keterangan.append(f"C-Organik kurang (perlu kompos)")
    else:
        skor += 12
        keterangan.append(f"C-Organik tinggi (baik)")
    
    # 4. Cek Curah Hujan (bobot 25)
    max_skor += 25
    CH_zona = zona_char['ch_tahunan'] / 12
    CH_min = int(tanaman_row['Curah Hujan (mm/bln)'])
    
    if CH_zona >= CH_min:
        skor += 25
        keterangan.append(f"Curah hujan cukup")
    elif CH_zona >= CH_min * 0.8:
        skor += 18
        keterangan.append(f"Curah hujan cukup (perlu irigasi suplemen)")
    else:
        skor += 8
        keterangan.append(f"Curah hujan kurang (perlu irigasi)")
    
    # 5. Cek Temperatur (bobot 15)
    max_skor += 15
    Temp_zona = zona_char['temp_mean']
    Temp_range = tanaman_row['Rata-Rata Temperatur (C)'].split('-')
    Temp_min = int(Temp_range[0].strip())
    Temp_max = int(Temp_range[1].strip())
    
    if Temp_min <= Temp_zona <= Temp_max:
        skor += 15
        keterangan.append(f"Temperatur sesuai")
    else:
        skor += 8
        keterangan.append(f"Temperatur marginal")
    
    skor_persen = (skor / max_skor) * 100
    return skor_persen, keterangan

def create_zona_profiles(df_hasil):
    """Buat profil zona berdasarkan karakteristik"""
    zona_profiles = {}
    
    for cluster_id in sorted(df_hasil[df_hasil['cluster'] != -1]['cluster'].unique()):
        cluster_data = df_hasil[df_hasil['cluster'] == cluster_id]
        
        # Hitung rata-rata
        ch_mean = cluster_data['Curah_Hujan_mm_per_bulan'].mean()
        pH_mean = cluster_data['pH'].mean()
        n_mean = cluster_data['N_ppm'].mean()
        c_org_mean = cluster_data['C_Organik_persen'].mean()
        temp_mean = cluster_data['Temp_C'].mean()
        elevasi_mean = cluster_data['Elevasi_m'].mean()
        
        # Kategorisasi
        ch_tahunan = ch_mean * 12
        if ch_tahunan < 1400:
            tipe_iklim = "Zona Kering"
        elif ch_tahunan < 1700:
            tipe_iklim = "Zona Sedang"
        else:
            tipe_iklim = "Zona Basah"
        
        if pH_mean < 5.5:
            ph_kategori = "sangat asam"
        elif pH_mean < 6.0:
            ph_kategori = "asam"
        elif pH_mean < 6.5:
            ph_kategori = "agak asam"
        else:
            ph_kategori = "netral"
        
        if n_mean > 28000:
            kesuburan = "tinggi"
        elif n_mean > 26500:
            kesuburan = "sedang-tinggi"
        elif n_mean > 25000:
            kesuburan = "sedang"
        else:
            kesuburan = "rendah"
        
        if elevasi_mean < 40:
            elevasi_kat = "dataran rendah"
        elif elevasi_mean < 100:
            elevasi_kat = "dataran sedang"
        else:
            elevasi_kat = "dataran tinggi"
        
        if temp_mean < 27.0:
            temp_kat = "sejuk"
        elif temp_mean < 27.5:
            temp_kat = "sedang"
        else:
            temp_kat = "hangat"
        
        zona_profiles[cluster_id] = {
            'zona': f'Zona {cluster_id + 1}',
            'profil': f'{tipe_iklim}, {elevasi_kat.title()}, pH {ph_kategori.title()}',
            'tipe_iklim': tipe_iklim,
            'elevasi_kat': elevasi_kat,
            'pH_kategori': ph_kategori,
            'kesuburan': kesuburan,
            'temp_kat': temp_kat,
            'jumlah_desa': len(cluster_data),
            'ch_tahunan': ch_tahunan,
            'pH_mean': pH_mean,
            'n_mean': n_mean,
            'c_org_mean': c_org_mean,
            'temp_mean': temp_mean,
            'elevasi_mean': elevasi_mean
        }
    
    return zona_profiles

def prediksi_zona_baru(model, scaler, pH, c_organik, n, p, k, curah_hujan, temp, penyinaran, elevasi):
    """Prediksi zona untuk desa baru"""
    try:
        # Siapkan input data sesuai urutan fitur saat training
        input_data = np.array([[pH, c_organik, n, p, k, curah_hujan, temp, penyinaran, elevasi]])
        
        # Normalisasi
        input_scaled = scaler.transform(input_data)
        
        # Coba gunakan approximate_predict jika model mendukung
        if hasattr(model, 'prediction_data_') and model.prediction_data_ is not None:
            zona, strength = hdbscan.approximate_predict(model, input_scaled)
            zona_id = int(zona[0])
            zona_label = 'Noise' if zona_id == -1 else f'Zona {zona_id + 1}'
            prediction_strength = float(strength[0])
            return zona_id, zona_label, prediction_strength
        else:
            # Model tidak punya prediction_data, gunakan k-NN
            raise AttributeError("Model tidak memiliki prediction_data_")
    
    except Exception as e:
        # Gunakan metode k-NN sebagai fallback
        try:
            # Load data training
            df_hasil = pd.read_csv('Hasil_Zonasi_Agroekologi_Pamekasan.csv')
            fitur_cols = ['pH', 'C_Organik_persen', 'N_ppm', 'P_ppm', 'K_ppm',
                         'Curah_Hujan_mm_per_bulan', 'Temp_C', 'Penyinaran_jam_per_hari', 'Elevasi_m']
            
            X_train = df_hasil[fitur_cols].values
            X_train_scaled = scaler.transform(X_train)
            
            # Hitung jarak euclidean ke semua titik training
            distances = euclidean_distances(input_scaled, X_train_scaled)[0]
            
            # Ambil 10 tetangga terdekat (bukan 5)
            k = 10
            nearest_indices = np.argsort(distances)[:k]
            nearest_zones = df_hasil.iloc[nearest_indices]['cluster'].values
            nearest_distances = distances[nearest_indices]
            
            # Filter hanya zona yang bukan noise (-1)
            valid_mask = nearest_zones != -1
            if np.sum(valid_mask) > 0:
                valid_zones = nearest_zones[valid_mask]
                valid_distances = nearest_distances[valid_mask]
                
                # Weighted voting berdasarkan inverse distance
                weights = 1 / (valid_distances + 1e-10)  # Tambah epsilon untuk avoid division by zero
                
                # Hitung weighted vote untuk setiap zona
                unique_zones = np.unique(valid_zones)
                zone_scores = {}
                for zone in unique_zones:
                    zone_mask = valid_zones == zone
                    zone_scores[zone] = np.sum(weights[zone_mask])
                
                # Pilih zona dengan skor tertinggi
                zona_id = int(max(zone_scores.items(), key=lambda x: x[1])[0])
                zona_label = f'Zona {zona_id + 1}'
                
                # Strength: proporsi weighted vote untuk zona terpilih
                total_weight = np.sum(weights)
                prediction_strength = float(zone_scores[zona_id] / total_weight)
            else:
                # Semua tetangga adalah noise
                zona_id = -1
                zona_label = 'Noise'
                prediction_strength = float(np.sum(nearest_zones == -1) / k)
            
            return zona_id, zona_label, prediction_strength
            
        except Exception as e2:
            st.error(f"Error saat prediksi: {str(e)}")
            st.error(f"Fallback prediction juga gagal: {str(e2)}")
            return -1, 'Error', 0.0

# Load data
try:
    df_hasil, df_clean, best_params = load_data()
    model, scaler = load_model()
    zona_profiles = create_zona_profiles(df_hasil)
    
    # Header dengan styling custom
    st.markdown("""
    <div class="main-header">
        <div class="main-title">Sistem Zonasi Agroekologi Pamekasan</div>
        <div class="main-subtitle">Platform analisis zonasi agroekologi menggunakan HDBSCAN Clustering berbasis data multi-temporal tanah, iklim, dan topografi periode 2014-2024</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 0 0 1.75rem 0; margin-bottom: 1.25rem; border-bottom: 1px solid #21262d;'>
        <h2 style='color: #e6edf3; margin: 0; font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em;'>Zonasi Agroekologi</h2>
        <p style='color: #6e7681; margin: 0.5rem 0 0 0; font-size: 0.85rem; font-weight: 500;'>Kabupaten Pamekasan</p>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.sidebar.radio(
        "Menu Navigasi",
        ["Beranda", "Peta Zona", "Analisis Zona", "Statistik", "Rekomendasi Tanaman", "Prediksi Zona", "Upload Data GEE"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("<div style='margin: 1.75rem 0;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style='background: #161b22; 
                padding: 1.25rem; border-radius: 10px; color: #c9d1d9;
                border: 1px solid #21262d; box-shadow: 0 1px 3px rgba(0,0,0,0.3);'>
        <h4 style='margin: 0 0 1rem 0; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #6e7681;'>Ringkasan Data</h4>
        <div style='margin: 0.625rem 0;'>
            <div style='font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; color: #58a6ff;'>{}</div>
            <div style='font-size: 0.8rem; opacity: 0.75; color: #8b949e;'>Total Desa</div>
        </div>
        <div style='margin: 0.625rem 0;'>
            <div style='font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; color: #58a6ff;'>{}</div>
            <div style='font-size: 0.8rem; opacity: 0.75; color: #8b949e;'>Zona Agroekologi</div>
        </div>
        <div style='margin: 0.625rem 0 0 0;'>
            <div style='font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; color: #58a6ff;'>2014-2024</div>
            <div style='font-size: 0.8rem; opacity: 0.75; color: #8b949e;'>Periode Data</div>
        </div>
    </div>
    """.format(len(df_hasil), best_params['jumlah_zona']), unsafe_allow_html=True)
    
    st.sidebar.markdown("<div style='margin: 1.75rem 0;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style='background: #161b22; padding: 1.25rem; border-radius: 10px; 
                border: 1px solid #21262d; box-shadow: 0 1px 3px rgba(0,0,0,0.3);'>
        <h4 style='color: #6e7681; margin: 0 0 0.75rem 0; font-size: 0.7rem; font-weight: 700; 
                   text-transform: uppercase; letter-spacing: 1px;'>Tentang</h4>
        <p style='color: #8b949e; margin: 0; font-size: 0.85rem; line-height: 1.6;'>
            Platform analisis menggunakan algoritma <strong style='color: #58a6ff;'>HDBSCAN</strong> untuk mengidentifikasi 
            zona agroekologi berdasarkan 9 parameter tanah, iklim, dan topografi.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Map menu ke halaman
    menu_map = {
        "Beranda": "Dashboard",
        "Peta Zona": "Peta Zona",
        "Analisis Zona": "Analisis Zona",
        "Statistik": "Statistik",
        "Rekomendasi Tanaman": "Rekomendasi Tanaman",
        "Prediksi Zona": "Prediksi Zona Baru",
        "Upload Data GEE": "Upload Data GEE"
    }
    
    menu = menu_map[menu]
    
    # ==================== DASHBOARD ====================
    if menu == "Dashboard":
        st.markdown('<h2 class="section-header">Ringkasan Hasil Zonasi</h2>', unsafe_allow_html=True)
        
        # Metrics dengan styling custom
        col1, col2, col3, col4 = st.columns(4)
        
        noise_count = len(df_hasil[df_hasil['cluster'] == -1])
        noise_persen = (noise_count / len(df_hasil)) * 100
        
        metrics_data = [
            ("Total Desa Teranalisis", len(df_hasil), "🏘️"),
            ("Zona Terbentuk", best_params['jumlah_zona'], "🗺️"),
            ("Outlier Terdeteksi", f"{noise_count} ({noise_persen:.1f}%)", "⚠️"),
            ("Skor Silhouette", f"{best_params['silhouette_score']:.3f}" if best_params['silhouette_score'] else "-", "📊")
        ]
        
        for col, (label, value, icon) in zip([col1, col2, col3, col4], metrics_data):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #e6edf3;">{value}</div>
                    <div style="font-size: 0.9rem; color: #8b949e; margin-top: 0.3rem;">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Distribusi Zona
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown("#### Sebaran Desa per Zona Agroekologi")
            zona_dist = df_hasil['zona_agroekologi'].value_counts().reset_index()
            zona_dist.columns = ['Zona', 'Jumlah Desa']
            
            fig = px.bar(zona_dist, x='Zona', y='Jumlah Desa',
                        color='Jumlah Desa',
                        color_continuous_scale='Blues',
                        text='Jumlah Desa',
                        labels={'Jumlah Desa': 'Jumlah Desa'})
            fig.update_traces(textposition='outside', textfont_size=12)
            fig.update_layout(
                height=400, 
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Komposisi Zona")
            colors = px.colors.sequential.Blues_r
            fig = px.pie(zona_dist, values='Jumlah Desa', names='Zona',
                        color_discrete_sequence=colors,
                        hole=0.4)
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=11)
            fig.update_layout(
                height=400,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Profil Zona
        st.markdown('<h2 class="section-header">Karakteristik Zona Agroekologi</h2>', unsafe_allow_html=True)
        
        for cluster_id, profile in zona_profiles.items():
            with st.expander(f"**{profile['zona']}** — {profile['tipe_iklim']} • {profile['jumlah_desa']} desa"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**☁️ Kondisi Iklim**")
                    st.write(f"• Klasifikasi: {profile['tipe_iklim']}")
                    st.write(f"• CH Tahunan: {profile['ch_tahunan']:.0f} mm")
                    st.write(f"• Suhu Rata-rata: {profile['temp_mean']:.1f}°C ({profile['temp_kat']})")
                
                with col2:
                    st.markdown("**🌱 Sifat Tanah**")
                    st.write(f"• pH: {profile['pH_mean']:.2f} ({profile['pH_kategori']})")
                    st.write(f"• Nitrogen: {profile['n_mean']:.0f} ppm")
                    st.write(f"• Tingkat Kesuburan: {profile['kesuburan'].title()}")
                
                with col3:
                    st.markdown("**🏔️ Topografi**")
                    st.write(f"• Ketinggian: {profile['elevasi_mean']:.0f} mdpl")
                    st.write(f"• Kategori: {profile['elevasi_kat'].title()}")
                    zona_desa = df_hasil[df_hasil['cluster'] == cluster_id]
                    kecamatan = zona_desa['kecamatan'].nunique()
                    st.write(f"• Tersebar di {kecamatan} kecamatan")
        
        # Metrik Evaluasi
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Evaluasi Kualitas Clustering</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        metrics_eval = [
            ("Silhouette Coefficient", best_params.get('silhouette_score'), 
             "Mengukur seberapa mirip data dalam cluster dan berbeda dengan cluster lain",
             lambda x: "Sangat Baik" if x > 0.7 else "Baik" if x > 0.5 else "Cukup" if x > 0.25 else "Kurang"),
            ("Davies-Bouldin Index", best_params.get('davies_bouldin_score'),
             "Mengukur rasio scatter dalam cluster vs jarak antar cluster (semakin rendah semakin baik)",
             lambda x: f"Score: {x:.4f}"),
            ("Calinski-Harabasz Score", best_params.get('calinski_harabasz_score'),
             "Mengukur rasio between-cluster vs within-cluster dispersion (semakin tinggi semakin baik)",
             lambda x: f"Score: {x:.2f}")
        ]
        
        for col, (name, score, desc, eval_func) in zip([col1, col2, col3], metrics_eval):
            with col:
                st.markdown(f"""
                <div style='background: #161b22; padding: 1.5rem; border-radius: 10px; 
                            border: 1px solid #30363d; height: 180px;'>
                    <h4 style='margin: 0 0 0.5rem 0; color: #e6edf3;'>{name}</h4>
                    <p style='font-size: 0.85rem; color: #8b949e; margin-bottom: 1rem;'>{desc}</p>
                """, unsafe_allow_html=True)
                
                if score:
                    if name == "Silhouette Coefficient":
                        result = eval_func(score)
                        st.metric(label="Nilai", value=f"{score:.4f}", delta=result)
                    else:
                        st.metric(label="Nilai", value=eval_func(score))
                else:
                    st.info("Tidak tersedia")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # ==================== PETA ZONA ====================
    elif menu == "Peta Zona":
        st.markdown('<h2 class="section-header">Peta Distribusi Zona Agroekologi</h2>', unsafe_allow_html=True)
        
        # Tambahkan profil ke dataframe
        df_map = df_hasil.copy()
        df_map['Desa_Info'] = df_map['desa'] + ', Kec. ' + df_map['kecamatan']
        
        # Set default untuk noise dengan nilai aktual
        df_map['Tipe_Iklim'] = '-'
        df_map['Elevasi_Kat'] = '-'
        df_map['pH_Kategori'] = '-'
        df_map['Kesuburan'] = '-'
        df_map['Temp_Kat'] = '-'
        
        # Isi profil untuk desa yang masuk zona
        for cluster_id, profile in zona_profiles.items():
            mask = df_map['cluster'] == cluster_id
            df_map.loc[mask, 'Tipe_Iklim'] = profile['tipe_iklim']
            df_map.loc[mask, 'Elevasi_Kat'] = profile['elevasi_kat']
            df_map.loc[mask, 'pH_Kategori'] = profile['pH_kategori']
            df_map.loc[mask, 'Kesuburan'] = profile['kesuburan']
            df_map.loc[mask, 'Temp_Kat'] = profile['temp_kat']
        
        # Untuk noise, isi dengan kategori berdasarkan nilai aktual
        noise_mask = df_map['cluster'] == -1
        if noise_mask.any():
            for idx in df_map[noise_mask].index:
                row = df_map.loc[idx]
                
                # Kategorisasi iklim
                ch_tahunan = row['Curah_Hujan_mm_per_bulan'] * 12
                if ch_tahunan < 1400:
                    df_map.loc[idx, 'Tipe_Iklim'] = "Zona Kering"
                elif ch_tahunan < 1700:
                    df_map.loc[idx, 'Tipe_Iklim'] = "Zona Sedang"
                else:
                    df_map.loc[idx, 'Tipe_Iklim'] = "Zona Basah"
                
                # Kategorisasi elevasi
                if row['Elevasi_m'] < 40:
                    df_map.loc[idx, 'Elevasi_Kat'] = "dataran rendah"
                elif row['Elevasi_m'] < 100:
                    df_map.loc[idx, 'Elevasi_Kat'] = "dataran sedang"
                else:
                    df_map.loc[idx, 'Elevasi_Kat'] = "dataran tinggi"
                
                # Kategorisasi pH
                if row['pH'] < 5.5:
                    df_map.loc[idx, 'pH_Kategori'] = "sangat asam"
                elif row['pH'] < 6.0:
                    df_map.loc[idx, 'pH_Kategori'] = "asam"
                elif row['pH'] < 6.5:
                    df_map.loc[idx, 'pH_Kategori'] = "agak asam"
                else:
                    df_map.loc[idx, 'pH_Kategori'] = "netral"
                
                # Kategorisasi kesuburan (N)
                if row['N_ppm'] > 28000:
                    df_map.loc[idx, 'Kesuburan'] = "tinggi"
                elif row['N_ppm'] > 26500:
                    df_map.loc[idx, 'Kesuburan'] = "sedang-tinggi"
                elif row['N_ppm'] > 25000:
                    df_map.loc[idx, 'Kesuburan'] = "sedang"
                else:
                    df_map.loc[idx, 'Kesuburan'] = "rendah"
                
                # Kategorisasi temperatur
                if row['Temp_C'] < 27.0:
                    df_map.loc[idx, 'Temp_Kat'] = "sejuk"
                elif row['Temp_C'] < 27.5:
                    df_map.loc[idx, 'Temp_Kat'] = "sedang"
                else:
                    df_map.loc[idx, 'Temp_Kat'] = "hangat"
        
        # Filter zona dan pencarian desa
        zona_options = ['Semua Zona'] + sorted(df_map['zona_agroekologi'].unique().tolist())
        desa_options = ['Semua Desa'] + sorted(df_map['desa'].unique().tolist())
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            selected_zona = st.selectbox("Filter berdasarkan zona:", zona_options)
        with col2:
            selected_desa = st.selectbox("🔍 Cari desa:", desa_options, help="Ketik nama desa untuk mencari")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            show_noise = st.checkbox("Tampilkan Outlier", value=True)
        
        # Terapkan filter zona
        if selected_zona != 'Semua Zona':
            df_map_filtered = df_map[df_map['zona_agroekologi'] == selected_zona]
        else:
            df_map_filtered = df_map
        
        # Terapkan filter desa
        if selected_desa != 'Semua Desa':
            df_map_filtered = df_map_filtered[df_map_filtered['desa'] == selected_desa]
            # Jika ada desa spesifik, tampilkan info detail
            if len(df_map_filtered) > 0:
                desa_info = df_map_filtered.iloc[0]
                st.info(f"**{desa_info['desa']}**, Kec. {desa_info['kecamatan']} — {desa_info['zona_agroekologi']} | "
                       f"pH: {desa_info['pH']:.2f}, N: {desa_info['N_ppm']:.0f} ppm, "
                       f"CH: {desa_info['Curah_Hujan_mm_per_bulan']:.1f} mm/bln, "
                       f"Elevasi: {desa_info['Elevasi_m']:.0f} mdpl")
        
        if not show_noise:
            df_map_filtered = df_map_filtered[df_map_filtered['cluster'] != -1]
        
        # Peta Interaktif
        fig = px.scatter_mapbox(
            df_map_filtered,
            lat='lat',
            lon='lon',
            color='zona_agroekologi',
            hover_name='Desa_Info',
            hover_data={
                'zona_agroekologi': True,
                'Tipe_Iklim': True,
                'Elevasi_Kat': True,
                'pH_Kategori': True,
                'Kesuburan': True,
                'Temp_Kat': True,
                'pH': ':.2f',
                'N_ppm': ':.0f',
                'Curah_Hujan_mm_per_bulan': ':.1f',
                'Temp_C': ':.2f',
                'Elevasi_m': ':.0f',
                'lat': False,
                'lon': False
            },
            zoom=9.5,
            height=650,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabel data
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Detail Data Desa")
        
        show_columns = ['kecamatan', 'desa', 'zona_agroekologi', 'pH', 'N_ppm', 'P_ppm', 'K_ppm',
                       'Curah_Hujan_mm_per_bulan', 'Temp_C', 'Elevasi_m']
        
        st.dataframe(
            df_map_filtered[show_columns].style.format({
                'pH': '{:.2f}',
                'N_ppm': '{:.0f}',
                'P_ppm': '{:.0f}',
                'K_ppm': '{:.0f}',
                'Curah_Hujan_mm_per_bulan': '{:.1f}',
                'Temp_C': '{:.2f}',
                'Elevasi_m': '{:.0f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Download data
        csv = df_map_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data CSV",
            data=csv,
            file_name=f'zonasi_{selected_zona.lower().replace(" ", "_")}.csv',
            mime='text/csv',
        )
    
    # ==================== ANALISIS ZONA ====================
    elif menu == "Analisis Zona":
        st.markdown('<h2 class="section-header">Perbandingan Karakteristik Antar Zona</h2>', unsafe_allow_html=True)
        
        # Tambahkan pencarian desa
        st.markdown("### 🔍 Analisis Desa Spesifik")
        desa_list = ['Pilih untuk analisis desa spesifik'] + sorted(df_hasil['desa'].unique().tolist())
        selected_desa_analisis = st.selectbox("Cari dan analisis desa tertentu:", desa_list, 
                                              help="Pilih desa untuk melihat detail karakteristik")
        
        if selected_desa_analisis != 'Pilih untuk analisis desa spesifik':
            desa_data = df_hasil[df_hasil['desa'] == selected_desa_analisis].iloc[0]
            
            st.markdown(f"#### Detail Desa: {desa_data['desa']}, Kec. {desa_data['kecamatan']}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Zona", desa_data['zona_agroekologi'])
            with col2:
                st.metric("pH Tanah", f"{desa_data['pH']:.2f}")
            with col3:
                st.metric("Nitrogen", f"{desa_data['N_ppm']:.0f} ppm")
            with col4:
                st.metric("Elevasi", f"{desa_data['Elevasi_m']:.0f} mdpl")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Curah Hujan", f"{desa_data['Curah_Hujan_mm_per_bulan']:.1f} mm/bln")
            with col2:
                st.metric("Temperatur", f"{desa_data['Temp_C']:.1f}°C")
            with col3:
                st.metric("Fosfor (P)", f"{desa_data['P_ppm']:.0f} ppm")
            with col4:
                st.metric("Kalium (K)", f"{desa_data['K_ppm']:.0f} ppm")
            
            # Tambahkan perbandingan dengan zona
            if desa_data['cluster'] != -1:
                zona_id = desa_data['cluster']
                zona_profile = zona_profiles[zona_id]
                st.markdown(f"**Perbandingan dengan {zona_profile['zona']}:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"• pH desa: {desa_data['pH']:.2f} vs rata-rata zona: {zona_profile['pH_mean']:.2f}")
                    st.write(f"• N desa: {desa_data['N_ppm']:.0f} ppm vs rata-rata zona: {zona_profile['n_mean']:.0f} ppm")
                with col2:
                    st.write(f"• CH desa: {desa_data['Curah_Hujan_mm_per_bulan']:.1f} mm/bln vs rata-rata zona: {zona_profile['ch_tahunan']/12:.1f} mm/bln")
                    st.write(f"• Temp desa: {desa_data['Temp_C']:.1f}°C vs rata-rata zona: {zona_profile['temp_mean']:.1f}°C")
            
            st.markdown("<hr>", unsafe_allow_html=True)
        
        # Filter data (tanpa noise)
        df_plot = df_hasil[df_hasil['cluster'] != -1].copy()
        
        # Pilih zona untuk perbandingan
        st.markdown("### Perbandingan Antar Zona")
        st.markdown("Visualisasi distribusi dan perbandingan fitur-fitur agroekologi di setiap zona untuk memahami perbedaan karakteristiknya.")
        st.markdown("<br>", unsafe_allow_html=True)
        
        fitur_options = {
            'pH': 'pH',
            'C_Organik_persen': 'C Organik (%)',
            'N_ppm': 'Nitrogen (ppm)',
            'P_ppm': 'Fosfor (ppm)',
            'K_ppm': 'Kalium (ppm)',
            'Curah_Hujan_mm_per_bulan': 'Curah Hujan (mm/bulan)',
            'Temp_C': 'Temperatur (°C)',
            'Penyinaran_jam_per_hari': 'Penyinaran (jam/hari)',
            'Elevasi_m': 'Elevasi (m)'
        }
        
        # Box plot per fitur
        tab1, tab2, tab3 = st.tabs(["Distribusi Box Plot", "Violin Plot", "Profil Radar"])
        
        with tab1:
            st.markdown("##### Pilih fitur untuk dibandingkan:")
            selected_fitur = st.multiselect(
                "Pilih Fitur untuk Ditampilkan:",
                options=list(fitur_options.keys()),
                default=['pH', 'N_ppm', 'Curah_Hujan_mm_per_bulan'],
                format_func=lambda x: fitur_options[x]
            )
            
            if selected_fitur:
                n_fitur = len(selected_fitur)
                n_cols = 3
                n_rows = (n_fitur + n_cols - 1) // n_cols
                
                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=[fitur_options[f] for f in selected_fitur]
                )
                
                for idx, fitur in enumerate(selected_fitur):
                    row = idx // n_cols + 1
                    col = idx % n_cols + 1
                    
                    for zona in sorted(df_plot['zona_agroekologi'].unique()):
                        zona_data = df_plot[df_plot['zona_agroekologi'] == zona]
                        fig.add_trace(
                            go.Box(y=zona_data[fitur], name=zona, showlegend=(idx == 0)),
                            row=row, col=col
                        )
                
                fig.update_layout(height=300 * n_rows, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            selected_fitur_violin = st.selectbox(
                "Pilih Fitur:",
                options=list(fitur_options.keys()),
                format_func=lambda x: fitur_options[x]
            )
            
            fig = px.violin(df_plot, y=selected_fitur_violin, x='zona_agroekologi',
                           color='zona_agroekologi', box=True, points='all',
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(
                height=500,
                xaxis_title='Zona',
                yaxis_title=fitur_options[selected_fitur_violin],
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("##### Perbandingan Profil Multi-Dimensi")
            st.caption("Semua nilai dinormalisasi ke skala 0-1 untuk perbandingan yang adil")
            
            # Normalisasi data untuk radar chart
            fitur_clustering = ['pH', 'C_Organik_persen', 'N_ppm', 'P_ppm', 'K_ppm',
                              'Curah_Hujan_mm_per_bulan', 'Temp_C', 'Penyinaran_jam_per_hari', 'Elevasi_m']
            
            cluster_means = df_plot.groupby('zona_agroekologi')[fitur_clustering].mean()
            cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
            
            categories = ['pH', 'C Organik', 'N', 'P', 'K', 'Curah Hujan', 'Temp', 'Penyinaran', 'Elevasi']
            
            fig = go.Figure()
            
            for zona in cluster_means_norm.index:
                fig.add_trace(go.Scatterpolar(
                    r=cluster_means_norm.loc[zona].values.tolist(),
                    theta=categories,
                    fill='toself',
                    name=zona
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabel ringkasan statistik
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("#### Statistik Ringkasan per Zona")
        st.caption("Nilai rata-rata untuk setiap parameter di masing-masing zona")
        
        summary_stats = df_plot.groupby('zona_agroekologi')[fitur_clustering].mean()
        
        st.dataframe(
            summary_stats.style.format({
                'pH': '{:.2f}',
                'C_Organik_persen': '{:.2f}',
                'N_ppm': '{:.0f}',
                'P_ppm': '{:.0f}',
                'K_ppm': '{:.0f}',
                'Curah_Hujan_mm_per_bulan': '{:.1f}',
                'Temp_C': '{:.2f}',
                'Penyinaran_jam_per_hari': '{:.2f}',
                'Elevasi_m': '{:.0f}'
            }).background_gradient(cmap='Greens'),
            use_container_width=True
        )
    
    # ==================== STATISTIK ====================
    elif menu == "Statistik":
        st.markdown('<h2 class="section-header">Analisis Statistik Detail</h2>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Distribusi Fitur", "Analisis Korelasi"])
        
        with tab1:
            st.markdown("#### Distribusi Fitur per Zona")
            
            fitur_selected = st.selectbox(
                "Pilih Fitur:",
                options=['pH', 'C_Organik_persen', 'N_ppm', 'P_ppm', 'K_ppm',
                        'Curah_Hujan_mm_per_bulan', 'Temp_C', 'Penyinaran_jam_per_hari', 'Elevasi_m'],
                format_func=lambda x: {
                    'pH': 'pH',
                    'C_Organik_persen': 'C Organik (%)',
                    'N_ppm': 'Nitrogen (ppm)',
                    'P_ppm': 'Fosfor (ppm)',
                    'K_ppm': 'Kalium (ppm)',
                    'Curah_Hujan_mm_per_bulan': 'Curah Hujan (mm/bulan)',
                    'Temp_C': 'Temperatur (°C)',
                    'Penyinaran_jam_per_hari': 'Penyinaran (jam/hari)',
                    'Elevasi_m': 'Elevasi (m)'
                }[x]
            )
            
            # Histogram dengan KDE
            fig = px.histogram(
                df_hasil[df_hasil['cluster'] != -1],
                x=fitur_selected,
                color='zona_agroekologi',
                marginal='box',
                nbins=30,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistik deskriptif
            st.markdown("#### Statistik Deskriptif")
            stats = df_hasil[df_hasil['cluster'] != -1].groupby('zona_agroekologi')[fitur_selected].describe()
            st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
        
        with tab2:
            st.markdown("#### Matriks Korelasi Fitur")
            
            fitur_clustering = ['pH', 'C_Organik_persen', 'N_ppm', 'P_ppm', 'K_ppm',
                              'Curah_Hujan_mm_per_bulan', 'Temp_C', 'Penyinaran_jam_per_hari', 'Elevasi_m']
            
            corr_matrix = df_hasil[fitur_clustering].corr()
            
            # Heatmap dengan nilai korelasi
            labels = ['pH', 'C Organik', 'N', 'P', 'K', 'CH', 'Temp', 'Penyinaran', 'Elevasi']
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Korelasi"),
                x=labels,
                y=labels,
                color_continuous_scale='RdBu_r',
                aspect='auto',
                zmin=-1, zmax=1,
                text_auto='.3f'  # Tampilkan nilai dengan 3 desimal
            )
            fig.update_traces(
                textfont=dict(size=10),
                texttemplate='%{text}'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Korelasi tinggi
            st.markdown("#### Korelasi Tinggi (|r| > 0.5)")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        high_corr.append({
                            'Fitur 1': corr_matrix.columns[i],
                            'Fitur 2': corr_matrix.columns[j],
                            'Korelasi': corr_matrix.iloc[i, j]
                        })
            
            if high_corr:
                df_high_corr = pd.DataFrame(high_corr).sort_values('Korelasi', key=abs, ascending=False)
                st.dataframe(df_high_corr.style.format({'Korelasi': '{:.3f}'}), use_container_width=True)
            else:
                st.info("Tidak ada korelasi tinggi (|r| > 0.5)")
    
    # ==================== REKOMENDASI TANAMAN ====================
    elif menu == "Rekomendasi Tanaman":
        st.markdown('<h2 class="section-header">Rekomendasi Komoditas Tanaman Pangan</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4 style='margin: 0 0 0.5rem 0;'>Sumber Data</h4>
            <p style='margin: 0; font-size: 0.95rem;'>
            Rekomendasi tanaman berdasarkan data syarat tumbuh tanaman pangan Madura dari <b>Dinas Pertanian Kabupaten Pamekasan</b>. 
            Sistem menghitung kesesuaian setiap tanaman dengan karakteristik zona menggunakan metode scoring multi-kriteria.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load data tanaman
        df_tanaman = load_tanaman_data()
        
        if df_tanaman is None:
            st.error("File tanaman_madura_kondisi_tumbuh.csv tidak ditemukan. Pastikan file ada di folder yang sama dengan aplikasi.")
            st.stop()
        
        # Tambahkan pencarian desa untuk rekomendasi
        st.markdown("### 🔍 Rekomendasi untuk Desa Spesifik")
        desa_rekomendasi = ['Pilih desa untuk rekomendasi'] + sorted(df_hasil['desa'].unique().tolist())
        selected_desa_rek = st.selectbox("Cari desa:", desa_rekomendasi, 
                                         help="Pilih desa untuk mendapatkan rekomendasi tanaman yang sesuai")
        
        if selected_desa_rek != 'Pilih desa untuk rekomendasi':
            desa_rek_data = df_hasil[df_hasil['desa'] == selected_desa_rek].iloc[0]
            
            st.success(f"**{desa_rek_data['desa']}**, Kec. {desa_rek_data['kecamatan']} — {desa_rek_data['zona_agroekologi']}")
            
            if desa_rek_data['cluster'] != -1:
                zona_id = desa_rek_data['cluster']
                zona_char = zona_profiles[zona_id]
                
                # Hitung kesesuaian untuk desa ini
                rekomendasi_desa = []
                for _, tanaman in df_tanaman.iterrows():
                    skor, keterangan = cek_kesesuaian_tanaman(zona_char, tanaman)
                    rekomendasi_desa.append({
                        'Tanaman': tanaman['Tanaman'],
                        'Skor Kesesuaian': skor,
                        'Status': 'Sangat Sesuai' if skor >= 80 else 'Sesuai' if skor >= 60 else 'Cukup Sesuai' if skor >= 40 else 'Kurang Sesuai',
                        'Keterangan': ', '.join(keterangan)
                    })
                
                df_rek_desa = pd.DataFrame(rekomendasi_desa).sort_values('Skor Kesesuaian', ascending=False)
                
                # Tampilkan top 3 rekomendasi
                st.markdown("**Top 3 Tanaman yang Direkomendasikan:**")
                for idx, row in df_rek_desa.head(3).iterrows():
                    status_color = "🟢" if row['Skor Kesesuaian'] >= 80 else "🟡" if row['Skor Kesesuaian'] >= 60 else "🟠"
                    st.markdown(f"{status_color} **{row['Tanaman']}** — Skor: {row['Skor Kesesuaian']:.1f}% ({row['Status']})")
                    st.caption(row['Keterangan'])
                
                # Tabel lengkap
                with st.expander("Lihat semua rekomendasi"):
                    st.dataframe(df_rek_desa, use_container_width=True, hide_index=True)
            else:
                st.warning("Desa ini termasuk dalam kategori outlier. Rekomendasi mungkin kurang akurat. Konsultasikan dengan ahli pertanian setempat.")
            
            st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("### Rekomendasi per Zona Agroekologi")
        
        # Tampilkan data tanaman
        with st.expander("Lihat Data Syarat Tumbuh Tanaman Pangan Madura", expanded=False):
            st.dataframe(df_tanaman, use_container_width=True, height=300)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Pilih zona untuk analisis
            zona_selected = st.selectbox(
                "Pilih Zona untuk Analisis Rekomendasi:",
                options=[0, 1, 2],
                format_func=lambda x: f"Zona {x+1}"
            )
            
            if zona_selected in zona_profiles:
                profile = zona_profiles[zona_selected]
                cluster_data = df_hasil[df_hasil['cluster'] == zona_selected]
                
                # Karakteristik zona
                zona_char = {
                    'pH_mean': profile['pH_mean'],
                    'n_mean': profile['n_mean'],
                    'c_org_mean': cluster_data['C_Organik_persen'].mean(),
                    'ch_tahunan': profile['ch_tahunan'],
                    'temp_mean': profile['temp_mean']
                }
                
                # Header zona
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;'>
                    <h3 style='margin: 0;'>Zona {zona_selected + 1}: {profile['profil']}</h3>
                    <p style='margin: 0.5rem 0 0 0; font-size: 0.95rem;'>
                    {profile['jumlah_desa']} desa | {profile['tipe_iklim']} | {profile['elevasi_kat'].title()}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Karakteristik zona
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("pH Tanah", f"{zona_char['pH_mean']:.2f}", 
                             f"{profile['pH_kategori'].title()}")
                    st.metric("Nitrogen (N)", f"{zona_char['n_mean']:.0f} ppm",
                             f"{profile['kesuburan'].title()}")
                with col2:
                    st.metric("C-Organik", f"{zona_char['c_org_mean']:.2f}%")
                    st.metric("Curah Hujan", f"{profile['ch_tahunan']:.0f} mm/thn",
                             f"{zona_char['ch_tahunan']/12:.0f} mm/bln")
                with col3:
                    st.metric("Temperatur", f"{zona_char['temp_mean']:.2f}°C",
                             f"{profile['temp_kat'].title()}")
                    st.metric("Elevasi", f"{profile['elevasi_mean']:.0f} m",
                             f"{profile['elevasi_kat'].title()}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Evaluasi setiap tanaman
                hasil_evaluasi = []
                
                for idx, tanaman in df_tanaman.iterrows():
                    skor, keterangan = cek_kesesuaian_tanaman(zona_char, tanaman)
                    hasil_evaluasi.append({
                        'Tanaman': tanaman['Tanaman Pangan di Madura'],
                        'Skor': skor,
                        'Kategori': 'Sangat Sesuai' if skor >= 80 else 'Sesuai' if skor >= 60 else 'Cukup Sesuai' if skor >= 40 else 'Kurang Sesuai',
                        'Keterangan': ', '.join(keterangan)
                    })
                
                df_evaluasi = pd.DataFrame(hasil_evaluasi).sort_values('Skor', ascending=False)
                
                # Kategorisasi
                tanaman_prioritas = df_evaluasi[df_evaluasi['Skor'] >= 80]
                tanaman_sesuai = df_evaluasi[(df_evaluasi['Skor'] >= 60) & (df_evaluasi['Skor'] < 80)]
                tanaman_alternatif = df_evaluasi[(df_evaluasi['Skor'] >= 40) & (df_evaluasi['Skor'] < 60)]
                
                # Visualisasi skor
                st.markdown("#### Skor Kesesuaian Tanaman")
                fig = px.bar(
                    df_evaluasi,
                    x='Skor',
                    y='Tanaman',
                    orientation='h',
                    color='Skor',
                    color_continuous_scale=['#ff4444', '#ffbb33', '#00C851', '#007E33'],
                    range_color=[0, 100],
                    labels={'Skor': 'Skor Kesesuaian (%)'}
                )
                fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Rekomendasi berdasarkan kategori
                st.markdown("#### 🌟 Rekomendasi Tanaman")
                
                tab1, tab2, tab3 = st.tabs(["🌟 Prioritas Utama", "✓ Sesuai", "⚠ Alternatif"])
                
                with tab1:
                    if len(tanaman_prioritas) > 0:
                        st.success(f"**{len(tanaman_prioritas)} tanaman** dengan skor ≥ 80% (Sangat Sesuai)")
                        for idx, row in tanaman_prioritas.iterrows():
                            with st.container():
                                st.markdown(f"""
                                <div style='background: #1c3d2b; border-left: 4px solid #238636; 
                                            padding: 1rem; margin: 0.5rem 0; border-radius: 6px;'>
                                    <h4 style='margin: 0; color: #4caf50;'>
                                        {row['Tanaman']} <span style='float: right; background: #238636; 
                                        color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.9rem;'>
                                        {row['Skor']:.0f}%</span>
                                    </h4>
                                    <p style='margin: 0.5rem 0 0 0; color: #4caf50; font-size: 0.9rem;'>
                                        ✓ {row['Keterangan']}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("Tidak ada tanaman dengan kategori Sangat Sesuai untuk zona ini.")
                
                with tab2:
                    if len(tanaman_sesuai) > 0:
                        st.info(f"**{len(tanaman_sesuai)} tanaman** dengan skor 60-79% (Sesuai dengan Penyesuaian Kecil)")
                        for idx, row in tanaman_sesuai.iterrows():
                            with st.container():
                                st.markdown(f"""
                                <div style='background: #1a2f3a; border-left: 4px solid #58a6ff; 
                                            padding: 1rem; margin: 0.5rem 0; border-radius: 6px;'>
                                    <h4 style='margin: 0; color: #58a6ff;'>
                                        {row['Tanaman']} <span style='float: right; background: #58a6ff; 
                                        color: #0d1117; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.9rem;'>
                                        {row['Skor']:.0f}%</span>
                                    </h4>
                                    <p style='margin: 0.5rem 0 0 0; color: #58a6ff; font-size: 0.9rem;'>
                                        ℹ {row['Keterangan']}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("Tidak ada tanaman dengan kategori Sesuai untuk zona ini.")
                
                with tab3:
                    if len(tanaman_alternatif) > 0:
                        st.warning(f"**{len(tanaman_alternatif)} tanaman** dengan skor 40-59% (Perlu Input Tambahan)")
                        for idx, row in tanaman_alternatif.iterrows():
                            with st.container():
                                st.markdown(f"""
                                <div style='background: #3d3420; border-left: 4px solid #ffc107; 
                                            padding: 1rem; margin: 0.5rem 0; border-radius: 6px;'>
                                    <h4 style='margin: 0; color: #ffc107;'>
                                        {row['Tanaman']} <span style='float: right; background: #ffc107; 
                                        color: #0d1117; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.9rem;'>
                                        {row['Skor']:.0f}%</span>
                                    </h4>
                                    <p style='margin: 0.5rem 0 0 0; color: #ffc107; font-size: 0.9rem;'>
                                        ⚠ {row['Keterangan']}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("Tidak ada tanaman dengan kategori Alternatif untuk zona ini.")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Strategi pengelolaan
                st.markdown("#### 🔧 Strategi Pengelolaan Rekomendasi")
                
                strategi = []
                
                if zona_char['pH_mean'] < 5.5:
                    strategi.append("🔴 **PRIORITAS**: Pengapuran dengan kapur pertanian (Dolomit) 1-2 ton/ha")
                elif zona_char['pH_mean'] < 6.0:
                    strategi.append("🟡 Pengapuran ringan 0.5-1 ton/ha untuk tanaman sensitif pH")
                
                if zona_char['n_mean'] < 25000:
                    strategi.append("🔴 **PRIORITAS**: Pemupukan Nitrogen intensif - Urea 250-300 kg/ha")
                    strategi.append("🌱 Rotasi dengan legume (kedelai/kacang tanah) untuk fiksasi N")
                elif zona_char['n_mean'] < 28000:
                    strategi.append("🟡 Pemupukan Nitrogen standar: Urea 150-200 kg/ha")
                
                if zona_char['c_org_mean'] < 2.0:
                    strategi.append("🟤 Aplikasi kompos 2-3 ton/ha untuk meningkatkan bahan organik")
                
                if zona_char['ch_tahunan'] < 1400:
                    strategi.append("💧 **PRIORITAS**: Irigasi wajib - tetes atau sprinkler untuk efisiensi air")
                    strategi.append("🍃 Mulsa organik untuk konservasi kelembaban tanah")
                elif zona_char['ch_tahunan'] < 1700:
                    strategi.append("💧 Irigasi suplemen saat musim kemarau")
                
                if profile['elevasi_mean'] < 40:
                    strategi.append("🌊 Perhatian drainase: buat saluran pembuangan untuk cegah genangan")
                
                for i, s in enumerate(strategi, 1):
                    st.markdown(f"{i}. {s}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Daftar desa di zona ini
                with st.expander(f"Lihat Daftar Desa di Zona {zona_selected + 1} ({profile['jumlah_desa']} desa)", expanded=False):
                    desa_zona = df_hasil[df_hasil['cluster'] == zona_selected][['kecamatan', 'desa']].sort_values(['kecamatan', 'desa'])
                    
                    for kecamatan in sorted(desa_zona['kecamatan'].unique()):
                        desa_list = desa_zona[desa_zona['kecamatan'] == kecamatan]['desa'].tolist()
                        st.markdown(f"**Kec. {kecamatan}** ({len(desa_list)} desa):")
                        cols = st.columns(3)
                        for i, desa in enumerate(desa_list):
                            cols[i % 3].write(f"• {desa}")
                
                # Catatan penting
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div style='background: #1a2734; border-left: 4px solid #58a6ff; 
                            padding: 1rem; border-radius: 6px;'>
                    <h4 style='margin: 0 0 0.5rem 0; color: #58a6ff;'>📌 Catatan Penting</h4>
                    <ul style='margin: 0; color: #c9d1d9; font-size: 0.9rem;'>
                        <li>Skor kesesuaian dihitung dari: <b>pH (25%)</b>, <b>N (20%)</b>, <b>C-Organik (15%)</b>, <b>Curah Hujan (25%)</b>, <b>Temperatur (15%)</b></li>
                        <li>Tanaman dengan skor ≥80% <b>sangat direkomendasikan</b> untuk zona tersebut</li>
                        <li>Lakukan <b>uji coba skala kecil</b> sebelum implementasi luas</li>
                        <li>Konsultasikan dengan <b>Penyuluh Pertanian Lapangan (PPL)</b> setempat</li>
                        <li>Sesuaikan dengan kondisi mikro masing-masing desa dan <b>preferensi petani</b></li>
                        <li>Pertimbangkan aspek <b>ekonomi dan pasar</b> lokal</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # ==================== PREDIKSI ====================
    elif menu == "Prediksi Zona Baru":
        st.markdown('<h2 class="section-header">Prediksi Zona untuk Lokasi Baru</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4 style='margin: 0 0 0.5rem 0;'>ℹ️ Cara Penggunaan</h4>
            <p style='margin: 0; font-size: 0.95rem;'>
            Masukkan data karakteristik tanah, iklim, dan topografi dari lokasi yang ingin diprediksi.
            Sistem akan menggunakan model HDBSCAN yang sudah dilatih untuk menentukan zona agroekologi yang paling sesuai.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Preset nilai berdasarkan zona yang ada
        st.markdown("#### Gunakan Preset atau Input Manual")
        
        preset_options = {
            "Input Manual": None,
            "Contoh Zona 1 (CH Tinggi, Dataran Tinggi)": {
                "pH": 5.6, "c_organik": 2.7, "n": 28000.0, "p": 40.0, "k": 100.0,
                "curah_hujan": 157.0, "temp": 27.1, "penyinaran": 10.7, "elevasi": 100.0
            },
            "Contoh Zona 2 (CH Sedang, pH Netral)": {
                "pH": 5.7, "c_organik": 2.2, "n": 23000.0, "p": 39.0, "k": 85.0,
                "curah_hujan": 150.0, "temp": 27.1, "penyinaran": 10.8, "elevasi": 20.0
            },
            "Contoh Zona 3 (CH Rendah, Dataran Rendah)": {
                "pH": 5.7, "c_organik": 2.1, "n": 24000.0, "p": 39.0, "k": 90.0,
                "curah_hujan": 145.0, "temp": 27.6, "penyinaran": 10.9, "elevasi": 15.0
            }
        }
        
        selected_preset = st.selectbox("Pilih preset atau input manual:", list(preset_options.keys()))
        
        # Set nilai default berdasarkan preset
        if preset_options[selected_preset] is not None:
            preset = preset_options[selected_preset]
            default_pH = preset["pH"]
            default_c_organik = preset["c_organik"]
            default_n = preset["n"]
            default_p = preset["p"]
            default_k = preset["k"]
            default_ch = preset["curah_hujan"]
            default_temp = preset["temp"]
            default_penyinaran = preset["penyinaran"]
            default_elevasi = preset["elevasi"]
        else:
            # Default manual: rata-rata dari semua data
            default_pH = 5.7
            default_c_organik = 2.5
            default_n = 26000.0
            default_p = 40.0
            default_k = 95.0
            default_ch = 150.0
            default_temp = 27.2
            default_penyinaran = 10.8
            default_elevasi = 50.0
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌱 Parameter Tanah")
            pH = st.slider("pH Tanah", min_value=4.0, max_value=8.0, value=default_pH, step=0.1, 
                          help="Tingkat keasaman tanah (4.0-8.0)")
            c_organik = st.slider("C Organik (%)", min_value=0.0, max_value=10.0, value=default_c_organik, step=0.1,
                                 help="Kandungan karbon organik dalam tanah")
            n = st.slider("Nitrogen (ppm)", min_value=10000.0, max_value=40000.0, value=default_n, step=500.0,
                         help="Kandungan nitrogen dalam tanah")
            p = st.slider("Fosfor (ppm)", min_value=20.0, max_value=80.0, value=default_p, step=5.0,
                         help="Kandungan fosfor dalam tanah")
            k = st.slider("Kalium (ppm)", min_value=50.0, max_value=200.0, value=default_k, step=5.0,
                         help="Kandungan kalium dalam tanah")
        
        with col2:
            st.markdown("#### ☀️ Parameter Iklim & Topografi")
            curah_hujan = st.slider("Curah Hujan (mm/bulan)", min_value=100.0, max_value=200.0, value=default_ch, step=5.0,
                                   help="Rata-rata curah hujan per bulan")
            temp = st.slider("Temperatur Rata-rata (°C)", min_value=26.0, max_value=28.5, value=default_temp, step=0.1,
                            help="Temperatur rata-rata harian")
            penyinaran = st.slider("Penyinaran Matahari (jam/hari)", min_value=10.0, max_value=12.0, value=default_penyinaran, step=0.1,
                                  help="Rata-rata lama penyinaran matahari per hari")
            elevasi = st.slider("Ketinggian (mdpl)", min_value=0.0, max_value=250.0, value=default_elevasi, step=5.0,
                               help="Ketinggian dari permukaan laut")
        
        st.markdown("<br>", unsafe_allow_html=True)

        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔍 Analisis Zona", type="primary", use_container_width=True)
        
        if predict_button:
            with st.spinner("Menganalisis karakteristik lokasi..."):
                zona_id, zona_label, strength = prediksi_zona_baru(
                    model, scaler, pH, c_organik, n, p, k, curah_hujan, temp, penyinaran, elevasi
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Hasil Prediksi")
            
            # Debug info: Tampilkan 5 desa terdekat
            with st.expander("🔍 Debug: Lihat Analisis Detail"):
                df_hasil_debug = pd.read_csv('Hasil_Zonasi_Agroekologi_Pamekasan.csv')
                fitur_cols = ['pH', 'C_Organik_persen', 'N_ppm', 'P_ppm', 'K_ppm',
                             'Curah_Hujan_mm_per_bulan', 'Temp_C', 'Penyinaran_jam_per_hari', 'Elevasi_m']
                
                input_data = np.array([[pH, c_organik, n, p, k, curah_hujan, temp, penyinaran, elevasi]])
                input_scaled = scaler.transform(input_data)
                
                X_train = df_hasil_debug[fitur_cols].values
                X_train_scaled = scaler.transform(X_train)
                
                distances = euclidean_distances(input_scaled, X_train_scaled)[0]
                nearest_indices = np.argsort(distances)[:5]
                
                st.markdown("**5 Desa Terdekat (berdasarkan karakteristik):**")
                for i, idx in enumerate(nearest_indices, 1):
                    desa_row = df_hasil_debug.iloc[idx]
                    zona_desa = desa_row['cluster']
                    zona_label_desa = 'Noise' if zona_desa == -1 else f'Zona {int(zona_desa) + 1}'
                    st.write(f"{i}. **{desa_row['desa']}** ({zona_label_desa}) - Jarak: {distances[idx]:.4f}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if zona_id == -1:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                                padding: 2rem; border-radius: 12px; color: white;'>
                        <h2 style='margin: 0;'>Outlier Terdeteksi</h2>
                        <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                        Lokasi ini memiliki karakteristik yang unik dan tidak masuk ke zona manapun.
                        Pertimbangkan untuk analisis lebih lanjut.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 2rem; border-radius: 12px; color: white;'>
                        <h2 style='margin: 0;'>{zona_label}</h2>
                    """, unsafe_allow_html=True)
                    
                    if zona_id in zona_profiles:
                        profile = zona_profiles[zona_id]
                        st.markdown(f"""
                        <div style='margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.2); 
                                    border-radius: 8px;'>
                            <p style='margin: 0.3rem 0;'><b>Tipe Iklim:</b> {profile['tipe_iklim']}</p>
                            <p style='margin: 0.3rem 0;'><b>Topografi:</b> {profile['elevasi_kat'].title()}</p>
                            <p style='margin: 0.3rem 0;'><b>Karakteristik pH:</b> {profile['pH_kategori'].title()}</p>
                            <p style='margin: 0.3rem 0;'><b>Kesuburan:</b> {profile['kesuburan'].title()}</p>
                            <p style='margin: 0.3rem 0;'><b>Suhu:</b> {profile['temp_kat'].title()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background: #161b22; padding: 1.5rem; border-radius: 10px; 
                            border: 2px solid #30363d; text-align: center;'>
                    <p style='margin: 0; color: #8b949e; font-size: 0.9rem;'>Tingkat Keyakinan</p>
                    <h1 style='margin: 0.5rem 0; color: #e6edf3; font-size: 3rem;'>{strength:.1%}</h1>
                """, unsafe_allow_html=True)
                
                if strength > 0.7:
                    st.markdown("<p style='margin: 0; color: #4caf50; font-weight: 600;'>Sangat Tinggi ✓</p>", unsafe_allow_html=True)
                elif strength > 0.5:
                    st.markdown("<p style='margin: 0; color: #58a6ff; font-weight: 600;'>Tinggi</p>", unsafe_allow_html=True)
                elif strength > 0.3:
                    st.markdown("<p style='margin: 0; color: #dd6b20; font-weight: 600;'>Sedang</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='margin: 0; color: #e53e3e; font-weight: 600;'>Rendah</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Perbandingan dengan rata-rata zona
            if zona_id != -1 and zona_id in zona_profiles:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("#### Perbandingan dengan Karakteristik Zona")
                st.caption("Membandingkan input Anda dengan nilai rata-rata zona yang diprediksi")
                
                zona_data = df_hasil[df_hasil['cluster'] == zona_id]
                
                comparison_data = {
                    'Fitur': ['pH', 'C Organik (%)', 'N (ppm)', 'P (ppm)', 'K (ppm)',
                             'CH (mm/bln)', 'Temp (°C)', 'Penyinaran (jam)', 'Elevasi (m)'],
                    'Input': [pH, c_organik, n, p, k, curah_hujan, temp, penyinaran, elevasi],
                    'Rata-rata Zona': [
                        zona_data['pH'].mean(),
                        zona_data['C_Organik_persen'].mean(),
                        zona_data['N_ppm'].mean(),
                        zona_data['P_ppm'].mean(),
                        zona_data['K_ppm'].mean(),
                        zona_data['Curah_Hujan_mm_per_bulan'].mean(),
                        zona_data['Temp_C'].mean(),
                        zona_data['Penyinaran_jam_per_hari'].mean(),
                        zona_data['Elevasi_m'].mean()
                    ]
                }
                
                df_comparison = pd.DataFrame(comparison_data)
                df_comparison['Selisih (%)'] = ((df_comparison['Input'] - df_comparison['Rata-rata Zona']) / 
                                                df_comparison['Rata-rata Zona'] * 100)
                
                st.dataframe(
                    df_comparison.style.format({
                        'Input': '{:.2f}',
                        'Rata-rata Zona': '{:.2f}',
                        'Selisih (%)': '{:.1f}%'
                    }).background_gradient(subset=['Selisih (%)'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
    
    # ==================== UPLOAD DATA GEE ====================
    elif menu == "Upload Data GEE":
        st.markdown('<h2 class="section-header">Upload & Analisis Data GEE</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>Upload Data dari Google Earth Engine</h4>
            <p>Upload file CSV hasil export dari GEE dengan format yang sama seperti data training. 
            File harus memiliki kolom: <code>kecamatan, desa, lat, lon, tahun, pH, N_ppm, P_ppm, K_ppm, 
            C_Organik_persen, Curah_Hujan_mm_per_bulan, Temp_C, Penyinaran_jam_per_hari, Elevasi_m</code></p>
            <p><b>Catatan:</b> Data akan diproses menggunakan model HDBSCAN yang sudah dilatih untuk 
            mengidentifikasi zona agroekologi.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload file
        uploaded_file = st.file_uploader(
            "Pilih file CSV dari GEE",
            type=['csv'],
            help="Upload file CSV dengan format yang sesuai"
        )
        
        if uploaded_file is not None:
            try:
                # Baca file
                df_upload = pd.read_csv(uploaded_file)
                
                # Validasi kolom yang diperlukan
                required_cols = ['kecamatan', 'desa', 'lat', 'lon', 'tahun', 'pH', 'N_ppm', 'P_ppm', 
                               'K_ppm', 'C_Organik_persen', 'Curah_Hujan_mm_per_bulan', 'Temp_C', 
                               'Penyinaran_jam_per_hari', 'Elevasi_m']
                
                missing_cols = [col for col in required_cols if col not in df_upload.columns]
                
                if missing_cols:
                    st.error(f"Kolom yang hilang: {', '.join(missing_cols)}")
                    st.info("Kolom yang diperlukan: " + ", ".join(required_cols))
                else:
                    st.success(f"File berhasil diupload! Total {len(df_upload)} baris data")
                    
                    # Tampilkan preview data
                    with st.expander("Preview Data (5 baris pertama)"):
                        st.dataframe(df_upload.head(), use_container_width=True)
                    
                    # Info dataset
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Baris", len(df_upload))
                    with col2:
                        st.metric("Jumlah Desa", df_upload['desa'].nunique())
                    with col3:
                        st.metric("Jumlah Kecamatan", df_upload['kecamatan'].nunique())
                    with col4:
                        tahun_range = f"{df_upload['tahun'].min()}-{df_upload['tahun'].max()}"
                        st.metric("Periode Tahun", tahun_range)
                    
                    st.markdown("---")
                    
                    # Opsi preprocessing
                    st.markdown('<h3 class="section-header">Konfigurasi Preprocessing</h3>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        aggregate_method = st.selectbox(
                            "Metode Agregasi Multi-Temporal",
                            ["mean", "median"],
                            help="Metode untuk mengagregasi data multi-tahun menjadi satu nilai per desa"
                        )
                    with col2:
                        drop_na = st.checkbox(
                            "Hapus baris dengan missing values",
                            value=True,
                            help="Drop baris yang memiliki nilai kosong (NaN)"
                        )
                    
                    # Tombol proses
                    if st.button("Proses & Analisis Data", type="primary", use_container_width=True):
                        with st.spinner("Memproses data..."):
                            try:
                                # 1. Drop missing values jika diminta
                                if drop_na:
                                    df_processed = df_upload.dropna()
                                    if len(df_processed) < len(df_upload):
                                        st.warning(f"{len(df_upload) - len(df_processed)} baris dengan missing values dihapus")
                                else:
                                    df_processed = df_upload.copy()
                                
                                # 2. Agregasi temporal
                                fitur_cols = ['pH', 'C_Organik_persen', 'N_ppm', 'P_ppm', 'K_ppm',
                                             'Curah_Hujan_mm_per_bulan', 'Temp_C', 'Penyinaran_jam_per_hari', 'Elevasi_m']
                                
                                group_cols = ['kecamatan', 'desa', 'lat', 'lon']
                                
                                if aggregate_method == "mean":
                                    df_agg = df_processed.groupby(group_cols)[fitur_cols].mean().reset_index()
                                else:
                                    df_agg = df_processed.groupby(group_cols)[fitur_cols].median().reset_index()
                                
                                st.success(f"Data diagregasi: {len(df_agg)} desa unik")
                                
                                # 3. Normalisasi menggunakan scaler yang sudah ada
                                X = df_agg[fitur_cols].values
                                X_scaled = scaler.transform(X)
                                
                                # 4. Prediksi cluster
                                st.info("Melakukan clustering...")
                                labels = model.fit_predict(X_scaled)
                                
                                # Tambahkan hasil ke dataframe
                                df_agg['cluster'] = labels
                                df_agg['zona_agroekologi'] = df_agg['cluster'].apply(
                                    lambda x: 'Noise/Outlier' if x == -1 else f'Zona {x + 1}'
                                )
                                
                                # 5. Hitung statistik zona
                                st.markdown("---")
                                st.markdown('<h3 class="section-header">Hasil Clustering</h3>', unsafe_allow_html=True)
                                
                                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                n_noise = list(labels).count(-1)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Jumlah Zona", n_clusters)
                                with col2:
                                    st.metric("Desa Terklaster", len(df_agg) - n_noise)
                                with col3:
                                    st.metric("Noise/Outlier", n_noise)
                                with col4:
                                    noise_pct = (n_noise / len(df_agg)) * 100
                                    st.metric("% Noise", f"{noise_pct:.1f}%")
                                
                                # 6. Distribusi zona
                                st.markdown("**Distribusi Desa per Zona:**")
                                zona_counts = df_agg['zona_agroekologi'].value_counts().sort_index()
                                
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    for zona, count in zona_counts.items():
                                        pct = (count / len(df_agg)) * 100
                                        st.write(f"**{zona}:** {count} desa ({pct:.1f}%)")
                                
                                with col2:
                                    # Pie chart distribusi
                                    fig_pie = px.pie(
                                        values=zona_counts.values,
                                        names=zona_counts.index,
                                        title="Distribusi Zona",
                                        color_discrete_sequence=px.colors.qualitative.Set3
                                    )
                                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                
                                # 7. Peta hasil clustering
                                st.markdown("---")
                                st.markdown('<h3 class="section-header">Peta Zona Hasil Analisis</h3>', unsafe_allow_html=True)
                                
                                # Color mapping
                                zona_colors = {
                                    'Zona 1': '#1f77b4',
                                    'Zona 2': '#ff7f0e', 
                                    'Zona 3': '#2ca02c',
                                    'Zona 4': '#d62728',
                                    'Zona 5': '#9467bd',
                                    'Noise/Outlier': '#7f7f7f'
                                }
                                
                                df_agg['color'] = df_agg['zona_agroekologi'].map(zona_colors)
                                
                                # Buat hover text
                                df_agg['hover_text'] = (
                                    '<b>' + df_agg['desa'] + '</b><br>' +
                                    'Kecamatan: ' + df_agg['kecamatan'] + '<br>' +
                                    'Zona: ' + df_agg['zona_agroekologi'] + '<br>' +
                                    'pH: ' + df_agg['pH'].round(2).astype(str) + '<br>' +
                                    'N: ' + df_agg['N_ppm'].round(0).astype(str) + ' ppm<br>' +
                                    'CH: ' + df_agg['Curah_Hujan_mm_per_bulan'].round(0).astype(str) + ' mm/bln<br>' +
                                    'Temp: ' + df_agg['Temp_C'].round(1).astype(str) + ' °C<br>' +
                                    'Elevasi: ' + df_agg['Elevasi_m'].round(0).astype(str) + ' m'
                                )
                                
                                # Plotly map
                                fig_map = px.scatter_mapbox(
                                    df_agg,
                                    lat='lat',
                                    lon='lon',
                                    color='zona_agroekologi',
                                    color_discrete_map=zona_colors,
                                    hover_name='desa',
                                    hover_data={
                                        'lat': False,
                                        'lon': False,
                                        'zona_agroekologi': True,
                                        'kecamatan': True,
                                        'pH': ':.2f',
                                        'N_ppm': ':.0f',
                                        'Curah_Hujan_mm_per_bulan': ':.0f',
                                        'Temp_C': ':.1f',
                                        'Elevasi_m': ':.0f',
                                        'color': False
                                    },
                                    zoom=10,
                                    height=600,
                                    title="Peta Distribusi Zona Agroekologi"
                                )
                                
                                fig_map.update_layout(
                                    mapbox_style="open-street-map",
                                    margin={"r":0,"t":40,"l":0,"b":0},
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color='#c9d1d9'),
                                    title_font_color='#e6edf3'
                                )
                                
                                st.plotly_chart(fig_map, use_container_width=True)
                                
                                # 8. Tabel hasil lengkap
                                st.markdown("---")
                                st.markdown('<h3 class="section-header">Tabel Hasil Lengkap</h3>', unsafe_allow_html=True)
                                
                                # Pilih kolom untuk ditampilkan
                                display_cols = ['kecamatan', 'desa', 'zona_agroekologi', 'pH', 'N_ppm', 'P_ppm', 'K_ppm',
                                              'C_Organik_persen', 'Curah_Hujan_mm_per_bulan', 'Temp_C', 'Elevasi_m']
                                
                                st.dataframe(
                                    df_agg[display_cols].sort_values(['zona_agroekologi', 'desa']),
                                    use_container_width=True,
                                    height=400
                                )
                                
                                # 9. Statistik per zona
                                st.markdown("---")
                                st.markdown('<h3 class="section-header">Statistik Karakteristik per Zona</h3>', unsafe_allow_html=True)
                                
                                # Filter hanya zona (exclude noise)
                                df_zones = df_agg[df_agg['cluster'] != -1].copy()
                                
                                if len(df_zones) > 0:
                                    zona_stats = df_zones.groupby('zona_agroekologi')[fitur_cols].mean()
                                    
                                    st.dataframe(
                                        zona_stats.style.format('{:.2f}').background_gradient(cmap='YlGnBu'),
                                        use_container_width=True
                                    )
                                    
                                    # Radar chart perbandingan zona
                                    if len(zona_stats) > 1:
                                        st.markdown("**Perbandingan Karakteristik Zona (Normalized):**")
                                        
                                        # Normalisasi untuk radar chart
                                        from sklearn.preprocessing import MinMaxScaler
                                        scaler_viz = MinMaxScaler()
                                        zona_stats_norm = pd.DataFrame(
                                            scaler_viz.fit_transform(zona_stats),
                                            columns=zona_stats.columns,
                                            index=zona_stats.index
                                        )
                                        
                                        fig_radar = go.Figure()
                                        
                                        for zona in zona_stats_norm.index:
                                            fig_radar.add_trace(go.Scatterpolar(
                                                r=zona_stats_norm.loc[zona].values.tolist(),
                                                theta=zona_stats_norm.columns.tolist(),
                                                fill='toself',
                                                name=zona
                                            ))
                                        
                                        fig_radar.update_layout(
                                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                            showlegend=True,
                                            title="Radar Chart Karakteristik per Zona",
                                            height=500
                                        )
                                        
                                        st.plotly_chart(fig_radar, use_container_width=True)
                                
                                # 10. Download hasil
                                st.markdown("---")
                                st.markdown('<h3 class="section-header">Download Hasil</h3>', unsafe_allow_html=True)
                                
                                # Siapkan file download
                                csv_result = df_agg.to_csv(index=False)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        label="Download Hasil Zonasi (CSV)",
                                        data=csv_result,
                                        file_name=f"hasil_zonasi_{uploaded_file.name}",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                
                                with col2:
                                    # Summary JSON
                                    summary = {
                                        "total_desa": len(df_agg),
                                        "jumlah_zona": n_clusters,
                                        "noise": n_noise,
                                        "zona_distribusi": zona_counts.to_dict(),
                                        "aggregate_method": aggregate_method
                                    }
                                    st.download_button(
                                        label="Download Summary (JSON)",
                                        data=json.dumps(summary, indent=2),
                                        file_name=f"summary_zonasi_{uploaded_file.name.replace('.csv', '.json')}",
                                        mime="application/json",
                                        use_container_width=True
                                    )
                                
                                st.success("Analisis selesai! Anda dapat mendownload hasil di atas.")
                                
                            except Exception as e:
                                st.error(f"Terjadi kesalahan saat memproses: {str(e)}")
                                st.exception(e)
            
            except Exception as e:
                st.error(f"Gagal membaca file: {str(e)}")
                st.info("Pastikan file adalah CSV dengan encoding UTF-8")
        
        else:
            # Tampilkan template format CSV
            st.markdown("---")
            st.markdown('<h3 class="section-header">Template Format CSV</h3>', unsafe_allow_html=True)
            st.markdown("""
            File CSV harus memiliki kolom berikut (sesuai urutan atau dengan header yang tepat):
            """)
            
            template_data = {
                'kecamatan': ['Tlanakan', 'Tlanakan'],
                'desa': ['Dabuan', 'Terrak'],
                'lat': [-7.197528, -7.182909],
                'lon': [113.512889, 113.528417],
                'tahun': [2014, 2014],
                'pH': [6.2, 6.5],
                'N_ppm': [25000, 26500],
                'P_ppm': [800, 850],
                'K_ppm': [1200, 1300],
                'C_Organik_persen': [2.5, 2.8],
                'Curah_Hujan_mm_per_bulan': [120, 125],
                'Temp_C': [27.5, 27.4],
                'Penyinaran_jam_per_hari': [11.2, 11.3],
                'Elevasi_m': [22, 56]
            }
            
            df_template = pd.DataFrame(template_data)
            st.dataframe(df_template, use_container_width=True)
            
            # Download template
            csv_template = df_template.to_csv(index=False)
            st.download_button(
                label="Download Template CSV",
                data=csv_template,
                file_name="template_data_gee.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <p style='margin: 0.5rem 0;'><b>Sistem Zonasi Agroekologi Pamekasan</b></p>
        <p style='margin: 0.5rem 0;'>Menggunakan Algoritma HDBSCAN dengan 9 Parameter Tanah, Iklim & Topografi</p>
        <p style='margin: 0.5rem 0; font-size: 0.85rem;'>
            Data Sumber: SoilGrids (ISRIC) • ERA5-Land (Copernicus) • SRTM DEM • 2014-2024
        </p>
        <p style='margin: 1rem 0 0 0; font-size: 0.8rem;'>
            © 2026 | Kabupaten Pamekasan, Jawa Timur
        </p>
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError as e:
    st.error(f"""
    File tidak ditemukan: {e}
    
    Pastikan file-file berikut ada di direktori yang sama:
    - Hasil_Zonasi_Agroekologi_Pamekasan.csv
    - Tanah-Iklim_2014-2024_clean.csv
    - best_parameters.json
    - model_hdbscan_pamekasan.pkl
    - scaler_zonasi.pkl
    """)
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
    st.exception(e)
