import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import smtplib
import os
import tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import openai
import requests
from datetime import datetime

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", 465))
SMTP_USER      = os.getenv("SMTP_USERNAME")
SMTP_PASS      = os.getenv("SMTP_PASSWORD")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Mining Lab – Photo Booth",
    page_icon="📷",
    layout="centered",
)

# ── Custom CSS  (warna dari logo: biru #1A6FA8, kuning/emas #F5A623, abu #5A5A5A)
st.markdown("""
<style>
/* ---- Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* ---- Root variables ---- */
:root {
    --primary:   #1A6FA8;
    --accent:    #F5A623;
    --dark:      #1C2B3A;
    --surface:   #EAF3FB;
    --text-dark: #1C2B3A;
    --text-light: #FFFFFF;
}

/* ---- Body / app background ---- */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1C2B3A 0%, #1A6FA8 60%, #0d3d61 100%) !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: rgba(28, 43, 58, 0.95) !important;
    border-right: 2px solid #F5A623;
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stRadio label { color: #FFFFFF !important; }

/* ---- Main content card ---- */
.main-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(245,166,35,0.4);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
}

/* ---- Section titles ---- */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {
    color: #FFFFFF !important;
    text-shadow: 0 2px 6px rgba(0,0,0,0.5);
}

/* ---- Labels & captions ---- */
label, .stRadio > label, .stSelectbox label,
[data-testid="stWidgetLabel"], p, span, li {
    color: #FFFFFF !important;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(90deg, #F5A623, #e0921a) !important;
    color: #1C2B3A !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.4rem !important;
    transition: transform .15s, box-shadow .15s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(245,166,35,0.55) !important;
}

/* ── Download button override ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(90deg, #1A6FA8, #0d5387) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
}

/* ── Text input / email box ── */
.stTextInput > div > div > input {
    background: #FFFFFF !important;
    color: #1C2B3A !important;
    border: 2px solid #F5A623 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stTextInput > div > div > input::placeholder { color: #888888 !important; }

/* ── Info / success / warning boxes ── */
.stAlert { border-radius: 10px !important; }

/* ── Camera widget frame ── */
[data-testid="stCameraInput"] label { color: #FFFFFF !important; }

/* ── Divider ── */
hr { border-color: rgba(245,166,35,0.3) !important; }

/* ── Badge chip ── */
.badge {
    display: inline-block;
    background: #F5A623;
    color: #1C2B3A;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 99px;
    margin-bottom: 8px;
}

/* ── Gallery card ── */
.gallery-item {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(245,166,35,0.35);
    border-radius: 12px;
    padding: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "saved_photos" not in st.session_state:
    st.session_state.saved_photos = []   # list of {"name": str, "bytes": bytes, "mode": str}

# ── Helper functions ──────────────────────────────────────────────────────────

def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── FIX 2: Mirror / flip image horizontally (selfie mirror effect) ────────────
def mirror_image(img: Image.Image) -> Image.Image:
    """Flip image horizontally to simulate selfie mirror effect."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def apply_monochrome(img: Image.Image) -> Image.Image:
    """Convert to greyscale, return RGB PIL image."""
    grey = img.convert("L")
    return grey.convert("RGB")


# ── FIX 3: Photo Booth mode – add Data Mining logo border ────────────────────
def apply_photobooth_border(img: Image.Image) -> Image.Image:
    """
    Add a decorative border inspired by the Data Mining Lab logo colors:
    - Outer border  : abu-abu gelap  #5A5A5A
    - Middle stripe : biru muda/cyan #4BB8D4
    - Inner stripe  : kuning/emas    #F5A623
    The logo is overlaid at the bottom-right corner.
    """
    # Border thickness proportional to image size
    w, h = img.size
    outer  = max(12, int(min(w, h) * 0.022))   # abu-abu  #5A5A5A
    middle = max(7,  int(min(w, h) * 0.013))   # cyan     #4BB8D4
    inner  = max(5,  int(min(w, h) * 0.009))   # emas     #F5A623

    total_border = outer + middle + inner

    # New canvas size
    new_w = w + total_border * 2
    new_h = h + total_border * 2

    # ── Draw layered border ──
    canvas = Image.new("RGB", (new_w, new_h), "#5A5A5A")   # outer

    mid_canvas = Image.new("RGB", (new_w - outer*2, new_h - outer*2), "#4BB8D4")
    canvas.paste(mid_canvas, (outer, outer))

    inner_canvas = Image.new("RGB",
        (new_w - (outer+middle)*2, new_h - (outer+middle)*2), "#F5A623")
    canvas.paste(inner_canvas, (outer+middle, outer+middle))

    # Paste original photo
    canvas.paste(img, (total_border, total_border))

    # ── Overlay logo at bottom-right ──
    logo_path = "/mnt/user-data/uploads/1776697830754_image.png"
    if os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            logo_max = int(min(new_w, new_h) * 0.18)
            logo.thumbnail((logo_max, logo_max), Image.LANCZOS)

            # Position: bottom-right with small padding
            pad = int(total_border * 0.6)
            lx = new_w - logo.width - pad
            ly = new_h - logo.height - pad

            # Semi-transparent white backing for readability
            backing = Image.new("RGBA", (logo.width + 8, logo.height + 8), (255, 255, 255, 180))
            canvas_rgba = canvas.convert("RGBA")
            canvas_rgba.paste(backing, (lx - 4, ly - 4), backing)
            canvas_rgba.paste(logo, (lx, ly), logo)
            canvas = canvas_rgba.convert("RGB")
        except Exception:
            pass  # If logo fails, border is still applied

    return canvas


# ── Ghibli with strict token budget (max $0.05, target ~$0.01–$0.02) ──────────
#
# Breakdown biaya per panggilan API:
#   Model  : gpt-image-1-mini   → output token $8/1M  (vs gpt-image-1 $40/1M)
#   Size   : 512x512            → ~272 output token   ≈ $0.002 output
#   Quality: low                → token paling sedikit
#   Input  : di-resize ke 256px → input image token ~$2.50/1M ≈ $0.001
#   Prompt : singkat ~30 token  → text token $2/1M    ≈ $0.0001
#   ─────────────────────────────────────────────────────────────────
#   TOTAL estimasi              ≈ $0.003–$0.015  (jauh di bawah $0.05)
#
def apply_ghibli(img: Image.Image) -> Image.Image:
    """
    Kirim foto ke OpenAI Images Edit dan kembalikan hasil Ghibli-style.

    Kontrol biaya:
      - Model  : gpt-image-1-mini  → 5× lebih murah dari gpt-image-1
      - Input  : di-resize ke max 256×256 sebelum upload
      - Output : size=512x512, quality=low
      - Estimasi total biaya ≈ $0.003–$0.015 per panggilan
    """
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        # ── Resize input ke max 256px → potong input token secara drastis ──
        input_img = img.copy()
        input_img.thumbnail((256, 256), Image.LANCZOS)

        buf = io.BytesIO()
        input_img.save(buf, format="PNG")
        buf.seek(0)

        response = client.images.edit(
            model="gpt-image-1-mini",   # 5× lebih murah dari gpt-image-1
            image=("photo.png", buf, "image/png"),
            prompt=(
                "Ghibli anime style. Soft watercolour, warm palette, "
                "keep face recognisable."
            ),
            n=1,
            size="512x512",             # resolusi kecil → output token minimal
            quality="low",              # tier paling hemat
        )

        import base64 as _b64
        img_data = _b64.b64decode(response.data[0].b64_json)
        result_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        # Upscale kembali ke ukuran asli agar tampil proporsional
        result_img = result_img.resize(img.size, Image.LANCZOS)
        return result_img

    except Exception as e:
        st.warning(f"⚠️ OpenAI API error ({e}). Applying local cartoon fallback.")
        return _cartoon_fallback(img)


def _cartoon_fallback(img: Image.Image) -> Image.Image:
    """Local cartoon / sketch effect as fallback."""
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey  = cv2.medianBlur(grey, 5)
    edges = cv2.adaptiveThreshold(
        grey, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cartoon_rgb)


def send_email(recipient: str, photos: list) -> bool:
    """Send photos as email attachments via SMTP SSL."""
    try:
        msg = MIMEMultipart()
        msg["From"]    = SMTP_USER
        msg["To"]      = recipient
        msg["Subject"] = "📷 Foto dari Data Mining Lab Photo Booth"

        body = (
            "Halo!\n\n"
            f"Terlampir {len(photos)} foto dari sesi Photo Booth\n"
            "Laboratorium Data Mining – Universitas Islam Indonesia.\n\n"
            "Salam,\nPhoto Booth App"
        )
        msg.attach(MIMEText(body, "plain"))

        for photo in photos:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(photo["bytes"])
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{photo["name"]}"',
            )
            msg.attach(part)

        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, recipient, msg.as_string())

        return True
    except Exception as e:
        st.error(f"Gagal mengirim email: {e}")
        return False

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    logo_path = "/mnt/user-data/uploads/1776697830754_image.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
with col_title:
    st.markdown("""
    <h1 style='margin-bottom:0; color:#FFFFFF;'>📷 Photo Booth</h1>
    <p style='color:#F5A623; font-size:0.9rem; margin-top:2px;'>
        Laboratorium Data Mining – Universitas Islam Indonesia
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Mode Foto")
    mode = st.radio(
        "Pilih mode kamera:",
        ["🔲 Monokrom / Greyscale", "🌸 Ghibli Style"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### 📁 Foto Tersimpan")
    st.markdown(f"<span class='badge'>{len(st.session_state.saved_photos)} foto</span>", unsafe_allow_html=True)
    if st.session_state.saved_photos:
        if st.button("🗑️ Hapus Semua Foto"):
            st.session_state.saved_photos = []
            st.rerun()
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:rgba(255,255,255,0.5);'>"
        "Versi 1.0 &nbsp;|&nbsp; Data Mining Lab</p>",
        unsafe_allow_html=True
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_filter, tab_booth = st.tabs(["📸 Mode Filter", "🖼️ Photo Booth"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – Mode Filter (Monokrom & Ghibli) — tidak ada perubahan logika
# ════════════════════════════════════════════════════════════════════════════════
with tab_filter:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("### 📸 Ambil Foto")

    is_ghibli = "Ghibli" in mode
    is_mono   = "Monokrom" in mode

    if is_ghibli:
        mode_label = "Ghibli Style"
        mode_color = "#F5A623"
    else:
        mode_label = "Monokrom"
        mode_color = "#1A6FA8"

    st.markdown(
        f"<p>Mode aktif: <span style='color:{mode_color}; font-weight:700;'>{mode_label}</span></p>",
        unsafe_allow_html=True
    )

    camera_image = st.camera_input("Klik tombol di bawah untuk mengambil foto", key="cam_filter")

    if camera_image is not None:
        raw_img = Image.open(camera_image).convert("RGB")

        # Mirror the captured image (selfie mirror)
        raw_img = mirror_image(raw_img)

        with st.spinner("⏳ Memproses foto..."):
            if is_ghibli:
                processed = apply_ghibli(raw_img)
            else:
                processed = apply_monochrome(raw_img)

        col_orig, col_proc = st.columns(2)
        with col_orig:
            st.markdown("<p style='text-align:center; color:#F5A623;'>📷 Original</p>", unsafe_allow_html=True)
            st.image(raw_img, use_container_width=True)
        with col_proc:
            st.markdown(f"<p style='text-align:center; color:#F5A623;'>✨ {mode_label}</p>", unsafe_allow_html=True)
            st.image(processed, use_container_width=True)

        proc_bytes = pil_to_bytes(processed)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{mode_label.replace(' ', '_')}_{ts}.png"

        st.markdown("#### Simpan Foto")
        col_dl, col_save = st.columns(2)

        with col_dl:
            st.download_button(
                label="⬇️ Download Langsung",
                data=proc_bytes,
                file_name=filename,
                mime="image/png",
                key="dl_filter_direct",
            )

        with col_save:
            MAX_PHOTOS = 3
            jumlah = len(st.session_state.saved_photos)
            if jumlah >= MAX_PHOTOS:
                st.warning(f"🚫 Galeri sudah penuh ({MAX_PHOTOS} foto). Kirim dulu ke email, lalu hapus.")
            else:
                if st.button(f"💾 Simpan ke Galeri ({jumlah}/{MAX_PHOTOS})", key="save_filter"):
                    st.session_state.saved_photos.append({
                        "name":  filename,
                        "bytes": proc_bytes,
                        "mode":  mode_label,
                    })
                    st.success(f"✅ Foto disimpan ke galeri! ({jumlah + 1}/{MAX_PHOTOS})")
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – Photo Booth (border logo Data Mining)
# ════════════════════════════════════════════════════════════════════════════════
with tab_booth:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("### 🖼️ Photo Booth – Border Logo Data Mining")
    st.markdown(
        "<p style='color:#4BB8D4;'>Foto biasa dengan border berlapis warna logo "
        "Laboratorium Data Mining + overlay logo di pojok kanan bawah.</p>",
        unsafe_allow_html=True
    )

    camera_booth = st.camera_input("Klik tombol di bawah untuk mengambil foto", key="cam_booth")

    if camera_booth is not None:
        raw_booth = Image.open(camera_booth).convert("RGB")

        # Mirror the captured image (selfie mirror)
        raw_booth = mirror_image(raw_booth)

        with st.spinner("⏳ Menambahkan border..."):
            processed_booth = apply_photobooth_border(raw_booth)

        col_orig_b, col_proc_b = st.columns(2)
        with col_orig_b:
            st.markdown("<p style='text-align:center; color:#F5A623;'>📷 Original</p>", unsafe_allow_html=True)
            st.image(raw_booth, use_container_width=True)
        with col_proc_b:
            st.markdown("<p style='text-align:center; color:#4BB8D4;'>✨ Photo Booth</p>", unsafe_allow_html=True)
            st.image(processed_booth, use_container_width=True)

        proc_bytes_b = pil_to_bytes(processed_booth)
        ts_b = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_b = f"photo_PhotoBooth_{ts_b}.png"

        st.markdown("#### Simpan Foto")
        col_dl_b, col_save_b = st.columns(2)

        with col_dl_b:
            st.download_button(
                label="⬇️ Download Langsung",
                data=proc_bytes_b,
                file_name=filename_b,
                mime="image/png",
                key="dl_booth_direct",
            )

        with col_save_b:
            MAX_PHOTOS = 3
            jumlah_b = len(st.session_state.saved_photos)
            if jumlah_b >= MAX_PHOTOS:
                st.warning(f"🚫 Galeri sudah penuh ({MAX_PHOTOS} foto). Kirim dulu ke email, lalu hapus.")
            else:
                if st.button(f"💾 Simpan ke Galeri ({jumlah_b}/{MAX_PHOTOS})", key="save_booth"):
                    st.session_state.saved_photos.append({
                        "name":  filename_b,
                        "bytes": proc_bytes_b,
                        "mode":  "Photo Booth",
                    })
                    st.success(f"✅ Foto disimpan ke galeri! ({jumlah_b + 1}/{MAX_PHOTOS})")
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ── Saved gallery ─────────────────────────────────────────────────────────────
if st.session_state.saved_photos:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("### 🖼️ Galeri Foto Tersimpan")

    cols = st.columns(min(3, len(st.session_state.saved_photos)))
    for idx, photo in enumerate(st.session_state.saved_photos):
        with cols[idx % 3]:
            st.markdown("<div class='gallery-item'>", unsafe_allow_html=True)
            st.image(photo["bytes"], use_container_width=True)
            st.markdown(
                f"<p style='font-size:0.72rem; color:#F5A623;'>{photo['name']}</p>",
                unsafe_allow_html=True
            )
            st.download_button(
                label="⬇️",
                data=photo["bytes"],
                file_name=photo["name"],
                mime="image/png",
                key=f"dl_{idx}",
            )
            if st.button("❌", key=f"del_{idx}"):
                st.session_state.saved_photos.pop(idx)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Send email ────────────────────────────────────────────────────────────
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("### 📧 Kirim Foto via Email")
    st.markdown(
        "<p>Semua foto di galeri akan dikirim sebagai lampiran email.</p>",
        unsafe_allow_html=True
    )

    recipient_email = st.text_input(
        "Alamat email tujuan",
        placeholder="contoh@email.com"
    )

    if st.button("📤 Kirim Sekarang"):
        if not recipient_email or "@" not in recipient_email:
            st.warning("⚠️ Masukkan alamat email yang valid.")
        else:
            with st.spinner("📨 Mengirim email..."):
                ok = send_email(recipient_email, st.session_state.saved_photos)
            if ok:
                st.success(
                    f"✅ {len(st.session_state.saved_photos)} foto berhasil "
                    f"dikirim ke **{recipient_email}**!"
                )

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("💡 Belum ada foto di galeri. Ambil foto lalu tekan **Simpan ke Galeri**.")