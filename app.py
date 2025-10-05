# -*- coding: utf-8 -*-
# Pra-Pemrosesan Citra Suhu (Surabaya)
# Median + CLAHE (OpenCV), peta suhu (kalibrasi 2 titik),
# metrik kualitas, & model pendinginan Newton (seri kopi).

import io, os, re, math
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import streamlit as st
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Opsional: dukungan HEIC/HEIF
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

# ---------------- Utils ----------------
def load_image(file):
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)

def preprocess_rgb(img_rgb, ksize=3, clip=2.0, grid=8):
    """Median blur pada L (LAB) + CLAHE → kembali RGB & channel L-equalized."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    Ld = cv2.medianBlur(L, int(ksize))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    Le = clahe.apply(Ld)
    lab_e = cv2.merge([Le, a, b])
    bgr_e = cv2.cvtColor(lab_e, cv2.COLOR_LAB2BGR)
    rgb_e = cv2.cvtColor(bgr_e, cv2.COLOR_BGR2RGB)
    return rgb_e, Le  # Le: uint8

def metrics_hist(gray_u8):
    hist, _ = np.histogram(gray_u8.ravel(), bins=256, range=(0,255))
    var = float(np.var(hist))
    p = hist / (hist.sum() if hist.sum() else 1)
    ent = float(-(p[p>0]*np.log2(p[p>0])).sum())
    return var, ent

def parse_ts_from_name(name):
    base = os.path.basename(str(name))
    pats = [
        r'(\d{4})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})',
        r'(\d{4})[-_](\d{2})[-_](\d{2})[ _-](\d{2})[:_-](\d{2})[:_-](\d{2})'
    ]
    for pat in pats:
        m = re.search(pat, base)
        if m:
            y,M,d,h,mi,s = map(int, m.groups())
            try: return datetime(y,M,d,h,mi,s)
            except: pass
    return None

def clean_float(x):
    if x is None: return None
    s = str(x).strip().replace("°C","").replace("C","").replace("c","").replace(",", ".")
    try: return float(s)
    except: return None

def is_num(v):
    return (v is not None) and not (isinstance(v, float) and math.isnan(v))

def auto_refs(gray_u8, amb_C, hot_guess_C):
    """ref1 = patch pojok kiri-atas; ref2 = hotspot persentil 99.5%."""
    thr = np.percentile(gray_u8.ravel(), 99.5)
    pos = np.argwhere(gray_u8 >= thr)
    if pos.size:
        y2,x2 = map(int, pos[0])
    else:
        y2,x2 = gray_u8.shape[0]//2, gray_u8.shape[1]//2
    return {"x":5,"y":5,"T":amb_C}, {"x":x2,"y":y2,"T":hot_guess_C}

def temp_map_from_refs(gray_u8, ref1, ref2, win=5):
    """Linear map T = a*I + b dari 2 titik referensi (mean patch)."""
    h,w = gray_u8.shape

    def patch_mean(x,y):
        # Tahan None/NaN & clamp ke batas citra
        if (x is None) or (isinstance(x,float) and np.isnan(x)): x = 0.0
        if (y is None) or (isinstance(y,float) and np.isnan(y)): y = 0.0
        x = int(np.clip(round(float(x)), 0, w-1))
        y = int(np.clip(round(float(y)), 0, h-1))
        xs, xe = max(0,x-win), min(w,x+win+1)
        ys, ye = max(0,y-win), min(h,y+win+1)
        return float(np.mean(gray_u8[ys:ye, xs:xe]))

    I1 = patch_mean(ref1["x"], ref1["y"])
    I2 = patch_mean(ref2["x"], ref2["y"])
    if abs(I2-I1) < 1e-6:
        a = 0.0; b = ref1["T"]
    else:
        a = (ref2["T"] - ref1["T"]) / (I2 - I1)
        b = ref1["T"] - a*I1
    T = a*gray_u8.astype(np.float32) + b
    return T, (a,b), (I1,I2)

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ---- CSV loader: auto deteksi encoding & delimiter ----
def read_csv_smart(uploaded_file):
    """
    Menerima st.uploaded_file. Auto deteksi encoding & delimiter.
    Return: (DataFrame, encoding, delimiter)
    """
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin1"]
    delimiters = [None, ",", ";", "\t"]  # None => auto (engine='python')

    last_err = None
    for enc in encodings:
        try:
            text = raw.decode(enc)
        except Exception as e:
            last_err = e; continue
        sio = io.StringIO(text)
        for sep in delimiters:
            try:
                df = pd.read_csv(sio, sep=sep, engine="python")
                if df.shape[1] <= 1:
                    sio.seek(0); continue
                uploaded_file.seek(0)
                return df, enc, (sep if sep else "auto")
            except Exception as e:
                last_err = e; sio.seek(0)
    raise ValueError(f"Gagal membaca CSV. Simpan ulang sebagai 'CSV UTF-8'. Detail: {last_err}")

# ---------------- UI ----------------
st.set_page_config(page_title="Pra-Pemrosesan Citra Suhu", layout="wide")
st.title("Pra-Pemrosesan Citra Suhu (Surabaya)")
st.caption("Median + CLAHE + Colormap, kalibrasi 2 titik, metrik kualitas, & laju pendinginan Newton (kopi).")
st.sidebar.caption(f"App path: {__file__}")

st.sidebar.header("Pengaturan")
ksize   = st.sidebar.selectbox("Median ksize", [3,5,7], index=0)
clip    = st.sidebar.slider("CLAHE clip limit", 1.0, 4.0, 2.0, 0.1)
grid    = st.sidebar.selectbox("CLAHE tile size", [4,8,16], index=1)
cmap    = st.sidebar.selectbox("Colormap", ["inferno","magma","plasma","viridis"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Kalibrasi Dua Titik")
use_auto          = st.sidebar.checkbox("Auto-kalibrasi saat referensi kosong", True)
ambient_default   = st.sidebar.number_input("Asumsi ambien (°C) jika CSV kosong", value=31.0, step=0.5)
hot_guess_coffee  = st.sidebar.number_input("Hotspot default KOPI (°C)", value=33.0, step=0.5)
hot_guess_asphalt = st.sidebar.number_input("Hotspot default ASPAL (°C)", value=34.0, step=0.5)

files    = st.file_uploader("Upload gambar (JPG/PNG/HEIC)—bisa banyak",
                            type=["jpg","jpeg","png","bmp","tif","tiff","heic"], accept_multiple_files=True)
csv_file = st.file_uploader("Upload CSV metadata (opsional)", type=["csv"])

# CSV opsional (tahan file non-UTF8 & delimiter ;)
meta = None
if csv_file is not None:
    try:
        meta, enc_used, sep_used = read_csv_smart(csv_file)
        st.caption(f"CSV terbaca • encoding: {enc_used} • delimiter: {sep_used}")
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        meta = None

    if meta is not None:
        meta.columns = [c.strip().lower() for c in meta.columns]
        if "ambient_c" not in meta.columns:
            for c in list(meta.columns):
                if c.startswith("ambient"):
                    meta = meta.rename(columns={c:"ambient_c"})
                    break
        for c in ["filename","timestamp_wib","scene","notes","ambient_c",
                  "ref1_x","ref1_y","ref1_c","ref2_x","ref2_y","ref2_c"]:
            if c not in meta.columns:
                meta[c] = ""

def meta_row(name):
    if meta is None: return None
    m = meta[meta["filename"].astype(str) == str(name)]
    if m.empty: return None
    return m.iloc[0].to_dict()

def norm_scene(s):
    if not s: return s
    s = str(s).strip().lower()
    if s in ["kopi","coffee"]: return "coffee"
    if s in ["aspal","asphalt"]: return "asphalt"
    return s

# ---------------- Proses ----------------
if files:
    st.subheader("Hasil per gambar")
    rows = []

    for f in files:
        try:
            rgb = load_image(f)
        except Exception as e:
            st.error(f"Gagal baca {f.name}: {e}"); continue

        proc, L = preprocess_rgb(rgb, ksize=ksize, clip=clip, grid=grid)
        v0, e0 = metrics_hist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        v1, e1 = metrics_hist(L)

        rmeta = meta_row(f.name)
        scene = norm_scene((rmeta.get("scene") if rmeta else "") or "")
        amb   = clean_float(rmeta["ambient_c"]) if rmeta else None
        if amb is None: amb = ambient_default

        def get_ref(rx, ry, rc):
            if rmeta is None: return None
            x = clean_float(rmeta.get(rx)); y = clean_float(rmeta.get(ry)); c = clean_float(rmeta.get(rc))
            if not (is_num(x) and is_num(y) and is_num(c)):
                return None
            return {"x": float(x), "y": float(y), "T": float(c)}

        r1 = get_ref("ref1_x","ref1_y","ref1_c")
        r2 = get_ref("ref2_x","ref2_y","ref2_c")
        if (r1 is None or r2 is None) and use_auto:
            hot = hot_guess_coffee if scene == "coffee" else hot_guess_asphalt
            r1a, r2a = auto_refs(L, amb, hot)
            if r1 is None: r1 = r1a
            if r2 is None: r2 = r2a

        # Visual 1: original vs denoise+CLAHE
        fig1 = plt.figure(figsize=(8,4))
        plt.subplot(1,2,1); plt.imshow(rgb);  plt.title("Original");      plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(proc); plt.title("Denoise+CLAHE"); plt.axis("off")
        st.image(fig_to_bytes(fig1), caption=f"{f.name} — Original vs Denoise+CLAHE")

        # Visual 2: peta suhu / heat index
        if r1 is not None and r2 is not None:
            Tmap, (a,b), (I1,I2) = temp_map_from_refs(L, r1, r2, win=5)
            fig2 = plt.figure(figsize=(6,5))
            im = plt.imshow(Tmap, cmap=cmap); plt.title(f"{f.name} — Peta Suhu (°C)")
            cb = plt.colorbar(im); cb.set_label("°C"); plt.axis("off")
            buf2 = fig_to_bytes(fig2)
            st.image(buf2, caption=f"{f.name} — Peta Suhu (°C)")
            st.download_button("Download peta suhu", data=buf2, file_name=f"{os.path.splitext(f.name)[0]}_temp.png")
            peakT = float(np.nanmax(Tmap)); meanT = float(np.nanmean(Tmap))
        else:
            HI = cv2.normalize(L, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            fig2 = plt.figure(figsize=(6,5))
            im = plt.imshow(HI, cmap=cmap); plt.title(f"{f.name} — Heat Index (0..1)")
            cb = plt.colorbar(im); cb.set_label("index"); plt.axis("off")
            buf2 = fig_to_bytes(fig2)
            st.image(buf2, caption=f"{f.name} — Heat Index")
            st.download_button("Download heat index", data=buf2, file_name=f"{os.path.splitext(f.name)[0]}_heatindex.png")
            peakT = float(np.max(HI)); meanT = float(np.mean(HI))

        # Ringkasan baris
        ts = ""
        if rmeta and str(rmeta.get("timestamp_wib","")).strip():
            ts = str(rmeta["timestamp_wib"]).strip()
        else:
            dt = parse_ts_from_name(f.name)
            ts = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""

        rows.append({
            "filename": f.name,
            "scene": scene,
            "timestamp_wib": ts,
            "ambient_C": amb,
            "peak_T": peakT,
            "mean_T": meanT,
            "var_before": v0, "ent_before": e0,
            "var_after": v1, "ent_after": e1
        })

    # Tabel ringkasan
    st.markdown("---")
    st.subheader("Ringkasan")
    df_sum = pd.DataFrame(rows)
    st.dataframe(df_sum, use_container_width=True)

    # Model pendinginan Newton (kopi)
    st.markdown("---")
    st.subheader("Model Fisika: Laju Pendinginan Newton (kopi)")
    df_c = df_sum[df_sum["scene"].astype(str).str.lower()=="coffee"].copy()
    if len(df_c) >= 3:
        try:
            Tamb = float(df_c["ambient_C"].iloc[0])
            df_c["t"] = pd.to_datetime(df_c["timestamp_wib"], errors="coerce")
            df_c = df_c.dropna(subset=["t"]).sort_values("t")
            t0 = df_c["t"].iloc[0]
            X = (df_c["t"] - t0).dt.total_seconds().values / 60.0
            Y = df_c["peak_T"].values - Tamb
            mask = Y > 0
            if mask.sum() >= 2:
                k = -np.polyfit(X[mask], np.log(Y[mask]), 1)[0]
                ylog = np.log(Y[mask]); yhat = np.polyval([-k, ylog[0]], X[mask])
                ss_res = np.sum((ylog - yhat)**2); ss_tot = np.sum((ylog - ylog.mean())**2)
                r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
                t_pred = np.linspace(0, max(X[-1]+20, 20), 200)
                T_pred = Tamb + (Y[mask][0]) * np.exp(-k * t_pred)
                fig = plt.figure()
                plt.plot(X, Y + Tamb, 'o', label="peak T (°C)")
                plt.plot(t_pred, T_pred, '-', label="Newton fit")
                plt.xlabel("Waktu (menit)"); plt.ylabel("Suhu (°C)")
                plt.legend(); plt.tight_layout()
                st.pyplot(fig)
                st.write(f"Estimasi k ≈ **{k:.4f} min⁻¹**, R²(log) ≈ **{r2:.3f}**")
            else:
                st.info("ΔT kecil atau timestamp kurang → fitting k belum stabil.")
        except Exception as e:
            st.warning(f"Gagal memodelkan pendinginan: {e}")
    else:
        st.info("Butuh ≥3 foto kopi dengan timestamp untuk model Newton.")
else:
    st.info("Upload gambar (JPG/PNG/HEIC) dan CSV opsional untuk mulai.")


