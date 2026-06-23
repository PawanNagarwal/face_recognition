import streamlit as st
import numpy as np
import os
import pickle
import tempfile
import datetime
import threading
from PIL import Image, ImageDraw
import cv2
from insightface.app import FaceAnalysis
import pandas as pd
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

DB_DIR = "face_db_images"
DB_FILE = "face_embeddings.pkl"
SIMILARITY_THRESHOLD = 0.5
FRAME_SAMPLE_INTERVAL = 15
MIN_FACE_SIZE = 60
DEDUP_THRESHOLD = 0.55

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

os.makedirs(DB_DIR, exist_ok=True)

st.set_page_config(page_title="Sentinel Face Recognition Command Center", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b))

def normalize_embedding(emb):
    emb = np.asarray(emb, dtype=np.float32)
    n = np.linalg.norm(emb)
    return emb if n == 0 else emb / n

def get_embedding(model, img_array):
    faces = model.get(img_array)
    if not faces:
        return None, []
    largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return normalize_embedding(largest.normed_embedding), faces

def get_all_faces(model, img_array):
    return model.get(img_array) or []

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

def enroll_person(model, name, img_array):
    emb, _ = get_embedding(model, img_array)
    if emb is None:
        return False, "No face detected."
    db = load_db()
    db.setdefault(name, []).append(emb)
    save_db(db)
    return True, f"Enrolled {name}. Templates: {len(db[name])}"

def search_face(model, img_array):
    qemb, _ = get_embedding(model, img_array)
    if qemb is None:
        return None, "No face detected in the query image."
    db = load_db()
    if not db:
        return None, "Database is empty. Please enroll some faces first."
    results = []
    for name, embs in db.items():
        best = min(cosine_distance(qemb, e) for e in embs)
        results.append((name, best))
    results.sort(key=lambda x: x[1])
    return results, None

def search_all_faces_in_frame(model, img_array, threshold):
    faces = get_all_faces(model, img_array)
    if not faces:
        return []
    db = load_db()
    results = []
    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        emb = normalize_embedding(face.normed_embedding)
        if not db:
            results.append({
                "bbox": (x1, y1, x2, y2),
                "name": "UNKNOWN",
                "distance": 1.0,
                "similarity": 0.0,
                "matched": False
            })
            continue

        best_name, best_dist = None, 1.0
        for person, embs in db.items():
            d = min(cosine_distance(emb, e) for e in embs)
            if d < best_dist:
                best_dist = d
                best_name = person

        matched = best_dist <= threshold
        results.append({
            "bbox": (x1, y1, x2, y2),
            "name": best_name if matched else "UNKNOWN",
            "distance": best_dist,
            "similarity": round((1 - best_dist) * 100, 1),
            "matched": matched
        })
    return results

def draw_boxes_on_frame(frame_bgr, face_results):
    for r in face_results:
        x1, y1, x2, y2 = r["bbox"]
        color = (0, 220, 80) if r["matched"] else (0, 50, 235)
        label = f"{r['name']}  {r['similarity']}%" if r["matched"] else f"UNKNOWN  {r['similarity']}%"
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame_bgr, label, (x1 + 3, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if not r["matched"]:
            cv2.putText(frame_bgr, "! RED FLAG", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 50, 235), 2, cv2.LINE_AA)
    return frame_bgr

def annotate_frame_pil(pil_img, face_results):
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for r in face_results:
        x1, y1, x2, y2 = r["bbox"]
        color = (34, 197, 94) if r["matched"] else (239, 68, 68)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{r['name']}  {r['similarity']}%" if r["matched"] else f"UNKNOWN  {r['similarity']}%"
        label_h = 22
        draw.rectangle([x1, max(0, y1 - label_h), x2, y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - label_h) + 3), label, fill=(255, 255, 255))
    return img

def delete_person(name):
    db = load_db()
    if name in db:
        del db[name]
        save_db(db)
        img_path = os.path.join(DB_DIR, f"{name}.jpg")
        if os.path.exists(img_path):
            os.remove(img_path)
        return True
    return False

def pil_to_bgr(pil_img):
    img = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def add_to_cluster(clusters, emb, crop, frame_no, bbox):
    for c in clusters:
        if cosine_distance(emb, c["mean_embedding"]) < DEDUP_THRESHOLD:
            c["embeddings"].append(emb)
            mean = np.mean(np.asarray(c["embeddings"]), axis=0)
            c["mean_embedding"] = normalize_embedding(mean)
            old = c["bbox"]
            old_area = (old[2] - old[0]) * (old[3] - old[1])
            new_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if new_area > old_area:
                c["crop"] = crop
                c["frame_no"] = frame_no
                c["bbox"] = bbox
            return
    clusters.append({
        "embeddings": [emb],
        "mean_embedding": emb.copy(),
        "crop": crop,
        "frame_no": frame_no,
        "bbox": bbox
    })

def extract_faces_from_video(model, video_path, frame_interval=FRAME_SAMPLE_INTERVAL, min_face_px=MIN_FACE_SIZE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], "Could not open video file."
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    clusters = []
    frame_no = 0
    sampled = 0
    progress_bar = st.progress(0, text="Scanning video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % frame_interval == 0:
            faces = get_all_faces(model, frame)
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                w, h = x2 - x1, y2 - y1
                if w < min_face_px or h < min_face_px:
                    continue
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1c:y2c, x1c:x2c].copy()
                if crop.size == 0:
                    continue
                add_to_cluster(clusters, normalize_embedding(face.normed_embedding), crop, frame_no, [x1, y1, x2, y2])
            sampled += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_no / total_frames, 1.0), text=f"Scanning... frame {frame_no}/{total_frames} | Unique faces: {len(clusters)}")
        frame_no += 1
    cap.release()
    progress_bar.progress(1.0, text=f"Done. Found {len(clusters)} unique face(s) in {sampled} sampled frames.")
    return clusters, None

def save_named_faces(faces_data, names_input):
    db = load_db()
    saved = 0
    skipped = 0
    for idx, name in names_input:
        name = (name or "").strip()
        if not name:
            skipped += 1
            continue
        face = faces_data[idx]
        for emb in face["embeddings"]:
            db.setdefault(name, []).append(emb)
        thumb = os.path.join(DB_DIR, f"{name}.jpg")
        if not os.path.exists(thumb):
            cv2.imwrite(thumb, face["crop"])
        saved += 1
    save_db(db)
    return saved, skipped

def save_bulk_folder(model, folder_files):
    db = load_db()
    enrolled = 0
    skipped = 0
    for file in folder_files:
        fname = os.path.basename(file.name)
        name, _ = os.path.splitext(fname)
        name = name.strip()
        if not name:
            skipped += 1
            continue
        try:
            pil_img = Image.open(file)
            bgr_img = pil_to_bgr(pil_img)
            emb, _ = get_embedding(model, bgr_img)
            if emb is None:
                skipped += 1
                continue
            db.setdefault(name, []).append(emb)
            thumb_path = os.path.join(DB_DIR, f"{name}.jpg")
            if not os.path.exists(thumb_path):
                pil_img.save(thumb_path)
            enrolled += 1
        except Exception:
            skipped += 1
    save_db(db)
    return enrolled, skipped

def process_recorded_video(model, input_video_path, output_video_path, threshold, frame_stride=1):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return None, "Could not open uploaded video."
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return None, "Could not create output video."
    frame_no = 0
    processed_frames = 0
    detected_frames = 0
    total_known = 0
    total_unknown = 0
    progress = st.progress(0, text="Processing recorded video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % max(1, frame_stride) == 0:
            face_results = search_all_faces_in_frame(model, frame, threshold)
            if face_results:
                detected_frames += 1
                total_known += sum(1 for r in face_results if r["matched"])
                total_unknown += sum(1 for r in face_results if not r["matched"])
                annotated = draw_boxes_on_frame(frame.copy(), face_results)
                if face_results:
                    if any(not r["matched"] for r in face_results):
                        bar_color = (0, 30, 200)
                        bar_text = f"RED FLAG: {sum(1 for r in face_results if not r['matched'])} UNKNOWN FACE(S)"
                    else:
                        bar_color = (20, 160, 50)
                        bar_text = f"ALL CLEAR: {sum(1 for r in face_results if r['matched'])} FACE(S) RECOGNIZED"
                    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 32), bar_color, -1)
                    cv2.putText(annotated, bar_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(annotated)
                processed_frames += 1
            else:
                writer.write(frame)
        frame_no += 1
        if total_frames > 0:
            progress.progress(min(frame_no / total_frames, 1.0), text=f"Processing recorded video... {frame_no}/{total_frames} frames")
    cap.release()
    writer.release()
    progress.progress(1.0, text="Recorded video processing complete.")
    summary = {
        "frames_read": frame_no,
        "frames_processed": processed_frames,
        "frames_with_faces": detected_frames,
        "known_faces_total": total_known,
        "unknown_faces_total": total_unknown
    }
    return summary, None


class FaceRecognitionProcessor:
    RECOGNIZE_EVERY = 10

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_count = 0
        self._cached_results = []
        self.last_results = []
        self.has_unknown = False
        self.threshold = SIMILARITY_THRESHOLD
        self._model = load_model()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        with self._lock:
            self._frame_count += 1
            run_now = (self._frame_count % self.RECOGNIZE_EVERY == 0)
            threshold = self.threshold
        if run_now:
            results = search_all_faces_in_frame(self._model, img_bgr, threshold)
            with self._lock:
                self._cached_results = results
                self.last_results = results
                self.has_unknown = any(not r["matched"] for r in results)
        with self._lock:
            cached = self._cached_results[:]
        annotated = draw_boxes_on_frame(img_bgr.copy(), cached)
        if cached:
            if any(not r["matched"] for r in cached):
                bar_color = (0, 30, 200)
                bar_text = "!!! RED FLAG: UNRECOGNIZED FACE DETECTED !!!"
            else:
                bar_color = (20, 160, 50)
                bar_text = "ALL CLEAR: All faces recognized"
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 32), bar_color, -1)
            cv2.putText(annotated, bar_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root {
        --bg: #06111f; --bg-2: #0b1728;
        --panel: rgba(10,21,38,0.78); --panel-strong: rgba(15,27,46,0.92); --panel-soft: rgba(255,255,255,0.04);
        --text: #e8f1ff; --muted: #8fa8c7; --line: rgba(255,255,255,0.08);
        --cyan: #22d3ee; --cyan-2: #0891b2; --green: #22c55e; --red: #ef4444; --amber: #f59e0b;
        --shadow: 0 14px 40px rgba(0,0,0,0.35); --radius: 20px;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        color: var(--text);
        background:
            radial-gradient(circle at 12% 18%, rgba(34,211,238,0.10), transparent 24%),
            radial-gradient(circle at 88% 8%, rgba(59,130,246,0.11), transparent 22%),
            linear-gradient(180deg, #07111d 0%, #030711 100%);
    }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1460px; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8,16,30,0.98) 0%, rgba(10,20,36,0.96) 100%);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(12,23,41,0.96), rgba(9,17,31,0.96));
        border: 1px solid var(--line); border-radius: 18px; padding: 14px 16px; box-shadow: var(--shadow);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: var(--muted) !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: white !important; }
    .stButton > button, .stDownloadButton > button {
        width: 100%; border-radius: 14px; border: 1px solid rgba(34,211,238,0.20);
        background: linear-gradient(135deg, #0f766e, #164e63); color: white; font-weight: 700;
        padding: 0.78rem 1rem; box-shadow: 0 12px 24px rgba(8,145,178,0.18);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: rgba(34,211,238,0.42); transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] { background: rgba(255,255,255,0.03); }
    .stTextInput > div > div > input, .stNumberInput input, .stSelectbox > div > div,
    .stMultiSelect > div > div, .stTextArea textarea {
        background: rgba(255,255,255,0.04) !important; border: 1px solid var(--line) !important;
        color: var(--text) !important; border-radius: 14px !important;
    }
    .stSlider, .stFileUploader, .stDataFrame, .stAlert, .stExpander { border-radius: 18px !important; }
    .stFileUploader { background: rgba(255,255,255,0.03); border: 1px dashed rgba(34,211,238,0.20); padding: 0.35rem; }
    .streamlit-expanderHeader { background: rgba(255,255,255,0.03); border-radius: 14px; border: 1px solid var(--line); }
    div[data-baseweb="tab-list"] { gap: 0.75rem; background: transparent; margin-bottom: 1rem; }
    button[data-baseweb="tab"] {
        background: rgba(255,255,255,0.04); border: 1px solid var(--line); border-radius: 14px;
        color: var(--muted); padding: 0.75rem 1.1rem; font-weight: 700;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: white;
        background: linear-gradient(135deg, rgba(8,145,178,0.55), rgba(14,116,144,0.28));
        border: 1px solid rgba(34,211,238,0.28); box-shadow: 0 10px 24px rgba(34,211,238,0.12);
    }
    .hero-shell {
        position: relative; overflow: hidden; padding: 26px 28px; border-radius: 26px;
        background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(8,17,31,0.92)),
                    radial-gradient(circle at top right, rgba(34,211,238,0.25), transparent 30%);
        border: 1px solid rgba(255,255,255,0.08); box-shadow: var(--shadow); margin-bottom: 1rem;
    }
    .hero-shell::before {
        content: ''; position: absolute; inset: 0;
        background: radial-gradient(circle at 85% 20%, rgba(34,211,238,0.16), transparent 22%);
        pointer-events: none;
    }
    .badge {
        display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
        border-radius: 999px; background: rgba(34,211,238,0.10); color: #9be7f3;
        border: 1px solid rgba(34,211,238,0.20); font-size: 0.84rem; font-weight: 700; margin-bottom: 12px;
    }
    .hero-title { font-size: 2.4rem; font-weight: 800; line-height: 1.08; color: white; margin-bottom: 8px; letter-spacing: -0.03em; }
    .hero-subtitle { max-width: 78ch; color: var(--muted); font-size: 1rem; line-height: 1.7; }
    .section-card {
        background: linear-gradient(180deg, rgba(12,23,41,0.82), rgba(8,17,31,0.88));
        border: 1px solid var(--line); border-radius: 22px; padding: 18px 18px;
        box-shadow: var(--shadow); margin-bottom: 1rem;
    }
    .section-title { font-size: 1.08rem; font-weight: 800; color: white; margin-bottom: 4px; }
    .section-desc { color: var(--muted); font-size: 0.92rem; margin-bottom: 12px; }
    .status-banner {
        border-radius: 18px; padding: 16px 18px; color: white; font-weight: 800;
        margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.08);
    }
    .status-danger  { background: linear-gradient(135deg, rgba(127,29,29,0.95), rgba(239,68,68,0.95));  box-shadow: 0 12px 28px rgba(239,68,68,0.22); }
    .status-success { background: linear-gradient(135deg, rgba(20,83,45,0.95), rgba(34,197,94,0.95));   box-shadow: 0 12px 28px rgba(34,197,94,0.18); }
    .status-neutral { background: linear-gradient(135deg, rgba(8,47,73,0.95), rgba(14,116,144,0.95));   box-shadow: 0 12px 28px rgba(8,145,178,0.18); }
    .result-card { border-radius: 16px; padding: 14px 16px; margin-bottom: 10px; border: 1px solid transparent; }
    .result-known   { background: rgba(34,197,94,0.09);  border-color: rgba(34,197,94,0.25); }
    .result-unknown { background: rgba(239,68,68,0.10);  border-color: rgba(239,68,68,0.26); }
    .result-title { font-weight: 800; color: white; margin-bottom: 4px; }
    .result-meta  { color: #b7c7dc; font-size: 0.9rem; }
    .tip-box {
        border-radius: 16px; padding: 15px 16px; margin-bottom: 12px;
        background: rgba(255,255,255,0.03); border: 1px solid var(--line); color: var(--muted); line-height: 1.65;
    }
    .sidebar-note { color: var(--muted); font-size: 0.85rem; margin-bottom: 10px; }
    .empty-box {
        text-align: center; padding: 44px 22px; color: var(--muted); border-radius: 18px;
        background: rgba(255,255,255,0.03); border: 1px dashed rgba(255,255,255,0.10);
    }
    </style>
    """, unsafe_allow_html=True)


def hero():
    db = load_db()
    total_templates = sum(len(v) for v in db.values()) if db else 0
    st.markdown(f"""
    <div class="hero-shell">
      <div class="badge">🛡️ Facial Recognition Dashboard</div>
      <div class="hero-title">Face Recognition System</div>
      <div class="hero-subtitle">Identify faces across snapshots, recorded video and live webcam feeds.</div>
    </div>
    """, unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Enrolled People", len(db))
    m2.metric("Face Templates", total_templates)
    m3.metric("Default Threshold", SIMILARITY_THRESHOLD)
    m4.metric("Sampling Interval", FRAME_SAMPLE_INTERVAL)


def section_header(title, desc):
    st.markdown(f"""
    <div class="section-card">
      <div class="section-title">{title}</div>
      <div class="section-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def status_banner(kind, title, subtitle=""):
    cls = {"danger": "status-danger", "success": "status-success", "neutral": "status-neutral"}.get(kind, "status-neutral")
    body = f'<div style="font-weight:600;font-size:0.95rem;opacity:0.95;margin-top:4px">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="status-banner {cls}">{title}{body}</div>', unsafe_allow_html=True)


def result_card(index, result):
    if result["matched"]:
        st.markdown(
            f'<div class="result-card result-known">'
            f'<div class="result-title">Face {index} — {result["name"]}</div>'
            f'<div class="result-meta">Similarity: {result["similarity"]}% &nbsp;|&nbsp; Distance: {round(result["distance"], 4)}</div>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="result-card result-unknown">'
            f'<div class="result-title">Face {index} — UNKNOWN</div>'
            f'<div class="result-meta">Best attempt: {result["name"]} &nbsp;|&nbsp; Similarity: {result["similarity"]}%</div>'
            f'</div>', unsafe_allow_html=True)


def empty_box(icon, text):
    st.markdown(f'<div class="empty-box"><div style="font-size:2.2rem;margin-bottom:8px">{icon}</div><div>{text}</div></div>', unsafe_allow_html=True)


def init_state():
    if "surveillance_log" not in st.session_state:
        st.session_state.surveillance_log = []
    if "live_alerts" not in st.session_state:
        st.session_state.live_alerts = []


def add_surveillance_entry(mode, faces, known, unknown, status, names):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    entry = {"Time": ts, "Mode": mode, "Faces": faces, "Known": known, "Unknown": unknown, "Status": status, "Names": names}
    log = st.session_state.surveillance_log
    if not log or log[0]["Time"] != ts or log[0]["Mode"] != mode:
        st.session_state.surveillance_log = [entry] + log[:49]


inject_css()
init_state()
hero()
model = load_model()
db = load_db()

with st.sidebar:
    st.markdown('<div class="badge">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown("### System Settings")
    threshold = st.slider("Similarity Threshold", 0.1, 1.0, SIMILARITY_THRESHOLD, 0.05)
    st.caption("Higher threshold is stricter. Lower threshold is more permissive.")
    st.markdown("---")
    st.markdown("### Database Overview")
    st.metric("Enrolled People", len(db))
    st.metric("Total Templates", sum(len(v) for v in db.values()) if db else 0)
    if db:
        st.markdown("---")
        st.markdown("### Delete Identity")
        del_name = st.selectbox("Select person", list(db.keys()))
        if st.button("🗑️ Delete Person", type="secondary", key="delete_person_btn"):
            if delete_person(del_name):
                st.success(f"Deleted {del_name}")
                st.rerun()
    st.markdown("---")
    st.markdown("### Enrollment")
    with st.expander("Single Person Enroll", expanded=False):
        st.markdown('<div class="sidebar-note">Add one person with multiple photos for better embedding quality.</div>', unsafe_allow_html=True)
        person_name = st.text_input("Employee Name / ID", placeholder="e.g. EMP001 or JohnDoe", key="sidebar_person_name")
        uploaded_files = st.file_uploader("Upload Face Photos", type=["jpg","jpeg","png"], accept_multiple_files=True, key="enroll_uploader")
        if uploaded_files:
            st.caption(f"{len(uploaded_files)} images selected")
            preview_cols = st.columns(min(len(uploaded_files), 2))
            for i, f in enumerate(uploaded_files[:2]):
                with preview_cols[i]:
                    st.image(f, use_container_width=True, caption=f.name)
        if st.button("✅ Enroll Person", disabled=not (person_name and uploaded_files), key="sidebar_enroll_person_btn"):
            progress = st.progress(0)
            success_count = 0
            for i, file in enumerate(uploaded_files):
                pil_img = Image.open(file)
                bgr_img = pil_to_bgr(pil_img)
                ok, msg = enroll_person(model, person_name.strip(), bgr_img)
                if ok:
                    success_count += 1
                else:
                    st.warning(f"{file.name}: {msg}")
                progress.progress((i + 1) / len(uploaded_files))
            if success_count > 0:
                Image.open(uploaded_files[-1]).save(os.path.join(DB_DIR, f"{person_name.strip()}.jpg"))
                st.success(f"Enrolled {person_name} with {success_count} templates")
                st.rerun()
            else:
                st.error("Enrollment failed. Ensure photos contain a clear visible face.")

    with st.expander("Video Enroll", expanded=False):
        st.markdown('<div class="sidebar-note">Extract unique faces from a video and assign names in the gallery tab.</div>', unsafe_allow_html=True)
        video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov","mkv"], key="video_enroll")
        frame_interval = st.number_input("Sample every N frames", min_value=1, max_value=120, value=FRAME_SAMPLE_INTERVAL, key="sidebar_frame_interval")
        min_face_size = st.number_input("Min face size (px)", min_value=20, max_value=300, value=MIN_FACE_SIZE, key="sidebar_min_face_size")
        if video_file:
            st.video(video_file)
        if st.button("🎞️ Extract Faces from Video", disabled=not video_file, key="sidebar_extract_faces_btn"):
            suffix = os.path.splitext(video_file.name)[1] if video_file.name else ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name
            with st.spinner("Processing video..."):
                unique_faces, err = extract_faces_from_video(model, tmp_path, int(frame_interval), int(min_face_size))
            os.remove(tmp_path)
            if err:
                st.error(err)
            elif not unique_faces:
                st.warning("No faces detected.")
            else:
                st.session_state.video_faces = unique_faces
                st.success(f"Found {len(unique_faces)} unique faces. Review them in the gallery section below.")

    with st.expander("Bulk Folder Enroll", expanded=False):
        st.markdown('<div class="sidebar-note">Upload many files at once. Each filename becomes the identity name.</div>', unsafe_allow_html=True)
        folder_files = st.file_uploader("Upload Folder Photos", type=["jpg","jpeg","png"], accept_multiple_files=True, key="bulk_folder_uploader")
        if folder_files:
            st.caption(f"{len(folder_files)} images selected")
            preview_cols = st.columns(min(len(folder_files), 2))
            for i, f in enumerate(folder_files[:2]):
                with preview_cols[i]:
                    st.image(f, use_container_width=True, caption=f.name)
        if st.button("📁 Enroll Folder Photos", disabled=not folder_files, key="sidebar_bulk_enroll_btn"):
            with st.spinner("Enrolling folder images..."):
                enrolled, skipped = save_bulk_folder(model, folder_files)
            if enrolled > 0:
                st.success(f"Enrolled {enrolled} images. Skipped {skipped}.")
                st.rerun()
            else:
                st.error("No valid faces were enrolled from the uploaded folder.")


tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Camera Snapshot",
    "🎬 Recorded Video",
    "📡 Live Stream",
    "🖼️ Face Gallery"
])

with tab1:
    section_header("Camera Snapshot Verification", "Capture photo by clicking on Take Photo button")
    # st.markdown('<div class="tip-box"><b>Workflow:</b> Capture one image from the webcam, run recognition on all visible faces, then review a clear status banner, annotated frame, and individual face cards. Enable full candidate scores when you want a more diagnostic view.</div>', unsafe_allow_html=True)
    db_snap = load_db()
    if not db_snap:
        st.warning("No faces enrolled yet. Please enroll identities first using the sidebar.")
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        auto_scan = st.checkbox("Auto-scan on capture", value=True)
    with ctrl2:
        show_candidates = st.checkbox("Show all candidate scores", value=False)
    with ctrl3:
        st.caption(f"Database size: {len(db_snap)} people  |  Threshold: {threshold}")
    cam_col, res_col = st.columns([1.15, 1])
    with cam_col:
        st.markdown("#### 📷 Camera Feed")
        camera_image = st.camera_input("Capture frame to verify", key="snap_cam")
    with res_col:
        st.markdown("#### 🔍 Recognition Panel")
        if camera_image is None:
            empty_box("📷", "Waiting for a captured frame from your webcam.")
        else:
            run_scan = auto_scan or st.button("Scan Snapshot Now", type="primary", key="scan_now_btn")
            if run_scan:
                with st.spinner("Running face recognition..."):
                    pil_img = Image.open(camera_image)
                    bgr_img = pil_to_bgr(pil_img)
                    face_results = search_all_faces_in_frame(model, bgr_img, threshold)
                if not face_results:
                    st.info("No faces detected. Try better lighting or move closer to the camera.")
                else:
                    unknown_faces = [r for r in face_results if not r["matched"]]
                    known_faces   = [r for r in face_results if r["matched"]]
                    if unknown_faces:
                        status_banner("danger", f"🚨 RED FLAG — {len(unknown_faces)} unknown face(s)", "At least one visible face did not satisfy the recognition threshold.")
                    else:
                        status_banner("success", f"✅ ALL CLEAR — {len(known_faces)} recognized face(s)", "Every visible face matched an enrolled identity.")
                    annotated = annotate_frame_pil(pil_img, face_results)
                    st.image(annotated, caption="Annotated snapshot", use_container_width=True)
                    for i, r in enumerate(face_results, 1):
                        result_card(i, r)
                    if show_candidates and db_snap:
                        st.markdown("#### Candidate Score Breakdown")
                        all_detected = get_all_faces(model, bgr_img)
                        for i, (r, face) in enumerate(zip(face_results, all_detected), 1):
                            with st.expander(f"Face {i} — {r['name']}"):
                                emb = normalize_embedding(face.normed_embedding)
                                scores = [{"Person": p, "Similarity": round((1 - min(cosine_distance(emb, e) for e in embs)) * 100, 2), "Decision": "MATCH" if min(cosine_distance(emb, e) for e in embs) <= threshold else "REVIEW"} for p, embs in db_snap.items()]
                                scores.sort(key=lambda x: -x["Similarity"])
                                st.dataframe(pd.DataFrame(scores), hide_index=True, use_container_width=True)
                    add_surveillance_entry(
                        mode="Snapshot", faces=len(face_results),
                        known=len(known_faces), unknown=len(unknown_faces),
                        status="RED FLAG" if unknown_faces else "Clear",
                        names=", ".join(r["name"] for r in face_results)
                    )

with tab2:
    section_header("Recorded Video Analysis", "Upload a recorded video, process frames in batch, generate an annotated result video, and review the video by downloading it")
    st.markdown('<div class="tip-box"><b>Lower the frames the more time will be taken for video processing.</div>', unsafe_allow_html=True)
    db_video = load_db()
    if not db_video:
        st.warning("No faces enrolled yet. Please enroll identities first using the sidebar.")
    rvc1, rvc2, rvc3 = st.columns([1, 1, 2])
    with rvc1:
        recorded_threshold = st.slider("Recorded video threshold", 0.1, 1.0, threshold, 0.05, key="recorded_video_thresh")
    with rvc2:
        frame_stride = st.selectbox("Process every Nth frame", options=[1, 2, 3, 5], index=0, help="Higher values are faster but may miss brief appearances.")
    # with rvc3:
    #     st.caption(f"Database size: {len(db_video)} people  |  Lower stride is slower but more thorough")
    recorded_video_file = st.file_uploader("Upload Recorded Video", type=["mp4","avi","mov","mkv"], key="recorded_video_uploader")
    if recorded_video_file is not None:
        st.video(recorded_video_file)
    if st.button("▶️ Run Detection on Recorded Video", disabled=recorded_video_file is None, key="run_recorded_video_detection"):
        input_suffix = os.path.splitext(recorded_video_file.name)[1] if recorded_video_file.name else ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp_in:
            tmp_in.write(recorded_video_file.read())
            input_path = tmp_in.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
            output_path = tmp_out.name
        with st.spinner("Processing recorded video for face detection..."):
            summary, err = process_recorded_video(
                model=model, input_video_path=input_path,
                output_video_path=output_path,
                threshold=recorded_threshold, frame_stride=int(frame_stride)
            )
        if err:
            st.error(err)
            if os.path.exists(input_path): os.remove(input_path)
            if os.path.exists(output_path): os.remove(output_path)
        else:
            with open(output_path, "rb") as f:
                video_bytes = f.read()
            st.session_state.processed_recorded_video_bytes = video_bytes
            st.session_state.processed_recorded_video_name = f"processed_{os.path.splitext(recorded_video_file.name)[0]}.mp4"
            st.session_state.processed_recorded_video_summary = summary
            add_surveillance_entry(
                mode="Recorded Video",
                faces=summary["known_faces_total"] + summary["unknown_faces_total"],
                known=summary["known_faces_total"], unknown=summary["unknown_faces_total"],
                status="RED FLAG" if summary["unknown_faces_total"] > 0 else "Clear",
                names=f"Frames with faces: {summary['frames_with_faces']}"
            )
            os.remove(input_path)
            os.remove(output_path)
            st.success("Recorded video processed successfully.")
    if "processed_recorded_video_bytes" in st.session_state:
        st.markdown("#### 📊 Processed Output")
        summary = st.session_state.get("processed_recorded_video_summary", {})
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Frames Read",        summary.get("frames_read", 0))
        s2.metric("Frames Processed",   summary.get("frames_processed", 0))
        s3.metric("Frames With Faces",  summary.get("frames_with_faces", 0))
        s4.metric("Known Faces",        summary.get("known_faces_total", 0))
        s5.metric("Unknown Faces",      summary.get("unknown_faces_total", 0))
        if summary.get("unknown_faces_total", 0) > 0:
            status_banner("danger", "⚠️ Review recommended", "The processed video contains one or more unknown-face detections.")
        else:
            status_banner("success", "✅ Processing clear", "No unknown faces were detected in the processed output video.")
        st.video(st.session_state.processed_recorded_video_bytes)
        st.download_button(
            label="⬇️ Download Processed Video",
            data=st.session_state.processed_recorded_video_bytes,
            file_name=st.session_state.get("processed_recorded_video_name", "processed_video.mp4"),
            mime="video/mp4",
            key="download_processed_video_btn"
        )

with tab3:
    section_header("Live Stream Surveillance", "Run continuous webcam monitoring through WebRTC, monitor red-flag activity, and inspect the live detection panel with cleaner operational controls.")
    st.markdown('<div class="tip-box"><b>Live mode:</b> Recognition is executed every N frames for CPU efficiency. Lower recognition intervals increase responsiveness but also increase compute cost. Unknown faces immediately trigger a red-flag event and are added to the alert log.</div>', unsafe_allow_html=True)
    db_live = load_db()
    if not db_live:
        st.warning("No faces enrolled yet. Please enroll identities first using the sidebar.")
    lv_col1, lv_col2, lv_col3 = st.columns([1, 1, 2])
    with lv_col1:
        recog_freq = st.selectbox("Recognition frequency", options=[5, 10, 15, 20, 30], index=1, help="Lower values scan more often.")
    with lv_col2:
        live_threshold = st.slider("Live threshold", 0.1, 1.0, threshold, 0.05, key="live_thresh")
    with lv_col3:
        st.caption(f"Database size: {len(db_live)} people  |  Lower frequency values use more CPU")
    stream_col, info_col = st.columns([1.6, 1])
    with stream_col:
        ctx = webrtc_streamer(
            key="face-recognition-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=FaceRecognitionProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx.video_processor:
            ctx.video_processor.threshold = live_threshold
            ctx.video_processor.RECOGNIZE_EVERY = recog_freq
    with info_col:
        st.markdown("#### 📡 Live Status")
        if not ctx.state.playing:
            empty_box("📡", "Click START in the WebRTC player to begin live surveillance.")
        else:
            status_banner("neutral", "🔴 Live stream active", "The recognition engine is currently monitoring the webcam feed.")
            if ctx.video_processor:
                last_results = ctx.video_processor.last_results
                has_unknown  = ctx.video_processor.has_unknown
                if not last_results:
                    st.info("No faces in frame yet.")
                else:
                    unknown_count = sum(1 for r in last_results if not r["matched"])
                    known_count   = sum(1 for r in last_results if r["matched"])
                    if has_unknown:
                        status_banner("danger", f"🚨 RED FLAG — {unknown_count} unknown face(s)")
                    else:
                        status_banner("success", f"✅ ALL CLEAR — {known_count} recognized face(s)")
                    for i, r in enumerate(last_results, 1):
                        result_card(i, r)
                    if has_unknown:
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        alert = {"Time": ts, "Unknown Faces": unknown_count, "Known Faces": known_count,
                                 "Details": ", ".join(r["name"] if r["matched"] else "UNKNOWN" for r in last_results)}
                        alerts = st.session_state.live_alerts
                        if not alerts or alerts[0]["Time"] != ts:
                            st.session_state.live_alerts = [alert] + alerts[:29]
        st.markdown("#### 🚨 Red Flag Alert Log")
        log_c1, log_c2 = st.columns([5, 1])
        with log_c1:
            if not st.session_state.live_alerts:
                empty_box("🟢", "No red-flag alerts yet. Unknown-face events will appear here during live monitoring.")
            else:
                st.dataframe(pd.DataFrame(st.session_state.live_alerts), use_container_width=True, hide_index=True)
        with log_c2:
            if st.button("Clear Alerts", key="clear_live_alerts_btn"):
                st.session_state.live_alerts = []
                st.rerun()
    st.markdown("#### 📋 Unified Surveillance Log")
    sl_c1, sl_c2 = st.columns([5, 1])
    with sl_c1:
        if not st.session_state.surveillance_log:
            empty_box("📋", "Snapshot and recorded-video events will be tracked here once scans are executed.")
        else:
            st.dataframe(pd.DataFrame(st.session_state.surveillance_log), use_container_width=True, hide_index=True)
    with sl_c2:
        if st.button("Clear Log", key="clear_surveillance_log_btn"):
            st.session_state.surveillance_log = []
            st.rerun()

with tab4:
    section_header("Face Gallery Review", "Browse enrolled identities")
    db_view = load_db()
    if not db_view:
        empty_box("🖼️", "No identities enrolled yet. Use the sidebar enrollment workflows to populate the gallery.")
    else:
        cols = st.columns(4)
        for i, name in enumerate(db_view.keys()):
            with cols[i % 4]:
                with st.container(border=False):
                    thumb = os.path.join(DB_DIR, f"{name}.jpg")
                    if os.path.exists(thumb):
                        st.image(thumb, caption=name, use_container_width=True)
                    else:
                        empty_box("👤", name)
                    st.caption(f"{len(db_view[name])} templates")

    if "video_faces" in st.session_state and st.session_state.video_faces:
        st.markdown("---")
        section_header("Video Enrollment Review", f"Assign labels to {len(st.session_state.video_faces)} extracted face clusters and save them directly into the database.")
        faces_data = st.session_state.video_faces
        names_input = []
        cols_per_row = 4
        for start in range(0, len(faces_data), cols_per_row):
            row_faces = faces_data[start:start + cols_per_row]
            cols = st.columns(cols_per_row)
            for idx_in_row, face_dict in enumerate(row_faces):
                global_idx = start + idx_in_row
                with cols[idx_in_row]:
                    st.image(cv2.cvtColor(face_dict["crop"], cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Face {global_idx + 1} | Frame {face_dict['frame_no']}")
                    name_val = st.text_input(f"Name for Face {global_idx + 1}", key=f"vname_{global_idx}", placeholder="Enter name or leave blank")
                    names_input.append((global_idx, name_val))
        action_c1, action_c2 = st.columns(2)
        with action_c1:
            if st.button("💾 Save Named Faces to Database", key="save_video_faces_btn"):
                saved, skipped = save_named_faces(faces_data, names_input)
                if saved:
                    st.success(f"Saved {saved} faces to database. Skipped {skipped} unlabeled clusters.")
                    del st.session_state.video_faces
                    st.rerun()
                else:
                    st.warning("No names provided.")
        with action_c2:
            if st.button("🗑️ Clear Review Queue", key="clear_video_faces_btn"):
                del st.session_state.video_faces
                st.rerun()