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


# WebRTC STUN config for NAT traversal
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


os.makedirs(DB_DIR, exist_ok=True)


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
        cv2.putText(frame_bgr, label, (x1 + 3, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if not r["matched"]:
            cv2.putText(frame_bgr, "! RED FLAG", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 50, 235), 2, cv2.LINE_AA)
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
                add_to_cluster(
                    clusters,
                    normalize_embedding(face.normed_embedding),
                    crop,
                    frame_no,
                    [x1, y1, x2, y2]
                )
            sampled += 1
            if total_frames > 0:
                progress_bar.progress(
                    min(frame_no / total_frames, 1.0),
                    text=f"Scanning... frame {frame_no}/{total_frames} | Unique faces: {len(clusters)}"
                )
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
            cv2.putText(annotated, bar_text, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


st.set_page_config(page_title="Face Recognition System", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@keyframes pulse-red {
    0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.7); }
    70%  { box-shadow: 0 0 0 14px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
.red-flag-banner {
    background: linear-gradient(135deg, #7f1d1d, #ef4444);
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    font-size: 1.2rem;
    font-weight: 800;
    text-align: center;
    animation: pulse-red 1.4s infinite;
    margin-bottom: 14px;
    letter-spacing: 0.03em;
}
.green-clear-banner {
    background: linear-gradient(135deg, #14532d, #22c55e);
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    font-size: 1.2rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 14px;
    letter-spacing: 0.03em;
}
.face-card-unknown {
    border: 2px solid #ef4444;
    border-radius: 10px;
    padding: 10px 14px;
    background: rgba(239,68,68,0.07);
    margin-bottom: 8px;
}
.face-card-known {
    border: 2px solid #22c55e;
    border-radius: 10px;
    padding: 10px 14px;
    background: rgba(34,197,94,0.07);
    margin-bottom: 8px;
}
.cam-tip {
    background: #0f172a;
    color: #94a3b8;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 0.87rem;
    margin-bottom: 16px;
    border-left: 3px solid #3b82f6;
}
.live-tip {
    background: #0f172a;
    color: #94a3b8;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 0.87rem;
    margin-bottom: 16px;
    border-left: 3px solid #ef4444;
}
.sidebar-section-note {
    font-size: 0.83rem;
    color: #94a3b8;
    margin-top: -6px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("Face Recognition System")
st.divider()

model = load_model()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Similarity Threshold", 0.1, 1.0, SIMILARITY_THRESHOLD, 0.05)

    st.divider()
    db = load_db()
    st.header("Database Stats")
    st.metric("Enrolled People", len(db))
    st.metric("Total Face Templates", sum(len(v) for v in db.values()))

    if db:
        st.divider()
        st.subheader("Delete Person")
        del_name = st.selectbox("Select person to delete", list(db.keys()))
        if st.button("Delete", type="secondary", key="delete_person_btn"):
            if delete_person(del_name):
                st.success(f"Deleted {del_name}")
                st.rerun()

    st.divider()
    st.header("Enrollment")

    with st.expander("Enroll Face", expanded=False):
        st.markdown('<div class="sidebar-section-note">Add one person using multiple face photos.</div>', unsafe_allow_html=True)
        person_name = st.text_input(
            "Employee Name / ID",
            placeholder="e.g. John_Doe or EMP001",
            key="sidebar_person_name"
        )
        uploaded_files = st.file_uploader(
            "Upload Face Photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="enroll_uploader"
        )

        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")
            preview_cols = st.columns(min(len(uploaded_files), 2))
            for i, f in enumerate(uploaded_files[:2]):
                with preview_cols[i]:
                    st.image(f, use_column_width=True, caption=f.name)

        if st.button("Enroll Person", disabled=not (person_name and uploaded_files), key="sidebar_enroll_person_btn"):
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
                st.success(f"Enrolled {person_name} with {success_count} template(s)")
                st.rerun()
            else:
                st.error("Enrollment failed. Ensure photos contain a clear visible face.")

    with st.expander("Video Enroll", expanded=False):
        st.markdown('<div class="sidebar-section-note">Extract unique faces from a video, then assign names.</div>', unsafe_allow_html=True)
        video_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_enroll"
        )
        frame_interval = st.number_input(
            "Sample every N frames",
            min_value=1,
            max_value=120,
            value=FRAME_SAMPLE_INTERVAL,
            key="sidebar_frame_interval"
        )
        min_face_size = st.number_input(
            "Min face size (px)",
            min_value=20,
            max_value=300,
            value=MIN_FACE_SIZE,
            key="sidebar_min_face_size"
        )

        if video_file:
            st.video(video_file)

        if st.button("Extract Faces from Video", disabled=not video_file, key="sidebar_extract_faces_btn"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
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
                st.session_state["video_faces"] = unique_faces
                st.success(f"Found {len(unique_faces)} unique face(s). Review them in the main page below.")

    with st.expander("Bulk Folder Enroll", expanded=False):
        st.markdown('<div class="sidebar-section-note">Upload multiple photos; filename becomes the person name.</div>', unsafe_allow_html=True)
        folder_files = st.file_uploader(
            "Upload Folder Photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="bulk_folder_uploader"
        )

        if folder_files:
            st.caption(f"{len(folder_files)} file(s) selected")
            preview_cols = st.columns(min(len(folder_files), 2))
            for i, f in enumerate(folder_files[:2]):
                with preview_cols[i]:
                    st.image(f, use_column_width=True, caption=f.name)

        if st.button("Enroll Folder Photos", disabled=not folder_files, key="sidebar_bulk_enroll_btn"):
            with st.spinner("Enrolling folder images..."):
                enrolled, skipped = save_bulk_folder(model, folder_files)

            if enrolled > 0:
                st.success(f"Enrolled {enrolled} image(s). Skipped {skipped}.")
                st.rerun()
            else:
                st.error("No valid faces were enrolled from the uploaded folder.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Recognize Face",
    "View Saved Faces",
    "📸 Camera Snapshot",
    "🔴 Live Stream"
])

with tab1:
    st.header("Recognize Face")
    c1, c2 = st.columns([1, 1.5])

    with c1:
        query_file = st.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"], key="query_uploader")
        if query_file:
            st.image(query_file, caption="Query Image", use_column_width=True)
        search_btn = st.button("Search", disabled=not query_file, key="search_btn")

    with c2:
        if query_file and search_btn:
            with st.spinner("Analyzing face..."):
                pil_img = Image.open(query_file)
                bgr_img = pil_to_bgr(pil_img)
                results, error = search_face(model, bgr_img)

            if error:
                st.error(error)
            else:
                best_name, best_dist = results[0]
                best_similarity = round((1 - best_dist) * 100, 2)

                if best_dist <= threshold:
                    st.markdown(
                        f'<div style="background:#14532d;color:white;padding:12px 16px;border-radius:10px;">'
                        f'<b>✅ MATCH FOUND</b><br>{best_name} — {best_similarity}% similarity</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div style="background:#7f1d1d;color:white;padding:12px 16px;border-radius:10px;">'
                        f'<b>❌ NO MATCH</b><br>Closest: {best_name} ({best_similarity}%)</div>',
                        unsafe_allow_html=True
                    )

                st.divider()
                st.subheader("Top Candidates")
                for name, dist in results[:5]:
                    sim = round((1 - dist) * 100, 2)
                    icon = "✅" if dist <= threshold else "❌"
                    st.write(f"{icon} **{name}** — {sim}%")

with tab2:
    st.header("View Saved Faces")
    db_view = load_db()

    if not db_view:
        st.info("No faces enrolled yet.")
    else:
        cols = st.columns(4)
        for i, name in enumerate(db_view.keys()):
            with cols[i % 4]:
                thumb = os.path.join(DB_DIR, f"{name}.jpg")
                if os.path.exists(thumb):
                    st.image(thumb, caption=name, use_column_width=True)
                else:
                    st.write(f"👤 {name}")
                st.caption(f"{len(db_view[name])} template(s)")

    if "video_faces" in st.session_state and st.session_state["video_faces"]:
        faces_data = st.session_state["video_faces"]
        st.divider()
        st.subheader(f"Assign Names to {len(faces_data)} Detected Face(s) from Video")

        names_input = []
        cols_per_row = 4
        for start in range(0, len(faces_data), cols_per_row):
            row_faces = faces_data[start:start + cols_per_row]
            cols = st.columns(cols_per_row)
            for idx_in_row, face_dict in enumerate(row_faces):
                global_idx = start + idx_in_row
                with cols[idx_in_row]:
                    st.image(
                        cv2.cvtColor(face_dict["crop"], cv2.COLOR_BGR2RGB),
                        use_column_width=True,
                        caption=f"Face {global_idx + 1} | Frame {face_dict['frame_no']}"
                    )
                    name_val = st.text_input(
                        f"Name for Face {global_idx + 1}",
                        key=f"vname_{global_idx}",
                        placeholder="Enter name or leave blank"
                    )
                    names_input.append((global_idx, name_val))

        action_c1, action_c2 = st.columns(2)
        with action_c1:
            if st.button("Save Named Faces to Database", key="save_video_faces_btn"):
                saved, skipped = save_named_faces(faces_data, names_input)
                if saved:
                    st.success(f"Saved {saved} face(s) to database! ({skipped} skipped)")
                    del st.session_state["video_faces"]
                    st.rerun()
                else:
                    st.warning("No names provided.")
        with action_c2:
            if st.button("Clear & Re-extract", key="clear_video_faces_btn"):
                del st.session_state["video_faces"]
                st.rerun()

with tab3:
    st.header("📸 Camera Snapshot")

    st.markdown("""
    <div class="cam-tip">
    📌 <b>How it works:</b> Capture a single snapshot from your webcam → all faces are detected and matched
    against the enrolled database → <span style="color:#ef4444;font-weight:700;">🚨 RED FLAG</span>
    for any unrecognized face, <span style="color:#22c55e;font-weight:700;">✅ green clearance</span> for enrolled faces.
    </div>
    """, unsafe_allow_html=True)

    db_snap = load_db()
    if not db_snap:
        st.warning("⚠️ No faces enrolled yet. Please enroll faces first using the sidebar enrollment options.")

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
    with ctrl1:
        auto_scan = st.checkbox("🔄 Auto-Scan on Capture", value=True)
    with ctrl2:
        show_candidates = st.checkbox("📋 Show All Scores", value=False)
    with ctrl3:
        st.caption(f"📊 DB: **{len(db_snap)} people enrolled**  |  Threshold: **{threshold}**")

    st.divider()
    cam_col, res_col = st.columns([1.1, 0.9])

    with cam_col:
        st.subheader("📷 Camera")
        camera_image = st.camera_input("Capture frame to verify", key="snap_cam")

    with res_col:
        st.subheader("🔍 Results")
        if camera_image is None:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#64748b;">
                <div style="font-size:3rem;">📸</div>
                <p style="margin-top:12px;">Waiting for camera frame…</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            run_scan = auto_scan or st.button("🔎 Scan Now", type="primary", key="scan_now_btn")
            if run_scan:
                with st.spinner("🧠 Running face recognition…"):
                    pil_img = Image.open(camera_image)
                    bgr_img = pil_to_bgr(pil_img)
                    face_results = search_all_faces_in_frame(model, bgr_img, threshold)

                if not face_results:
                    st.info("👀 No faces detected. Try better lighting or move closer.")
                else:
                    unknown_faces = [r for r in face_results if not r["matched"]]
                    known_faces = [r for r in face_results if r["matched"]]

                    if unknown_faces:
                        st.markdown(
                            f'<div class="red-flag-banner">🚨 RED FLAG — {len(unknown_faces)} UNRECOGNIZED FACE(S) DETECTED!</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="green-clear-banner">✅ ALL CLEAR — {len(known_faces)} RECOGNIZED FACE(S)</div>',
                            unsafe_allow_html=True
                        )

                    annotated = annotate_frame_pil(pil_img, face_results)
                    st.image(annotated, caption="Annotated Frame", use_column_width=True)
                    st.markdown("---")

                    for i, r in enumerate(face_results, 1):
                        if r["matched"]:
                            st.markdown(
                                f'<div class="face-card-known"><b>Face #{i}</b> &nbsp;|&nbsp; ✅ <b>{r["name"]}</b>'
                                f'<br><small>Similarity: {r["similarity"]}%  Distance: {round(r["distance"],4)}</small></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="face-card-unknown"><b>Face #{i}</b> &nbsp;|&nbsp; 🚨 <b>UNKNOWN — NOT IN DATABASE</b>'
                                f'<br><small>Best attempt: {r["name"]} @ {r["similarity"]}%</small></div>',
                                unsafe_allow_html=True
                            )

                    if show_candidates and db_snap:
                        st.divider()
                        st.subheader("📋 Full Candidate Scores")
                        all_detected = get_all_faces(model, bgr_img)
                        for i, (r, face) in enumerate(zip(face_results, all_detected), 1):
                            with st.expander(f"Face #{i} — {r['name']}"):
                                emb = normalize_embedding(face.normed_embedding)
                                scores = [{
                                    "Person": p,
                                    "Similarity (%)": round((1 - min(cosine_distance(emb, e) for e in embs)) * 100, 2),
                                    "Match": "✅" if min(cosine_distance(emb, e) for e in embs) <= threshold else "❌"
                                } for p, embs in db_snap.items()]
                                scores.sort(key=lambda x: -x["Similarity (%)"])
                                st.dataframe(pd.DataFrame(scores), hide_index=True, use_container_width=True)

                    if "surveillance_log" not in st.session_state:
                        st.session_state["surveillance_log"] = []

                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    entry = {
                        "Time": ts,
                        "Mode": "Snapshot",
                        "Faces": len(face_results),
                        "Known": len(known_faces),
                        "Unknown": len(unknown_faces),
                        "Status": "🚨 RED FLAG" if unknown_faces else "✅ Clear",
                        "Names": ", ".join(r["name"] for r in face_results)
                    }
                    log = st.session_state["surveillance_log"]
                    if not log or log[0]["Time"] != ts:
                        st.session_state["surveillance_log"] = [entry] + log[:49]

with tab4:
    st.header("🔴 Live Stream Face Surveillance")

    st.markdown("""
    <div class="live-tip">
    🔴 <b>True Real-Time Mode</b> — This tab streams your webcam live through the browser via WebRTC.
    Every frame is analyzed by InsightFace in the backend (recognition runs every ~10 frames for CPU performance).
    <br><br>
    • <span style="color:#22c55e;font-weight:600;">Green box + name</span> = Face recognized in database<br>
    • <span style="color:#ef4444;font-weight:600;">Red box + UNKNOWN</span> = Face NOT in database → RED FLAG raised<br>
    • A status bar appears at the top of the live feed showing overall frame status.<br><br>
    <b>Requires:</b> <code>pip install streamlit-webrtc</code>
    </div>
    """, unsafe_allow_html=True)

    db_live = load_db()
    if not db_live:
        st.warning("⚠️ No faces enrolled yet. Please enroll faces first using the sidebar enrollment options.")

    st.divider()

    lv_col1, lv_col2, lv_col3 = st.columns([1, 1, 2])
    with lv_col1:
        recog_freq = st.selectbox(
            "Recognition frequency",
            options=[5, 10, 15, 20, 30],
            index=1,
            help="Run face recognition every N frames. Lower = more accurate but slower."
        )
    with lv_col2:
        live_threshold = st.slider("Live Threshold", 0.1, 1.0, threshold, 0.05, key="live_thresh")
    with lv_col3:
        st.caption(f"📊 DB: **{len(db_live)} people enrolled**  |  Higher frequency = more CPU usage")

    st.divider()

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
        st.subheader("📊 Live Status")

        if not ctx.state.playing:
            st.markdown("""
            <div style="text-align:center;padding:40px 16px;color:#64748b;">
                <div style="font-size:2.5rem;">🎥</div>
                <p style="margin-top:10px;">Click <b>START</b> in the video player<br>to begin live surveillance.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("🟢 Live stream ACTIVE")

            if ctx.video_processor:
                last_results = ctx.video_processor.last_results
                has_unknown = ctx.video_processor.has_unknown

                if not last_results:
                    st.info("No faces in frame yet…")
                else:
                    unknown_count = sum(1 for r in last_results if not r["matched"])
                    known_count = sum(1 for r in last_results if r["matched"])

                    if has_unknown:
                        st.markdown(
                            f'<div class="red-flag-banner">🚨 RED FLAG!<br>{unknown_count} UNKNOWN FACE(S)</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="green-clear-banner">✅ ALL CLEAR<br>{known_count} recognized</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown("**Detected Faces:**")
                    for i, r in enumerate(last_results, 1):
                        if r["matched"]:
                            st.markdown(
                                f'<div class="face-card-known"><b>#{i} ✅ {r["name"]}</b><br>'
                                f'<small>{r["similarity"]}% similarity</small></div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="face-card-unknown"><b>#{i} 🚨 UNKNOWN</b><br>'
                                f'<small>Best: {r["name"]} @ {r["similarity"]}%</small></div>',
                                unsafe_allow_html=True
                            )

                    if has_unknown:
                        if "live_alerts" not in st.session_state:
                            st.session_state["live_alerts"] = []

                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        alert = {
                            "Time": ts,
                            "Unknown Faces": unknown_count,
                            "Known Faces": known_count,
                            "Details": ", ".join(
                                r["name"] if r["matched"] else "UNKNOWN"
                                for r in last_results
                            )
                        }
                        alerts = st.session_state["live_alerts"]
                        if not alerts or alerts[0]["Time"] != ts:
                            st.session_state["live_alerts"] = [alert] + alerts[:29]

    st.divider()
    st.subheader("🚨 Red Flag Alert Log")
    if "live_alerts" not in st.session_state:
        st.session_state["live_alerts"] = []

    log_c1, log_c2 = st.columns([5, 1])
    with log_c1:
        if not st.session_state["live_alerts"]:
            st.caption("No red flag events yet. Alerts will appear here when unknown faces are detected in live stream.")
        else:
            st.dataframe(pd.DataFrame(st.session_state["live_alerts"]), use_container_width=True, hide_index=True)
    with log_c2:
        if st.button("🗑️ Clear Alerts", key="clear_live_alerts_btn"):
            st.session_state["live_alerts"] = []
            st.rerun()

    st.divider()
    st.subheader("📝 Full Surveillance Log (Snapshot + Live)")
    if "surveillance_log" not in st.session_state:
        st.session_state["surveillance_log"] = []

    sl_c1, sl_c2 = st.columns([5, 1])
    with sl_c1:
        if not st.session_state["surveillance_log"]:
            st.caption("No snapshot scans yet.")
        else:
            st.dataframe(pd.DataFrame(st.session_state["surveillance_log"]), use_container_width=True, hide_index=True)
    with sl_c2:
        if st.button("🗑️ Clear Log", key="clear_surveillance_log_btn"):
            st.session_state["surveillance_log"] = []
            st.rerun()