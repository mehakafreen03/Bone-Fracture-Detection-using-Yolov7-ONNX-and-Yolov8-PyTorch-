# app.py — YOLOv7 (ONNX) + YOLOv8 (.pt), single-page compare, collapsible settings
import os, io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO          # YOLOv8 offline (.pt)
import onnxruntime as ort              # YOLOv7 offline (.onnx)

GREEN = (0, 200, 0)
THICK = 2

st.set_page_config(page_title="Bone Fracture Detection — YOLOv7 vs YOLOv8", layout="wide")

# --------- Small CSS polish ---------
st.markdown("""
<style>
    .badge { display:inline-block; padding:4px 10px; border-radius:12px; font-weight:600; margin-left:6px; }
    .ok { background:#e8fff0; color:#0a7a2a; border:1px solid #b9ebc9; }
    .warn { background:#fff3f3; color:#b00020; border:1px solid #ffd0d0; }
    .subtle { color:#6b7280; }
    .spacer { height: 6px; }
    .panel { padding:12px 14px; border:1px solid #e5e7eb; border-radius:12px; background:#fafafa; }
</style>
""", unsafe_allow_html=True)

st.title("🦴 Bone Fracture Detection — YOLOv7 vs YOLOv8")

# ======================= Utils =======================
def pil_to_cv2(pil_img: Image.Image):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img: np.ndarray):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def image_bytes(pil_img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def draw_boxes(img_bgr, boxes, min_area=0):
    H, W = img_bgr.shape[:2]
    for (x1, y1, x2, y2, conf) in boxes:
        area = max(0, int(x2 - x1)) * max(0, int(y2 - y1))
        if area < min_area:
            continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), GREEN, THICK)
        cv2.putText(img_bgr, f"{conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    return img_bgr

def filter_by_area(boxes, img_hw, min_area_ratio):
    if min_area_ratio <= 0:
        return boxes
    H, W = img_hw
    cutoff = H * W * float(min_area_ratio)
    out = []
    for (x1, y1, x2, y2, conf) in boxes:
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        if area >= cutoff:
            out.append((x1, y1, x2, y2, conf))
    return out

def decide_label(boxes):
    return "Fractured" if len(boxes) > 0 else "Not Fractured"

def max_conf(boxes):
    return max((b[4] for b in boxes), default=0.0)

# ======================= YOLOv8 (.pt) =======================
@st.cache_resource(show_spinner=False)
def load_yolov8(weight_path: str):
    return YOLO(weight_path)

def run_yolov8(model: YOLO, pil_img: Image.Image, conf=0.25):
    res = model.predict(source=np.array(pil_img), conf=conf, verbose=False)
    boxes = []
    for r in res:
        if getattr(r, "boxes", None) is None:
            continue
        for b in r.boxes:
            xyxy = b.xyxy.cpu().numpy()[0]
            score = float(b.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = xyxy
            boxes.append((x1, y1, x2, y2, score))
    return boxes

# ======================= YOLOv7 (.onnx) =======================
def get_onnx_input_hw(sess: ort.InferenceSession):
    ishape = sess.get_inputs()[0].shape  # [N, C, H, W]
    H = ishape[2] if isinstance(ishape[2], int) else None
    W = ishape[3] if isinstance(ishape[3], int) else None
    return H, W

def letterbox_hw(im, new_hw=(1280, 1280)):
    target_h, target_w = new_hw
    h0, w0 = im.shape[:2]
    r = min(target_h / h0, target_w / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))  # (w, h)
    dw, dh = target_w - new_unpad[0], target_h - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    im = cv2.resize(im, (new_unpad[0], new_unpad[1]), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (dw, dh)

def nms(boxes, confs, iou_thres=0.45, conf_thres=0.25):
    idxs = np.where(confs >= conf_thres)[0]
    boxes, confs = boxes[idxs], confs[idxs]
    if boxes.size == 0:
        return boxes, confs
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return boxes[keep], confs[keep]

@st.cache_resource(show_spinner=False)
def load_yolov7_onnx(onnx_path: str):
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

def _run_y7_once(sess: ort.InferenceSession, pil_img: Image.Image, size_hw, conf=0.25, iou=0.45):
    im0 = pil_to_cv2(pil_img)
    im, r, (dw, dh) = letterbox_hw(im0, size_hw)

    im_in = im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    im_in = np.expand_dims(im_in, 0)

    out = sess.run(None, {sess.get_inputs()[0].name: im_in})[0]
    if out.ndim == 3:
        out = out[0]
    if out.size == 0:
        return []

    xywh = out[:, :4]
    obj = out[:, 4]
    cls = out[:, 5:] if out.shape[1] > 5 else np.zeros((out.shape[0], 1), dtype=np.float32)
    cls_conf = cls.max(axis=1) if cls.size else np.ones_like(obj)
    confs = obj * cls_conf

    x, y, w, h = xywh.T
    x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    boxes, confs = nms(boxes, confs, iou, conf)
    if boxes.size == 0:
        return []

    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= r
    H0, W0 = im0.shape[:2]
    boxes = np.clip(boxes, 0, [W0 - 1, H0 - 1, W0 - 1, H0 - 1])
    return [(bx[0], bx[1], bx[2], bx[3], float(cf)) for bx, cf in zip(boxes, confs)]

def run_yolov7_safe(sess: ort.InferenceSession, pil_img: Image.Image, conf=0.25, iou=0.45):
    H_in, W_in = get_onnx_input_hw(sess)
    candidates = []
    if H_in and W_in:
        candidates.append((H_in, W_in))
    candidates += [(1280, 1280), (640, 640), (960, 960)]
    last_err = None
    for (h, w) in candidates:
        try:
            return _run_y7_once(sess, pil_img, (h, w), conf, iou)
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return []

# ======================= Collapsible Settings =======================
with st.expander("⚙️ Settings", expanded=False):
    st.write("Model paths, thresholds & post-processing")
    colA, colB = st.columns(2)
    with colA:
        y8_path = st.text_input("YOLOv8 (.pt) path", value=os.path.join("weights", "yolov8_fracture.pt"))
        conf_thres = st.slider("Confidence threshold (both)", 0.05, 0.90, 0.25, 0.05)
    with colB:
        y7_path = st.text_input("YOLOv7 (.onnx) path", value=os.path.join("weights", "yolov7_fracture.onnx"))
        iou_thres  = st.slider("YOLOv7 NMS IoU", 0.10, 0.90, 0.45, 0.05)

    min_area_ratio = st.slider("Ignore tiny boxes (area % of image)", 0.0, 1.0, 0.001, 0.001,
                               help="E.g., 0.002 = 0.2% of image area. Helps remove tiny false positives.")

# ======================= Input =======================
uploaded = st.file_uploader("Upload an X-ray image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

pil_input = Image.open(uploaded).convert("RGB")
st.image(pil_input, caption="Input X-ray", width="stretch")

# ======================= Load Weights =======================
errors = []
y8_model = y7_sess = None
if os.path.isfile(y8_path):
    try:
        y8_model = load_yolov8(y8_path)
    except Exception as e:
        errors.append(f"YOLOv8 load error: {e}")
else:
    errors.append(f"YOLOv8 weight not found: {y8_path}")

if os.path.isfile(y7_path):
    try:
        y7_sess = load_yolov7_onnx(y7_path)
    except Exception as e:
        errors.append(f"YOLOv7 ONNX load error: {e}")
else:
    errors.append(f"YOLOv7 weight not found: {y7_path}")

if errors:
    st.error(" • ".join(errors))
    st.stop()

# ======================= Inference =======================
im_bgr = pil_to_cv2(pil_input)
H0, W0 = im_bgr.shape[:2]

with st.spinner("Predicting (YOLOv8)…"):
    v8_boxes_raw = run_yolov8(y8_model, pil_input, conf=conf_thres)
v8_boxes = filter_by_area(v8_boxes_raw, (H0, W0), min_area_ratio)
v8_pred  = decide_label(v8_boxes)

with st.spinner("Predicting (YOLOv7)…"):
    try:
        v7_boxes_raw = run_yolov7_safe(y7_sess, pil_input, conf=conf_thres, iou=iou_thres)
        v7_err = None
    except Exception as e:
        v7_boxes_raw, v7_err = [], str(e)
v7_boxes = filter_by_area(v7_boxes_raw, (H0, W0), min_area_ratio)
v7_pred  = decide_label(v7_boxes)

# ======================= Side-by-side Compare =======================
left, right = st.columns(2, vertical_alignment="top")

with left:
    st.markdown("### YOLOv8")
    st.markdown(
        f'<span class="badge {"ok" if v8_pred=="Fractured" else "warn"}">{v8_pred}</span>', 
        unsafe_allow_html=True
    )
    img_v8 = draw_boxes(pil_to_cv2(pil_input.copy()), v8_boxes)
    img_v8_pil = cv2_to_pil(img_v8)
    st.image(img_v8_pil, width="stretch")
    st.markdown(
        f'<div class="panel"><b>Detections:</b> {len(v8_boxes)}'
        + (f' &nbsp;·&nbsp; <span class="subtle">max conf: {max_conf(v8_boxes):.3f}</span>' if v8_boxes else '')
        + '</div>', unsafe_allow_html=True
    )
    if v8_boxes:
        st.dataframe(
            [{"x1": int(a), "y1": int(b), "x2": int(c), "y2": int(d), "conf": round(e, 3)}
             for (a, b, c, d, e) in v8_boxes],
            height=250
        )
    st.download_button("Download YOLOv8 Image", data=image_bytes(img_v8_pil),
                       file_name="output_yolov8.png", mime="image/png")

with right:
    st.markdown("### YOLOv7")
    if v7_err:
        st.markdown('<span class="badge warn">ONNX Error</span>', unsafe_allow_html=True)
        with st.expander("Show debug details"):
            st.code(v7_err)
    else:
        st.markdown(
            f'<span class="badge {"ok" if v7_pred=="Fractured" else "warn"}">{v7_pred}</span>', 
            unsafe_allow_html=True
        )
    img_v7 = draw_boxes(pil_to_cv2(pil_input.copy()), v7_boxes)
    img_v7_pil = cv2_to_pil(img_v7)
    st.image(img_v7_pil, width="stretch")
    st.markdown(
        f'<div class="panel"><b>Detections:</b> {len(v7_boxes)}'
        + (f' &nbsp;·&nbsp; <span class="subtle">max conf: {max_conf(v7_boxes):.3f}</span>' if v7_boxes else '')
        + '</div>', unsafe_allow_html=True
    )
    if v7_boxes:
        st.dataframe(
            [{"x1": int(a), "y1": int(b), "x2": int(c), "y2": int(d), "conf": round(e, 3)}
             for (a, b, c, d, e) in v7_boxes],
            height=250
        )
    st.download_button("Download YOLOv7 Image", data=image_bytes(img_v7_pil),
                       file_name="output_yolov7.png", mime="image/png")

# ======================= Footer =======================
st.markdown("---")
st.caption(
    "Decision rule: **Fractured** if the model outputs ≥1 detection at or above the confidence threshold "
    "(after optional tiny-box filtering). Otherwise **Not Fractured**."
)
