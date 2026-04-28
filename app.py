import io
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


@dataclass
class Detection:
    cls_id: int
    label: str
    conf: float
    xyxy: Tuple[int, int, int, int]


@st.cache_resource(show_spinner=False)
def load_model(model_name: str = "yolov8n.pt") -> YOLO:
    model = YOLO(model_name)
    return model


def preprocess_image(pil_img: Image.Image, max_side: int = 960) -> np.ndarray:
    img = pil_img.convert("RGB")
    np_img = np.array(img)  # RGB

    h, w = np_img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        np_img = cv2.resize(np_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return np_img


def run_detection(
    model: YOLO,
    rgb_img: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
    max_det: int = 200,
) -> List[Detection]:
    results = model.predict(
        source=rgb_img,
        device="cpu",
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        verbose=False,
    )

    r = results[0]
    names: Dict[int, str] = getattr(r, "names", {}) or {}
    dets: List[Detection] = []

    if r.boxes is None or len(r.boxes) == 0:
        return dets

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        label = names.get(int(k), str(int(k)))
        dets.append(
            Detection(
                cls_id=int(k),
                label=label,
                conf=float(c),
                xyxy=(int(x1), int(y1), int(x2), int(y2)),
            )
        )

    return dets


def draw_boxes(rgb_img: np.ndarray, dets: List[Detection]) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        color = (0, 255, 0)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{d.label} {d.conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - th - baseline - 4)
        cv2.rectangle(bgr, (x1, y_text), (x1 + tw + 6, y_text + th + baseline + 6), color, -1)
        cv2.putText(
            bgr,
            text,
            (x1 + 3, y_text + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def display_results(annotated_rgb: np.ndarray, dets: List[Detection]) -> None:
    st.subheader("결과")
    st.image(annotated_rgb, caption="탐지 결과(바운딩 박스)", use_container_width=True)

    st.subheader("객체 통계")
    if not dets:
        st.info("탐지된 객체가 없습니다.")
        return

    counts = Counter([d.label for d in dets])
    st.write({k: int(v) for k, v in counts.items()})
    st.write(f"총 객체 수: **{len(dets)}**")

    st.subheader("탐지 리스트")
    rows: List[Dict[str, Any]] = []
    for i, d in enumerate(sorted(dets, key=lambda x: x.conf, reverse=True), start=1):
        x1, y1, x2, y2 = d.xyxy
        rows.append(
            {
                "#": i,
                "label": d.label,
                "confidence": round(d.conf, 3),
                "bbox_xyxy": f"({x1},{y1})-({x2},{y2})",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes))


def main() -> None:
    st.set_page_config(page_title="Mobile Object Detection (YOLOv8)", layout="wide")
    st.title("Mobile Object Detection App (YOLOv8)")
    st.write(
        "스마트폰 카메라로 촬영하거나 이미지를 업로드하면, **YOLOv8n(CPU)** 으로 객체를 탐지하고 "
        "바운딩 박스/라벨/신뢰도를 시각화합니다."
    )

    with st.sidebar:
        st.header("설정")
        conf_thres = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
        iou_thres = st.slider("IoU threshold", 0.1, 0.9, 0.5, 0.05)
        max_side = st.selectbox("입력 이미지 최대 변 길이(리사이즈)", [640, 800, 960, 1280], index=2)
        max_det = st.selectbox("최대 탐지 개수", [50, 100, 200, 300], index=2)

    st.subheader("이미지 입력")
    col1, col2 = st.columns(2)
    with col1:
        cam = st.camera_input("카메라로 촬영")
    with col2:
        up = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "webp"])

    image_bytes = None
    if cam is not None:
        image_bytes = cam.getvalue()
    elif up is not None:
        image_bytes = up.getvalue()

    if image_bytes is None:
        st.warning("카메라로 촬영하거나 이미지를 업로드하세요.")
        return

    pil = _bytes_to_pil(image_bytes)
    st.image(pil, caption="입력 이미지", use_container_width=True)

    st.subheader("탐지 실행")
    run = st.button("Detect Objects", type="primary", use_container_width=True)

    if not run:
        return

    with st.spinner("모델 로딩 및 추론 중..."):
        model = load_model("yolov8n.pt")
        rgb = preprocess_image(pil, max_side=max_side)
        dets = run_detection(model, rgb, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)
        annotated = draw_boxes(rgb, dets)

    display_results(annotated, dets)


if __name__ == "__main__":
    main()

