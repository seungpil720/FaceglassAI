import streamlit as st
import os
import math
import cv2
import numpy as np
import requests
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 0. 기본 설정 및 모델 로드
# ==========================================
st.set_page_config(page_title="AI Glasses Try-On", layout="wide")

@st.cache_resource
def load_detector():
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        with st.spinner("AI 모델 다운로드 중..."):
            r = requests.get(url)
            with open(model_path, 'wb') as f:
                f.write(r.content)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    return vision.FaceLandmarker.create_from_options(options)

detector = load_detector()

# ==========================================
# 1. 랜드마크 및 얼굴형 분석 로직
# ==========================================
LM = {
    "forehead_top": 10, "chin": 152,
    "left_cheek": 234, "right_cheek": 454,
    "left_jaw": 172, "right_jaw": 397,
    "left_temple": 127, "right_temple": 356,
    "left_forehead": 71, "right_forehead": 301,
}
EYE = {"lo": 33, "li": 133, "ri": 362, "ro": 263}
NOSE = 168

VERY_SUITABLE_FRAMES = {
    "oval": ["cat-eye", "square", "aviator"],
    "round": ["square"],
    "square": ["round"],
    "heart": ["oval"],
    "triangle": ["oval", "round"],
}

def 거리(a, b): return float(np.linalg.norm(a - b))

def 각도(a, b, c):
    ba, bc = a - b, c - b
    cosv = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return math.degrees(math.acos(np.clip(cosv, -1, 1)))

def 얼굴_측정치_계산(랜드마크, w, h):
    def 점(i): return np.array([랜드마크[i].x*w, 랜드마크[i].y*h], dtype=np.float32)

    얼굴_높이 = 거리(점(LM["forehead_top"]), 점(LM["chin"]))
    광대_너비 = 거리(점(LM["left_cheek"]), 점(LM["right_cheek"]))
    턱_너비 = 거리(점(LM["left_jaw"]), 점(LM["right_jaw"]))
    상부_너비 = max(
        거리(점(LM["left_temple"]), 점(LM["right_temple"])),
        거리(점(LM["left_forehead"]), 점(LM["right_forehead"]))
    )

    비율 = 얼굴_높이 / (광대_너비 + 1e-6)
    균형도 = 1 - (abs(상부_너비-광대_너비)+abs(턱_너비-광대_너비))/(2*광대_너비+1e-6)
    턱_각도 = 각도(점(LM["left_cheek"]), 점(LM["left_jaw"]), 점(LM["chin"]))
    return 비율, 균형도, 상부_너비, 턱_너비, 턱_각도

def 얼굴형_분류(비율, 균형도, 상부, 턱, 턱각):
    if 비율 < 1.15 and 균형도 > 0.9: return "round"
    if 비율 > 1.28: return "oval"
    if 턱 > 상부: return "triangle"
    if 상부 > 턱: return "heart"
    if 균형도 > 0.92 and 턱각 > 150: return "square"
    return "oval"

def 시력을_도수로_변환(시력):
    if 시력 >= 1.0: return 0.0
    elif 시력 >= 0.8: return -0.50
    elif 시력 >= 0.6: return -1.00
    elif 시력 >= 0.4: return -1.75
    elif 시력 >= 0.3: return -2.50
    elif 시력 >= 0.2: return -3.50
    elif 시력 >= 0.1: return -5.00
    else: return -6.00

def 착용_빈도_판단(평균_도수):
    도수_절댓값 = abs(평균_도수)
    if 도수_절댓값 < 1.0: return "착용 빈도 낮음 (필요할 때만 착용)"
    elif 도수_절댓값 < 3.0: return "착용 빈도 중간 (운전·수업·업무 시 착용 권장)"
    elif 도수_절댓값 < 5.0: return "착용 빈도 높음 (하루 대부분 착용 필요)"
    else: return "착용 빈도 매우 높음 (상시 착용 권장)"

# ==========================================
# 2. 이미지 처리 (투명 배경 및 오버레이)
# ==========================================
def pil_to_bgra(pil_rgba: Image.Image) -> np.ndarray:
    arr = np.array(pil_rgba.convert("RGBA"), dtype=np.uint8)
    return arr[:, :, [2,1,0,3]]

def remove_white_bg_to_alpha(bgra: np.ndarray, thr=240) -> np.ndarray:
    b,g,r,a = cv2.split(bgra)
    mask_bg = (r > thr) & (g > thr) & (b > thr)
    a2 = a.copy()
    a2[mask_bg] = 0
    return cv2.merge([b,g,r,a2])

def clean_alpha(bgra: np.ndarray, min_area=150, feather=2, close_ks=5, open_ks=3) -> np.ndarray:
    b,g,r,a = cv2.split(bgra)
    _, bin_a = cv2.threshold(a, 10, 255, cv2.THRESH_BINARY)

    if close_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        bin_a = cv2.morphologyEx(bin_a, cv2.MORPH_CLOSE, k, iterations=1)

    if open_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
        bin_a = cv2.morphologyEx(bin_a, cv2.MORPH_OPEN, k, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_a, connectivity=8)
    keep = np.zeros_like(bin_a)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            keep[labels == i] = 255

    if feather > 0:
        keep = cv2.GaussianBlur(keep, (0,0), sigmaX=feather, sigmaY=feather)

    a_new = keep.astype(np.uint8)
    return cv2.merge([b,g,r,a_new])

def remove_white_artifacts_even_if_opaque(bgra: np.ndarray, white_thr=235, strip_ar_thr=7.0, strip_h_frac=0.20):
    b,g,r,a = cv2.split(bgra)
    H, W = a.shape
    has_alpha = a > 10
    white = (r > white_thr) & (g > white_thr) & (b > white_thr) & has_alpha
    white_u8 = (white.astype(np.uint8) * 255)

    if white_u8.sum() == 0: return bgra

    n, labels, stats, _ = cv2.connectedComponentsWithStats(white_u8, connectivity=8)
    a2 = a.copy()

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 80:
            a2[labels == i] = 0
            continue
        ar = (w / (h + 1e-6))
        if (ar > strip_ar_thr) and (h < strip_h_frac * H) and (y > 0.35 * H):
            a2[labels == i] = 0

    whiteness = ((r.astype(np.int16) + g.astype(np.int16) + b.astype(np.int16)) / 3.0)
    bright = (whiteness > 225) & (a2 > 10)
    a2[bright] = (a2[bright] * 0.15).astype(np.uint8)
    return cv2.merge([b,g,r,a2])

def load_glasses_from_path(file_path) -> np.ndarray:
    pil = Image.open(file_path).convert("RGBA")
    bgra = pil_to_bgra(pil)

    # 흰색 배경 제거 로직 (JPG 안경 이미지 대응)
    if float(bgra[:, :, 3].mean()) > 250:
        bgra = remove_white_bg_to_alpha(bgra, thr=240)

    bgra = clean_alpha(bgra, min_area=150, feather=2, close_ks=5, open_ks=3)
    bgra = remove_white_artifacts_even_if_opaque(bgra, white_thr=235, strip_ar_thr=7.0, strip_h_frac=0.20)
    bgra = clean_alpha(bgra, min_area=150, feather=2, close_ks=3, open_ks=3)
    return bgra

def find_glasses_anchors(bgra: np.ndarray):
    a = bgra[:, :, 3]
    _, m = cv2.threshold(a, 10, 255, cv2.THRESH_BINARY)
    
    if m.sum() == 0: return None, None, None

    ys, xs = np.where(m > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bw = float(x_max - x_min + 1)
    bh = float(y_max - y_min + 1)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 500:
            comps.append((area, i))
    comps.sort(reverse=True, key=lambda x: x[0])

    if len(comps) >= 2:
        i1, i2 = comps[0][1], comps[1][1]
        c1 = np.array(centroids[i1], dtype=np.float32)
        c2 = np.array(centroids[i2], dtype=np.float32)
        if c1[0] <= c2[0]: pL, pR = c1, c2
        else: pL, pR = c2, c1
        mid = (pL + pR) / 2.0
        top_y = float(min(stats[i1, cv2.CC_STAT_TOP], stats[i2, cv2.CC_STAT_TOP]))
        pB = np.array([mid[0], 0.55 * top_y + 0.45 * mid[1]], dtype=np.float32)
        return pL, pR, pB

    pL = np.array([x_min + 0.33 * bw, y_min + 0.55 * bh], dtype=np.float32)
    pR = np.array([x_min + 0.67 * bw, y_min + 0.55 * bh], dtype=np.float32)
    pB = np.array([x_min + 0.50 * bw, y_min + 0.40 * bh], dtype=np.float32)
    return pL, pR, pB

def _np_point(lm, idx, w, h):
    return np.array([lm[idx].x*w, lm[idx].y*h], dtype=np.float32)

def overlay_glasses_affine(img_bgr, lm, glasses_bgra, big_scale=1.45, temple_width_factor=1.18, y_offset_factor=0.12):
    H, W = img_bgr.shape[:2]
    L_eye = (_np_point(lm, EYE["lo"], W, H) + _np_point(lm, EYE["li"], W, H)) / 2.0
    R_eye = (_np_point(lm, EYE["ro"], W, H) + _np_point(lm, EYE["ri"], W, H)) / 2.0
    N = _np_point(lm, NOSE, W, H)
    Lt = _np_point(lm, LM["left_temple"], W, H)
    Rt = _np_point(lm, LM["right_temple"], W, H)
    temple_w = 거리(Lt, Rt)

    pL, pR, pB = find_glasses_anchors(glasses_bgra)
    if pL is None: return img_bgr

    eye_dist = 거리(L_eye, R_eye)
    templ_dist = 거리(pL, pR)
    target = min(temple_w * temple_width_factor, eye_dist * 2.3)
    scale = (target / (templ_dist + 1e-6)) * big_scale

    gh, gw = glasses_bgra.shape[:2]
    new_w = max(2, int(gw * scale))
    new_h = max(2, int(gh * scale))
    g2 = cv2.resize(glasses_bgra, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    pL2, pR2, pB2 = pL * scale, pR * scale, pB * scale
    down = eye_dist * y_offset_factor
    L_t = np.array([L_eye[0], L_eye[1] + down], dtype=np.float32)
    R_t = np.array([R_eye[0], R_eye[1] + down], dtype=np.float32)
    N_t = np.array([N[0], N[1] + down * 0.35], dtype=np.float32)

    src = np.float32([pL2, pR2, pB2])
    dst = np.float32([L_t, R_t, N_t])
    M = cv2.getAffineTransform(src, dst)

    warped = cv2.warpAffine(g2, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    fg = warped.astype(np.float32) / 255.0
    bg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA).astype(np.float32) / 255.0
    alpha = fg[:, :, 3:4]
    out = fg[:, :, :3] * alpha + bg[:, :, :3] * (1 - alpha)
    return (out * 255.0).clip(0,255).astype(np.uint8)

# ==========================================
# 3. STREAMLIT 웹 앱 UI
# ==========================================
st.title("👓 AI Smart Glasses Fitting (Real Overlay)")
st.markdown("서버에 저장된 **얼굴 사진**과 **안경**을 선택하면 AI가 자동으로 합성해줍니다.")

# 1. 파일 목록 불러오기
try:
    all_files = os.listdir('.')
    # 안경 파일 키워드 (파일명에 이 단어가 들어가면 안경으로 분류)
    glasses_keywords = ['Glasses', 'Cat Eye', 'Aviator', 'Square', 'Round', 'Oval', 'Sunglass']
    
    glasses_files = [f for f in all_files if any(k.lower() in f.lower() for k in glasses_keywords) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    face_files = [f for f in all_files if f not in glasses_files and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.avif'))]
    
    glasses_files.sort()
    face_files.sort()
except Exception as e:
    st.error(f"파일 목록을 불러오는 중 오류 발생: {e}")
    glasses_files = []
    face_files = []

col1, col2 = st.columns([1, 1])

# --- 왼쪽 컬럼: 얼굴 선택 및 분석 ---
with col1:
    st.header("1. Face Analysis")
    l_eye = st.number_input("Left Eye Vision", 0.1, 2.0, 0.5, step=0.1)
    r_eye = st.number_input("Right Eye Vision", 0.1, 2.0, 0.5, step=0.1)
    
    # 얼굴 사진 선택
    if face_files:
        selected_face_file = st.selectbox("Select Face Photo:", face_files)
    else:
        st.warning("얼굴 사진이 없습니다.")
        selected_face_file = None

if selected_face_file:
    # 얼굴 로드 및 분석
    image = Image.open(selected_face_file).convert('RGB')
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    detection_result = detector.detect(mp_img)

    if detection_result.face_landmarks:
        lm = detection_result.face_landmarks[0]
        
        l_d = 시력을_도수로_변환(l_eye)
        r_d = 시력을_도수로_변환(r_eye)
        avg_d = (l_d + r_d) / 2
        freq = 착용_빈도_판단(avg_d)
        
        ratio, balance, upper_w, jaw_w, jaw_angle = 얼굴_측정치_계산(lm, W, H)
        face_shape = 얼굴형_분류(ratio, balance, upper_w, jaw_w, jaw_angle)
        recs = VERY_SUITABLE_FRAMES.get(face_shape, ["square"])

        with col1:
            st.success(f"**Face Shape:** {face_shape.upper()}")
            st.info(f"**Recommended:** {', '.join(recs).upper()}")
            st.warning(f"**Usage:** {freq}")
            # 원본 얼굴 보여주기
            st.image(image, caption="Original Face", use_column_width=True)

        # --- 오른쪽 컬럼: 안경 선택 및 결과 출력 ---
        with col2:
            st.header("2. Virtual Try-On Result")
            st.markdown(f"**{face_shape.upper()}** 얼굴형에 어울리는 안경을 선택하세요.")
            
            if glasses_files:
                selected_glasses_file = st.selectbox("Select Glasses:", glasses_files)
                
                if selected_glasses_file:
                    with st.spinner("안경 합성 중..."):
                        # 안경 이미지 처리 (배경 제거 등)
                        glasses_bgra = load_glasses_from_path(selected_glasses_file)
                        
                        # 오버레이 (합성) 수행
                        final_img = overlay_glasses_affine(
                            img_bgr.copy(), lm, glasses_bgra,
                            big_scale=1.45,
                            temple_width_factor=1.18,
                            y_offset_factor=0.12
                        )
                        
                        # [핵심] 최종 결과 출력
                        final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                        st.image(final_rgb, caption=f"Try-On: {selected_glasses_file}", use_column_width=True)
            else:
                st.warning("서버에 안경 이미지가 없습니다. (파일명에 'Glasses' 포함 필요)")
    else:
        with col1:
            st.error("얼굴을 찾을 수 없습니다. 정면 사진을 선택해주세요.")
