import streamlit as st
import os
import cv2
import numpy as np
import requests
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 0. 설정 및 모델 로드
# ==========================================
st.set_page_config(page_title="AI Glasses Try-On", layout="wide")

@st.cache_resource
def load_detector():
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        with st.spinner("AI 모델 다운로드 중..."):
            try:
                r = requests.get(url, timeout=30)
                with open(model_path, 'wb') as f:
                    f.write(r.content)
            except Exception as e:
                st.error(f"모델 다운로드 실패: {e}")
                return None
    
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        return vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        st.error(f"모델 초기화 실패: {e}")
        return None

detector = load_detector()

# ==========================================
# 1. 랜드마크 & 얼굴 분석 함수
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

def dist(a, b): return float(np.linalg.norm(a - b))

def angle(a, b, c):
    ba, bc = a - b, c - b
    cosv = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

def get_face_metrics(landmarks, w, h):
    def pt(i): return np.array([landmarks[i].x*w, landmarks[i].y*h], dtype=np.float32)

    face_h = dist(pt(LM["forehead_top"]), pt(LM["chin"]))
    cheek_w = dist(pt(LM["left_cheek"]), pt(LM["right_cheek"]))
    jaw_w = dist(pt(LM["left_jaw"]), pt(LM["right_jaw"]))
    upper_w = max(dist(pt(LM["left_temple"]), pt(LM["right_temple"])),
                  dist(pt(LM["left_forehead"]), pt(LM["right_forehead"])))

    ratio = face_h / (cheek_w + 1e-6)
    balance = 1 - (abs(upper_w - cheek_w) + abs(jaw_w - cheek_w)) / (2*cheek_w + 1e-6)
    jaw_ang = angle(pt(LM["left_cheek"]), pt(LM["left_jaw"]), pt(LM["chin"]))
    
    return ratio, balance, upper_w, jaw_w, jaw_ang

def classify_face_shape(ratio, balance, upper, jaw, jaw_ang):
    if ratio < 1.15 and balance > 0.9: return "round"
    if ratio > 1.28: return "oval"
    if jaw > upper: return "triangle"
    if upper > jaw: return "heart"
    if balance > 0.92 and jaw_ang > 150: return "square"
    return "oval"

def acuity_to_diopter(acuity):
    if acuity >= 1.0: return 0.0
    elif acuity >= 0.8: return -0.50
    elif acuity >= 0.6: return -1.00
    elif acuity >= 0.4: return -1.75
    elif acuity >= 0.3: return -2.50
    elif acuity >= 0.2: return -3.50
    elif acuity >= 0.1: return -5.00
    else: return -6.00

def check_frequency(avg_power):
    val = abs(avg_power)
    if val < 1.0: return "착용 빈도 낮음 (필요할 때만 착용)"
    elif val < 3.0: return "착용 빈도 중간 (운전·수업·업무 시 착용 권장)"
    elif val < 5.0: return "착용 빈도 높음 (하루 대부분 착용 필요)"
    else: return "착용 빈도 매우 높음 (상시 착용 권장)"

# ==========================================
# 2. 이미지 처리 (안경 합성 로직 개선)
# ==========================================
def pil_to_bgra(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGBA")), cv2.COLOR_RGBA2BGRA)

def cleanup_glasses(bgra):
    # 흰색 배경(JPG)을 투명하게 변환
    b,g,r,a = cv2.split(bgra)
    # 밝은 영역을 투명하게
    mask = (b > 240) & (g > 240) & (r > 240)
    a[mask] = 0
    return cv2.merge([b,g,r,a])

def find_anchors_robust(bgra):
    """
    안경의 좌/우 렌즈 중심을 찾습니다. 
    실패 시 이미지 비율 기반으로 강제 설정합니다 (무한 로딩 방지).
    """
    h, w = bgra.shape[:2]
    alpha = bgra[:, :, 3]
    
    # 1. 컨투어로 찾기 시도
    _, thresh = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pL, pR, pB = None, None, None

    if contours:
        # 면적순 정렬
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) >= 2:
            # 덩어리가 2개 이상 (안경알 분리형)
            M1 = cv2.moments(contours[0])
            M2 = cv2.moments(contours[1])
            if M1["m00"] > 0 and M2["m00"] > 0:
                c1 = np.array([M1["m10"]/M1["m00"], M1["m01"]/M1["m00"]])
                c2 = np.array([M2["m10"]/M2["m00"], M2["m01"]/M2["m00"]])
                if c1[0] < c2[0]: pL, pR = c1, c2
                else: pL, pR = c2, c1
    
    # 2. 실패했거나(None), 덩어리가 1개인 경우 -> 강제 설정 (Fallback)
    if pL is None:
        pL = np.array([w * 0.25, h * 0.5]) # 왼쪽 1/4 지점
        pR = np.array([w * 0.75, h * 0.5]) # 오른쪽 3/4 지점
    
    pB = (pL + pR) / 2 # 브릿지는 중간
    return pL, pR, pB

def overlay_glasses(face_img, landmarks, glasses_bgra):
    h, w = face_img.shape[:2]
    
    def pt(idx): 
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float32)

    # 얼굴 기준점
    f_L = pt(EYE["lo"])  # 왼쪽 눈 바깥
    f_R = pt(EYE["ro"])  # 오른쪽 눈 바깥
    f_N = pt(NOSE)       # 코

    # 안경 기준점
    g_L, g_R, g_B = find_anchors_robust(glasses_bgra)
    
    # 변환 좌표 계산 (눈 위치보다 살짝 바깥쪽으로 조정)
    eye_width = float(np.linalg.norm(f_L - f_R))
    
    # 목표 좌표 (얼굴 위)
    # 눈 너비의 10%만큼 바깥쪽으로 확장해서 안경이 눈을 덮도록 함
    t_L = f_L + np.array([-eye_width * 0.15, 0])
    t_R = f_R + np.array([eye_width * 0.15, 0])
    t_B = f_N + np.array([0, -eye_width * 0.2]) # 코보다 약간 위

    src_pts = np.float32([g_L, g_R, g_B])
    dst_pts = np.float32([t_L, t_R, t_B])

    # Affine 변환 행렬
    M = cv2.getAffineTransform(src_pts, dst_pts)
    
    # 안경 이미지 변형
    warped = cv2.warpAffine(
        glasses_bgra, M, (w, h), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0,0,0,0)
    )
    
    # 합성 (Alpha Blending)
    # 얼굴 이미지를 BGRA로 변환
    face_bgra = cv2.cvtColor(face_img, cv2.COLOR_BGR2BGRA)
    
    # 알파 채널 정규화 (0~1)
    alpha = warped[:, :, 3] / 255.0
    alpha = np.dstack([alpha, alpha, alpha]) # 3채널로 맞춤 (BGR 대상)

    # 전경(안경)과 배경(얼굴) 합성
    fg = warped[:, :, :3]
    bg = face_bgra[:, :, :3]
    
    out = (fg * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    return out

# ==========================================
# 3. 메인 UI (레이아웃 수정)
# ==========================================
st.title("👓 AI Smart Glasses Fitting")
st.markdown("서버에 저장된 **얼굴 사진**을 분석하고 **안경**을 가상으로 착용해보세요.")

# 파일 목록 로드
try:
    all_files = os.listdir('.')
    img_exts = ('.png', '.jpg', '.jpeg', '.webp')
    
    # 키워드로 안경/얼굴 파일 분류
    glasses_keywords = ['glass', 'eye', 'aviator', 'round', 'square']
    
    glasses_files = sorted([f for f in all_files if any(k in f.lower() for k in glasses_keywords) and f.endswith(img_exts)])
    face_files = sorted([f for f in all_files if f not in glasses_files and f.endswith(img_exts)])
except:
    glasses_files = []
    face_files = []

col1, col2 = st.columns(2)

# [왼쪽] 얼굴 선택 및 분석 결과
with col1:
    st.header("1. 얼굴 분석 (Face Analysis)")
    
    # 시력 입력
    c1, c2 = st.columns(2)
    with c1: l_eye = st.number_input("좌안 시력", 0.1, 2.0, 0.5, 0.1)
    with c2: r_eye = st.number_input("우안 시력", 0.1, 2.0, 0.5, 0.1)

    if face_files:
        selected_face = st.selectbox("얼굴 사진 선택", face_files)
        
        if selected_face:
            # 이미지 로드
            face_pil = Image.open(selected_face).convert("RGB")
            # 미리보기용 리사이즈 (속도 향상)
            face_pil.thumbnail((600, 600)) 
            face_cv2 = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)
            h, w = face_cv2.shape[:2]

            # 랜드마크 검출
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(face_cv2, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_img)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                
                # 분석 수행
                r, b, u, j, ang = get_face_metrics(lm, w, h)
                shape = classify_face_shape(r, b, u, j, ang)
                recs = VERY_SUITABLE_FRAMES.get(shape, ["square"])
                
                avg_d = (acuity_to_diopter(l_eye) + acuity_to_diopter(r_eye)) / 2
                freq = check_frequency(avg_d)

                # 분석 결과 표시
                st.success(f"**얼굴형:** {shape.upper()}")
                st.info(f"**추천 안경:** {', '.join(recs).upper()}")
                st.warning(f"**{freq}**")
                
                # 얼굴 이미지 표시
                st.image(face_pil, caption="분석된 얼굴", use_container_width=True)
            else:
                st.error("얼굴을 찾을 수 없습니다.")
    else:
        st.warning("얼굴 이미지가 없습니다.")

# [오른쪽] 안경 선택 및 가상 피팅
with col2:
    st.header("2. 가상 피팅 (Virtual Try-On)")
    
    if glasses_files:
        selected_glass = st.selectbox("안경 선택", glasses_files)
        
        # 버튼을 누르거나 선택하면 자동 실행
        if selected_glass and 'face_cv2' in locals() and 'lm' in locals():
            st.write("▼ 아래에서 착용 결과를 확인하세요.")
            
            try:
                with st.spinner("안경 착용 중..."):
                    # 안경 이미지 로드
                    g_pil = Image.open(selected_glass).convert("RGBA")
                    g_bgra = pil_to_bgra(g_pil)
                    g_bgra = cleanup_glasses(g_bgra) # 전처리
                    
                    # 합성 수행 (robust 함수 사용으로 멈춤 방지)
                    final_bgr = overlay_glasses(face_cv2, lm, g_bgra)
                    
                    # 결과 표시
                    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
                    st.image(final_rgb, caption=f"착용 결과: {selected_glass}", use_container_width=True)
            
            except Exception as e:
                st.error(f"오류 발생: {e}")
    else:
        st.warning("안경 이미지가 없습니다.")
