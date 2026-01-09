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
# 1. 유틸리티 함수 (좌표, 각도 등)
# ==========================================
LM = { "chin": 152, "left_temple": 127, "right_temple": 356, "nose": 168 }
EYE = { "left": 33, "right": 263 }

def dist(a, b):
    return float(np.linalg.norm(a - b))

def get_landmark_point(landmarks, idx, w, h):
    return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float32)

# ==========================================
# 2. 안경 이미지 처리 (핵심 로직)
# ==========================================
def pil_to_bgra(pil_image):
    return cv2.cvtColor(np.array(pil_image.convert("RGBA")), cv2.COLOR_RGBA2BGRA)

def cleanup_glasses_image(bgra):
    # 흰색 배경 제거 (JPG 대응)
    b, g, r, a = cv2.split(bgra)
    # 밝기가 매우 밝은 영역(흰색)을 투명하게 처리
    mask = (b > 240) & (g > 240) & (r > 240)
    a[mask] = 0
    return cv2.merge([b, g, r, a])

def find_glasses_anchors(bgra):
    """
    안경 이미지에서 좌/우 렌즈 중심과 브릿지(코) 위치를 찾습니다.
    실패 시 이미지 크기 기반으로 추정치를 반환합니다 (무한 로딩 방지).
    """
    h, w = bgra.shape[:2]
    alpha = bgra[:, :, 3]
    
    # 투명도가 아닌 영역 찾기
    _, thresh = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 컨투어가 감지되면 렌즈 위치 계산 시도
    if contours:
        # 면적이 큰 순서대로 정렬
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # 덩어리가 2개 이상이면 (양쪽 렌즈가 분리된 경우)
        if len(contours) >= 2:
            M1 = cv2.moments(contours[0])
            M2 = cv2.moments(contours[1])
            if M1["m00"] != 0 and M2["m00"] != 0:
                c1 = np.array([M1["m10"] / M1["m00"], M1["m01"] / M1["m00"]])
                c2 = np.array([M2["m10"] / M2["m00"], M2["m01"] / M2["m00"]])
                
                # 좌우 정렬
                if c1[0] < c2[0]: pL, pR = c1, c2
                else: pL, pR = c2, c1
                
                pB = (pL + pR) / 2  # 중간지점(브릿지)
                return pL, pR, pB

    # [Fallback] 컨투어 감지 실패하거나 덩어리가 1개인 경우 (테가 이어진 안경)
    # 이미지의 1/4, 3/4 지점을 렌즈 중심으로 가정
    pL = np.array([w * 0.25, h * 0.5])
    pR = np.array([w * 0.75, h * 0.5])
    pB = np.array([w * 0.50, h * 0.5])
    return pL, pR, pB

def overlay_glasses(face_img, landmarks, glasses_bgra):
    h, w = face_img.shape[:2]
    
    # 1. 얼굴 기준 좌표 계산
    face_L = get_landmark_point(landmarks, EYE["left"], w, h)
    face_R = get_landmark_point(landmarks, EYE["right"], w, h)
    face_N = get_landmark_point(landmarks, LM["nose"], w, h)
    
    # 2. 안경 기준 좌표 계산 (실패 없는 함수 호출)
    glass_L, glass_R, glass_B = find_glasses_anchors(glasses_bgra)
    
    # 3. 크기 및 회전 계산 (Affine Transform)
    # 소스 좌표 (안경)
    src_pts = np.float32([glass_L, glass_R, glass_B])
    # 타겟 좌표 (얼굴) - 눈 위치보다 약간 아래, 코 위치 고려
    face_width = dist(face_L, face_R)
    # 안경이 눈보다 약간 커야 하므로 스케일 조정
    
    # 미세 조정 파라미터
    target_L = face_L + np.array([-face_width * 0.1, 0]) 
    target_R = face_R + np.array([face_width * 0.1, 0])
    target_B = face_N + np.array([0, -face_width * 0.15]) # 코보다 약간 위

    dst_pts = np.float32([target_L, target_R, target_B])
    
    # 변환 행렬 계산
    matrix = cv2.getAffineTransform(src_pts, dst_pts)
    
    # 안경 이미지 변형
    warped_glasses = cv2.warpAffine(
        glasses_bgra, matrix, (w, h), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0,0,0,0)
    )
    
    # 4. 합성 (Alpha Blending)
    face_bgra = cv2.cvtColor(face_img, cv2.COLOR_BGR2BGRA)
    
    # 알파 채널 정규화 (0~1)
    alpha_mask = warped_glasses[:, :, 3] / 255.0
    alpha_mask = np.dstack([alpha_mask] * 3) # 3채널로 확장
    
    # 합성 공식: (안경 * 알파) + (얼굴 * (1-알파))
    foreground = warped_glasses[:, :, :3]
    background = face_bgra[:, :, :3]
    
    combined = (foreground * alpha_mask + background * (1.0 - alpha_mask)).astype(np.uint8)
    return combined

# ==========================================
# 3. 메인 UI
# ==========================================
st.title("👓 AI Smart Glasses Fitting")
st.write("서버에 업로드된 사진을 선택하여 안경을 착용해 보세요.")

# 파일 목록 로드
try:
    all_files = os.listdir('.')
    img_exts = ('.png', '.jpg', '.jpeg', '.webp')
    
    # 파일명에 'glass'가 포함되면 안경, 아니면 얼굴로 간단 분류
    glasses_files = sorted([f for f in all_files if 'glass' in f.lower() and f.endswith(img_exts)])
    # glasses가 아니고, 파이썬/텍스트 파일이 아닌 것들을 얼굴 사진으로 간주
    face_files = sorted([f for f in all_files if f not in glasses_files and f.endswith(img_exts)])
    
except Exception as e:
    st.error(f"파일 목록 로드 중 오류: {e}")
    glasses_files = []
    face_files = []

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 얼굴 사진 선택")
    if face_files:
        selected_face = st.selectbox("얼굴 이미지", face_files)
        if selected_face:
            face_pil = Image.open(selected_face).convert('RGB')
            face_cv2 = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR)
            st.image(face_pil, caption="선택된 얼굴", use_container_width=True)
    else:
        st.warning("얼굴 사진이 없습니다. (.jpg, .png 등)")

with col2:
    st.subheader("2. 안경 선택 및 결과")
    if glasses_files:
        selected_glass = st.selectbox("안경 이미지", glasses_files)
        
        if selected_glass and 'face_cv2' in locals():
            if st.button("안경 착용하기 (Click to Try-On)"):
                with st.spinner("AI가 안경을 씌우는 중입니다..."):
                    try:
                        # 1. 안경 이미지 로드 및 전처리
                        glass_pil = Image.open(selected_glass).convert("RGBA")
                        glass_bgra = pil_to_bgra(glass_pil)
                        glass_bgra = cleanup_glasses_image(glass_bgra)
                        
                        # 2. 얼굴 랜드마크 검출
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(face_cv2, cv2.COLOR_BGR2RGB))
                        detection_result = detector.detect(mp_img)
                        
                        if detection_result.face_landmarks:
                            # 3. 합성 수행
                            landmarks = detection_result.face_landmarks[0]
                            final_img = overlay_glasses(face_cv2, landmarks, glass_bgra)
                            
                            # 4. 결과 출력
                            st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="착용 결과", use_container_width=True)
                        else:
                            st.error("사진에서 얼굴을 찾을 수 없습니다.")
                            
                    except Exception as e:
                        st.error(f"합성 중 오류 발생: {e}")
                        # 디버깅을 위해 에러 상세 출력
                        import traceback
                        st.text(traceback.format_exc())
    else:
        st.warning("안경 이미지가 없습니다. (파일명에 'glass' 포함 필요)")
