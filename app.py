import os
import io
import base64
import math
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import pillow_avif  # avif 플러그인 활성화
from flask import Flask, request, render_template_string
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)

# ==========================================
# [설정] AI 모델 다운로드 (MediaPipe)
# ==========================================
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)

# MediaPipe Detector 초기화
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# ==========================================
# [로직 1] 시력 및 렌즈 추천
# ==========================================
def acuity_to_diopter(acuity):
    if acuity >= 1.0: return 0.0
    elif acuity >= 0.8: return -0.50
    elif acuity >= 0.6: return -1.00
    elif acuity >= 0.4: return -1.75
    elif acuity >= 0.3: return -2.50
    elif acuity >= 0.2: return -3.50
    elif acuity >= 0.1: return -5.00
    else: return -6.00

def lens_index(power):
    power = abs(power)
    if power < 2.0: return "1.56 (일반)"
    elif power < 4.0: return "1.60 (중굴절)"
    elif power < 6.0: return "1.67 (고굴절)"
    else: return "1.74 (초고굴절)"

def get_lens_features(lifestyle, high_power):
    features = ["UV 차단"]
    if lifestyle == "computer": features.append("블루라이트 차단")
    elif lifestyle == "driving": features.append("드라이브 코팅(난반사 방지)")
    elif lifestyle == "outdoor": features.append("변색 렌즈")
    if high_power: features.append("초슬림 가공")
    return features

# ==========================================
# [로직 2] 얼굴형 분석 및 가상 피팅
# ==========================================
SIZE_MULT = 0.95
VERT_SHIFT = 0.28
EYE_LIFT_FRAC = 0.07
FRAME_ALPHA = 235
LENS_TINT_ALPHA = 70
TEMPLE_LEN = 0.32

VERY_SUITABLE_FRAMES = {
    "oval": ["cat-eye", "square", "aviator", "oval"],
    "round": ["cat-eye", "square"],
    "square": ["round"],
    "triangle": ["cat-eye", "round", "oval"],
    "heart": ["round", "oval"],
    "diamond": ["round", "oval"],
}

LM = { "chin": 152, "left_cheek": 234, "right_cheek": 454, "forehead_top": 10, "left_jaw": 172, "right_jaw": 397, "left_temple": 127, "right_temple": 356, "left_forehead": 71, "right_forehead": 301 }
EYE = { "left_outer": 33, "left_inner": 133, "right_inner": 362, "right_outer": 263 }
NOSE_BRIDGE = 168

def dist(a, b): return float(np.linalg.norm(a - b))

def angle_deg(a, b, c):
    ba = a - b; bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    return math.degrees(math.acos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))

def compute_face_metrics(landmarks, w, h):
    def pt(i): return np.array([landmarks[i].x * w, landmarks[i].y * h], dtype=np.float32)
    face_height = dist(pt(LM["forehead_top"]), pt(LM["chin"]))
    cheek_w = dist(pt(LM["left_cheek"]), pt(LM["right_cheek"]))
    jaw_w = dist(pt(LM["left_jaw"]), pt(LM["right_jaw"]))
    upper_w = max(dist(pt(LM["left_temple"]), pt(LM["right_temple"])), dist(pt(LM["left_forehead"]), pt(LM["right_forehead"])))
    jaw_angle = np.mean([angle_deg(pt(LM["left_cheek"]), pt(LM["left_jaw"]), pt(LM["chin"])), 
                         angle_deg(pt(LM["right_cheek"]), pt(LM["right_jaw"]), pt(LM["chin"]))])
    hw = face_height / (cheek_w + 1e-6)
    balance = 1.0 - (abs(upper_w - cheek_w) + abs(jaw_w - cheek_w)) / (2 * cheek_w + 1e-6)
    return {"hw": hw, "balance": balance, "jaw_angle": jaw_angle, "cheek_w": cheek_w, "upper_w": upper_w, "jaw_w": jaw_w}

def classify_face(m):
    if m["hw"] <= 1.12 and m["balance"] >= 0.90: return "round"
    if m["hw"] >= 1.28: return "oval"
    return "oval" if m["hw"] > 1.2 else "round"

def draw_glasses_asset(style, W=1200, H=520):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    tL = np.array([W*0.35, H*0.52], dtype=np.float32)
    tR = np.array([W*0.65, H*0.52], dtype=np.float32)
    tN = np.array([W*0.50, H*0.58], dtype=np.float32)
    lens_w = W * 0.26; lens_h = H * 0.28; stroke = int(max(10, W * 0.010))
    
    if style == "round": lens_h *= 1.05; radius = int(lens_h * 0.60)
    elif style == "square": lens_h *= 0.95; radius = int(min(lens_w, lens_h) * 0.12)
    elif style == "oval": lens_h *= 0.90; radius = int(lens_h * 0.55)
    elif style == "cat-eye": lens_w *= 1.05; lens_h *= 0.85; radius = int(lens_h * 0.55)
    else: radius = int(lens_h * 0.55)
    
    frame_col = (10, 10, 10, FRAME_ALPHA); lens_tint = (40, 40, 40, LENS_TINT_ALPHA)
    def bbox(c, w, h): return [c[0]-w/2, c[1]-h/2, c[0]+w/2, c[1]+h/2]
    L_bb = bbox(tL, lens_w, lens_h); R_bb = bbox(tR, lens_w, lens_h)
    d.rounded_rectangle(L_bb, radius, fill=lens_tint, outline=frame_col, width=stroke)
    d.rounded_rectangle(R_bb, radius, fill=lens_tint, outline=frame_col, width=stroke)
    d.line([(L_bb[2], tL[1]), (R_bb[0], tR[1])], fill=frame_col, width=stroke)
    return img, tL, tR, tN

def apply_glasses(image_bgr, landmarks, w, h, style):
    def pt(i): return np.array([landmarks[i].x * w, landmarks[i].y * h], dtype=np.float32)
    L_eye = (pt(EYE["left_outer"]) + pt(EYE["left_inner"])) / 2.0
    R_eye = (pt(EYE["right_outer"]) + pt(EYE["right_inner"])) / 2.0
    nose = pt(NOSE_BRIDGE)
    center = (L_eye + R_eye) / 2.0
    target = center * (1.0 - VERT_SHIFT) + nose * VERT_SHIFT
    target[1] -= EYE_LIFT_FRAC * dist(L_eye, R_eye)
    L_s = center + (L_eye - center) * SIZE_MULT
    R_s = center + (R_eye - center) * SIZE_MULT
    glass_img, tL, tR, tN = draw_glasses_asset(style)
    src = np.float32([tL, tR, tN]); dst = np.float32([L_s, R_s, target])
    M = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(np.array(glass_img), M, (w, h), flags=cv2.INTER_LINEAR)
    pil_face = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    pil_overlay = Image.fromarray(warped).convert("RGBA")
    return cv2.cvtColor(np.array(Image.alpha_composite(pil_face, pil_overlay).convert("RGB")), cv2.COLOR_RGB2BGR)

# ==========================================
# [Flask Routes]
# ==========================================
@app.route('/', methods=['GET', 'POST'])
def home():
    # 1. 파일 확장자 지원 범위 확대 (.avif, .webp 포함)
    all_files = os.listdir('.')
    image_list = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif'))]
    image_list.sort()
    
    tab = request.form.get('tab', 'lens')
    lens_result = None
    face_result = None
    selected_image = None
    
    if request.method == 'POST':
        if 'btn_lens' in request.form:
            tab = 'lens'
            try:
                l_acuity = float(request.form.get('left_acuity', 0.5))
                r_acuity = float(request.form.get('right_acuity', 0.5))
                lifestyle = request.form.get('lifestyle')
                l_power = acuity_to_diopter(l_acuity)
                r_power = acuity_to_diopter(r_acuity)
                avg_power = (l_power + r_power) / 2
                idx = lens_index(avg_power)
                feats = get_lens_features(lifestyle, abs(avg_power) >= 4.0)
                lens_result = { "left_d": l_power, "right_d": r_power, "index": idx, "features": feats }
            except: pass

        elif 'btn_face' in request.form:
            tab = 'face'
            selected_image = request.form.get('filename')
            if selected_image and selected_image in image_list:
                try:
                    # 2. 이미지 로드 방식 개선 (Pillow -> OpenCV 변환)
                    # avif, webp 등 다양한 포맷 지원을 위해 Pillow로 먼저 엽니다.
                    pil_image = Image.open(selected_image).convert('RGB')
                    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    
                    h, w = img.shape[:2]
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    
                    res = detector.detect(mp_img)
                    if res.face_landmarks:
                        lms = res.face_landmarks[0]
                        metrics = compute_face_metrics(lms, w, h)
                        shape = classify_face(metrics)
                        recs = VERY_SUITABLE_FRAMES.get(shape, ["round"])
                        best_style = recs[0]
                        final_img = apply_glasses(img, lms, w, h, best_style)
                        
                        _, buf = cv2.imencode('.jpg', final_img)
                        b64_str = base64.b64encode(buf).decode('utf-8')
                        face_result = { "shape": shape, "recs": recs, "style": best_style, "img_data": b64_str }
                    else:
                        face_result = {"error": "얼굴을 찾을 수 없습니다."}
                except Exception as e:
                    face_result = {"error": f"파일 처리 중 오류 발생: {str(e)}"}

    return render_template_string(HTML_TEMPLATE, 
                                  images=image_list, 
                                  tab=tab,
                                  lens_result=lens_result,
                                  face_result=face_result,
                                  selected_image=selected_image)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI 안경 파트너</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #f4f6f9; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); overflow: hidden; }
        .header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        .tabs { display: flex; background: #ddd; }
        .tab-btn { flex: 1; padding: 15px; border: none; background: #ddd; cursor: pointer; font-size: 16px; font-weight: bold; }
        .tab-btn.active { background: white; color: #2c3e50; border-top: 3px solid #3498db; }
        .content { padding: 30px; display: none; }
        .content.active { display: block; }
        label { display: block; margin: 10px 0 5px; font-weight: bold; }
        select, input { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; margin-bottom: 15px; }
        button.action { width: 100%; padding: 12px; background: #3498db; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; }
        button.action:hover { background: #2980b9; }
        .result-box { background: #eef2f7; padding: 20px; border-radius: 8px; margin-top: 20px; border-left: 5px solid #27ae60; }
        img.fitted { width: 100%; border-radius: 8px; border: 2px solid #333; margin-top: 10px; }
        .tags span { display: inline-block; background: #2c3e50; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9em; margin-right: 5px; }
    </style>
    <script>
        function openTab(name) {
            document.getElementById('tab-lens').className = 'content';
            document.getElementById('tab-face').className = 'content';
            document.getElementById(name).className = 'content active';
            document.getElementById('btn-t-lens').className = 'tab-btn';
            document.getElementById('btn-t-face').className = 'tab-btn';
            if(name === 'tab-lens') document.getElementById('btn-t-lens').className += ' active';
            else document.getElementById('btn-t-face').className += ' active';
            document.getElementById('hidden_tab').value = (name === 'tab-lens') ? 'lens' : 'face';
        }
    </script>
</head>
<body onload="openTab('tab-{{ tab }}')">
<div class="container">
    <div class="header"><h1>👓 AI 안경 & 렌즈 파트너</h1></div>
    <div class="tabs">
        <button id="btn-t-lens" class="tab-btn" onclick="openTab('tab-lens')">📋 시력/렌즈 추천</button>
        <button id="btn-t-face" class="tab-btn" onclick="openTab('tab-face')">🤳 얼굴형/안경 추천</button>
    </div>
    <div id="tab-lens" class="content">
        <h2>나에게 맞는 렌즈 찾기</h2>
        <form method="POST">
            <input type="hidden" name="tab" value="lens">
            <div style="display:flex; gap:10px;">
                <div style="flex:1;"><label>왼쪽 시력</label><input type="number" step="0.1" name="left_acuity" value="0.5"></div>
                <div style="flex:1;"><label>오른쪽 시력</label><input type="number" step="0.1" name="right_acuity" value="0.5"></div>
            </div>
            <label>주요 생활 패턴</label>
            <select name="lifestyle">
                <option value="computer">컴퓨터/스마트폰 (블루라이트)</option>
                <option value="driving">운전 (빛 번짐 방지)</option>
                <option value="outdoor">야외활동 (자외선 차단)</option>
                <option value="general">일반</option>
            </select>
            <button type="submit" name="btn_lens" class="action">렌즈 분석하기</button>
        </form>
        {% if lens_result %}
        <div class="result-box">
            <h3>📊 분석 결과</h3>
            <p><strong>예상 도수:</strong> 좌 {{ lens_result.left_d }}D / 우 {{ lens_result.right_d }}D</p>
            <p><strong>추천 굴절률:</strong> {{ lens_result.index }}</p>
            <p><strong>추천 기능:</strong></p>
            <div class="tags">{% for f in lens_result.features %}<span>{{ f }}</span>{% endfor %}</div>
        </div>
        {% endif %}
    </div>
    <div id="tab-face" class="content">
        <h2>내 얼굴에 맞는 안경 찾기</h2>
        <form method="POST">
            <input type="hidden" name="tab" id="hidden_tab" value="face">
            <label>분석할 사진 선택 (avif, webp, jpg 지원)</label>
            <select name="filename">
                {% for img in images %}
                <option value="{{ img }}" {% if img == selected_image %}selected{% endif %}>{{ img }}</option>
                {% endfor %}
            </select>
            <button type="submit" name="btn_face" class="action">얼굴 분석 및 가상 피팅</button>
        </form>
        {% if face_result %}
        <div class="result-box">
            {% if face_result.error %}
                <p style="color:red;">⚠️ {{ face_result.error }}</p>
            {% else %}
                <h3>👩‍🦲 분석 완료: {{ face_result.shape.upper() }} 얼굴형</h3>
                <p>추천 스타일: {{ ', '.join(face_result.recs) }}</p>
                <p><strong>적용된 스타일: {{ face_result.style }}</strong></p>
                <img src="data:image/jpeg;base64,{{ face_result.img_data }}" class="fitted">
            {% endif %}
        </div>
        {% endif %}
    </div>
</div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
