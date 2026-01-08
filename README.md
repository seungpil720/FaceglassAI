# 👓 AI Glasses & Lens Recommender

이 프로젝트는 **Google Cloud Run**에서 실행되는 웹 애플리케이션으로, 두 가지 핵심 기능을 제공합니다.

1.  **얼굴형 분석 및 가상 피팅:**
    * 서버에 업로드된 사진 중 하나를 선택하면, AI(MediaPipe)가 얼굴형을 분석합니다.
    * 분석된 얼굴형(Round, Oval 등)에 가장 잘 어울리는 안경을 추천하고, 사진 위에 **자동으로 합성(Virtual Try-On)**해 줍니다.
2.  **시력 기반 렌즈 추천:**
    * 사용자가 시력과 생활 패턴을 입력하면, 적절한 안경 렌즈 도수와 옵션(블루라이트 차단 등)을 추천해 줍니다.

## 📂 파일 구조
* `app.py`: Flask 웹 서버 및 AI 분석 로직 전체
* `requirements.txt`: 필요한 파이썬 라이브러리 목록
* `Dockerfile`: 구글 클라우드 배포용 설정
* `*.jpg`: 분석 테스트를 위한 샘플 이미지 파일들

## 🚀 배포 방법
GitHub에 코드를 Push하면 Google Cloud Build 트리거를 통해 자동으로 배포됩니다.
