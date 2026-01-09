FROM python:3.9-slim

WORKDIR /app

# 시스템 의존성 (OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run은 8080 포트를 사용합니다
EXPOSE 8080

# Streamlit 실행 명령어 (포트 지정 필수)
CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
