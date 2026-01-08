FROM python:3.10-slim

# 파이썬 출력 버퍼링 제거
ENV PYTHONUNBUFFERED True

# 작업 디렉토리 설정
ENV APP_HOME /app
WORKDIR $APP_HOME

# 시스템 패키지 업데이트 및 필수 라이브러리 설치 (GL 관련 에러 방지)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# 파일 복사
COPY . ./

# 파이썬 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 서버 실행
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
