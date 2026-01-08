FROM python:3.10-slim

# 파이썬 로그 즉시 출력 설정
ENV PYTHONUNBUFFERED True

# 작업 디렉토리 설정
ENV APP_HOME /app
WORKDIR $APP_HOME

# [수정된 부분] libgl1-mesa-glx 대신 libgl1 설치
# 최신 Debian 버전 호환성을 위해 패키지명을 변경했습니다.
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# 파일 복사
COPY . ./

# 파이썬 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 서버 실행
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
