FROM mcr.microsoft.com/playwright/python:v1.63.0-noble
WORKDIR /opt/sub2api/prism-adapter
COPY requirements.txt ./
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt
COPY . ./
# Keep Chromium's sandbox enabled. The service runs as pwuser, never as root.
RUN python3 container_runtime.py --prepare-sandbox && \
    mkdir -p /var/lib/sub2api-prism && \
    chown pwuser:pwuser /var/lib/sub2api-prism && \
    chmod 700 /var/lib/sub2api-prism
USER pwuser
ENV PRISM_ADAPTER_STATE_DIR=/var/lib/sub2api-prism
ENTRYPOINT ["python3", "/opt/sub2api/prism-adapter/container_runtime.py"]
