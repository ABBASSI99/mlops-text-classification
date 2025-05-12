# --------------------------------------
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies with retry
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install PyTorch and transformers with specific versions
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers==4.36.0 && \
    pip install --no-cache-dir python-multipart && \
    pip install --no-cache-dir pandas scikit-learn

# Install remaining requirements with retry options
RUN pip install --no-cache-dir --timeout 1000 --retries 3 -r requirements.txt

# Copy the data directory first
COPY data/raw/data.csv /app/data/raw/

# Copy the rest of the application
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# FROM python:3.10-slim

# WORKDIR /app

# # Install system dependencies with retry
# RUN apt-get update --allow-releaseinfo-change && \
#     apt-get install -y --no-install-recommends \
#     build-essential \
#     gcc \
#     libffi-dev \
#     libssl-dev \
#     wget \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements file
# COPY requirements.txt .

# # Install PyTorch and transformers with specific versions
# RUN pip install --no-cache-dir --upgrade pip setuptools && \
#     pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
#     pip install --no-cache-dir transformers==4.36.0

# # Install remaining requirements with retry options
# RUN pip install --no-cache-dir --timeout 1000 --retries 3 -r requirements.txt

# # Copy the rest of the application
# COPY . .

# EXPOSE 8000

# CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
