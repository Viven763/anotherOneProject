# Dockerfile for Ethereum BIP39 Recovery on Vast.ai
# Supports NVIDIA GPU with CUDA/OpenCL

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Установка базовых зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

# Установка Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Создание рабочей директории
WORKDIR /app

# Копирование файлов проекта
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY cl ./cl
COPY config.json ./

# Примечание: eth20240925 будет монтироваться как volume
# Не копируем его в образ (слишком большой - 4 GB)

# Сборка проекта в release режиме
RUN cargo build --release

# Переменные окружения
ENV RUST_LOG=info
ENV WORK_SERVER_URL=http://localhost:3000
ENV WORK_SERVER_SECRET=secret
ENV DATABASE_PATH=/app/data/eth20240925

# Запуск worker
CMD ["./target/release/eth_recovery"]
