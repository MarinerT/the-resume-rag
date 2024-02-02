FROM python:3.10 as builder
COPY requirements.txt /build/
WORKDIR /build/


# Install GDAL dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gcc \
    g++ \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal


RUN pip3 install -U pip && pip3 install -r requirements.txt

FROM python:3.10 as app
COPY . /app/
WORKDIR /app/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /usr/local/lib/ /usr/local/lib/

RUN python3 scripts/load_data.py

ENTRYPOINT python3 src/app.py
