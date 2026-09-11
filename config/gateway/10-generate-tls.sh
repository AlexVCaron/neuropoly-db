#!/bin/sh
set -eu

TLS_DIR="/etc/nginx/tls"
TLS_CERT="${NB_GATEWAY_TLS_CERT_PATH:-$TLS_DIR/tls.crt}"
TLS_KEY="${NB_GATEWAY_TLS_KEY_PATH:-$TLS_DIR/tls.key}"
TLS_CN="${NB_GATEWAY_TLS_CN:-localhost}"
TLS_DAYS="${NB_GATEWAY_TLS_DAYS:-3650}"

mkdir -p "$TLS_DIR"

if [ -f "$TLS_CERT" ] && [ -f "$TLS_KEY" ]; then
  echo "Gateway TLS: using existing certificate at $TLS_CERT"
  exit 0
fi

echo "Gateway TLS: generating self-signed certificate for CN=$TLS_CN"
openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout "$TLS_KEY" \
  -out "$TLS_CERT" \
  -days "$TLS_DAYS" \
  -subj "/CN=$TLS_CN" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,DNS:$TLS_CN"

chmod 600 "$TLS_KEY"
chmod 644 "$TLS_CERT"
