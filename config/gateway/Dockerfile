FROM nginx:alpine

RUN apk add --no-cache openssl

# Copy the gateway's active nginx config into the image so the gateway
# always starts with the expected server blocks.
COPY conf.d/default.conf /etc/nginx/conf.d/default.conf

# Generate a fallback self-signed certificate at container startup for local TLS.
COPY 10-generate-tls.sh /docker-entrypoint.d/10-generate-tls.sh
RUN chmod +x /docker-entrypoint.d/10-generate-tls.sh
 
