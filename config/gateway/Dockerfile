FROM nginx:alpine

# Copy the gateway's active nginx config into the image so the gateway
# always starts with the expected server blocks.
COPY conf.d/default.conf /etc/nginx/conf.d/default.conf
 
