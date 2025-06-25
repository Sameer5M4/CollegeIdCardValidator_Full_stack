# Build stage
FROM node:18 AS builder
WORKDIR /app

# ✅ OPTIMIZATION:
# 1. Copy only the dependency files first.
COPY package.json package-lock.json ./

# 2. Install dependencies. Docker will now cache this layer.
#    It will only re-run if package.json or package-lock.json changes.
RUN npm install

# 3. Now, copy the rest of your application code.
COPY . .

# 4. Build your app. This will run when you change your code, but it's much faster
#    than re-installing all node modules.
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]