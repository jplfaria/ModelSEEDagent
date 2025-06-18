# Deployment Guide

ModelSEEDagent supports various deployment scenarios from local development to production-scale distributed systems.

## Deployment Scenarios

### 1. Local Development

For development and testing:

```bash
# Simple local installation
git clone https://github.com/ModelSEED/ModelSEEDagent.git
cd ModelSEEDagent
pip install -e .

# Run locally
modelseed-agent analyze
```

### 2. Single Server Deployment

For small teams or dedicated analysis servers:

```bash
# Production installation
pip install modelseed-agent

# System service setup (Ubuntu/Debian)
sudo cp deployment/systemd/modelseed-agent.service /etc/systemd/system/
sudo systemctl enable modelseed-agent
sudo systemctl start modelseed-agent
```

### 3. Container Deployment

Using Docker for consistent environments:

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -s /bin/bash modelseed

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install -e .

# Switch to non-root user
USER modelseed

# Default command
CMD ["modelseed-agent", "serve"]
```

### 4. Kubernetes Deployment

For scalable, cloud-native deployments:

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: modelseed-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: modelseed-agent
  template:
    metadata:
      labels:
        app: modelseed-agent
    spec:
      containers:
      - name: modelseed-agent
        image: modelseed/modelseed-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: anthropic-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Environment Setup

### Production Environment Variables

```bash
# Core configuration
MODELSEED_ENV=production
MODELSEED_DEBUG_LEVEL=WARNING
MODELSEED_LOG_DIR=/var/log/modelseed

# LLM configuration
ANTHROPIC_API_KEY=sk-your-production-key
OPENAI_API_KEY=sk-your-production-key

# Performance settings
MODELSEED_CACHE_ENABLED=true
MODELSEED_CACHE_DIR=/var/cache/modelseed
MODELSEED_PARALLEL_TOOLS=true
MODELSEED_MAX_WORKERS=8

# Security settings
MODELSEED_SECURE_MODE=true
MODELSEED_API_RATE_LIMIT=100
MODELSEED_SESSION_TIMEOUT=3600
```

### Load Balancer Configuration

```nginx
# nginx.conf
upstream modelseed_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name modelseed.example.com;

    location / {
        proxy_pass http://modelseed_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_timeout 300s;
    }

    location /health {
        proxy_pass http://modelseed_backend/health;
        access_log off;
    }
}
```

## Container Orchestration

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  modelseed-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODELSEED_ENV=production
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data:ro
      - logs:/app/logs
      - cache:/app/cache
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "modelseed-agent", "health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ssl_certs:/etc/ssl/certs:ro
    depends_on:
      - modelseed-agent
    restart: unless-stopped

volumes:
  logs:
  cache:
  redis_data:
  ssl_certs:
```

### Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: modelseed

---
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: llm-secrets
  namespace: modelseed
type: Opaque
data:
  anthropic-key: base64-encoded-key
  openai-key: base64-encoded-key

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: modelseed-config
  namespace: modelseed
data:
  config.yaml: |
    llm:
      default_provider: anthropic
      temperature: 0.1
    performance:
      cache_enabled: true
      parallel_tools: true

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: modelseed-agent-service
  namespace: modelseed
spec:
  selector:
    app: modelseed-agent
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: modelseed-agent-ingress
  namespace: modelseed
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - modelseed.example.com
    secretName: modelseed-tls
  rules:
  - host: modelseed.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: modelseed-agent-service
            port:
              number: 80
```

## Monitoring and Observability

### Health Checks

```bash
# Application health endpoint
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# LLM connectivity check
modelseed-agent test-llm-connection
```

### Logging Configuration

```yaml
# logging.yaml
version: 1
formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: json
    filename: /var/log/modelseed/app.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  modelseed:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console]
```

### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
ANALYSIS_REQUESTS = Counter('modelseed_analysis_requests_total', 'Total analysis requests')
ANALYSIS_DURATION = Histogram('modelseed_analysis_duration_seconds', 'Analysis duration')
ACTIVE_SESSIONS = Gauge('modelseed_active_sessions', 'Number of active sessions')
LLM_API_CALLS = Counter('modelseed_llm_api_calls_total', 'Total LLM API calls', ['provider'])

# Start metrics server
start_http_server(8001)
```

### Performance Monitoring

```bash
# System monitoring
htop
iotop
nethogs

# Application monitoring
modelseed-agent monitor --metrics
modelseed-agent monitor --performance

# Log analysis
tail -f /var/log/modelseed/app.log | grep ERROR
journalctl -u modelseed-agent -f
```

## Security Configuration

### SSL/TLS Configuration

```nginx
# SSL configuration in nginx
server {
    listen 443 ssl http2;
    server_name modelseed.example.com;

    ssl_certificate /etc/ssl/certs/modelseed.crt;
    ssl_certificate_key /etc/ssl/private/modelseed.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://modelseed_backend;
        proxy_set_header X-Forwarded-Proto https;
    }
}
```

### Authentication & Authorization

```python
# auth.py
from functools import wraps
import jwt

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return {'error': 'No token provided'}, 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return {'error': 'Invalid token'}, 401

        return f(*args, **kwargs)
    return decorated_function
```

### API Rate Limiting

```python
# rate_limiting.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour", "10 per minute"]
)

@app.route('/analyze')
@limiter.limit("5 per minute")
def analyze():
    # Analysis endpoint
    pass
```

## Scaling Strategies

### Horizontal Scaling

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: modelseed-agent-hpa
  namespace: modelseed
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: modelseed-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Scaling

```bash
# Optimize for CPU-intensive workloads
export MODELSEED_MAX_WORKERS=16
export COBRA_FBA_THREADS=4

# Optimize for memory-intensive workloads
export MODELSEED_MAX_MEMORY_GB=32
export MODELSEED_CACHE_MAX_SIZE=10000
```

### Database Scaling

```yaml
# Redis cluster for caching
redis-cluster:
  enabled: true
  nodes: 6
  persistence:
    enabled: true
    size: 10Gi
```

## Backup and Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/modelseed/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r /app/config "$BACKUP_DIR/"

# Backup data
rsync -av /app/data/ "$BACKUP_DIR/data/"

# Backup logs (last 7 days)
find /app/logs -mtime -7 -type f -exec cp {} "$BACKUP_DIR/logs/" \;

# Backup cache metadata
modelseed-agent export-cache-metadata > "$BACKUP_DIR/cache-metadata.json"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" -C /backup/modelseed "$(basename $BACKUP_DIR)"
rm -rf "$BACKUP_DIR"
```

### Disaster Recovery

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE="$1"
RESTORE_DIR="/app"

# Stop service
systemctl stop modelseed-agent

# Extract backup
tar -xzf "$BACKUP_FILE" -C /tmp/

# Restore files
cp -r /tmp/backup/config/* "$RESTORE_DIR/config/"
cp -r /tmp/backup/data/* "$RESTORE_DIR/data/"

# Restore cache metadata
modelseed-agent import-cache-metadata /tmp/backup/cache-metadata.json

# Start service
systemctl start modelseed-agent

# Verify restoration
modelseed-agent health
```

## Performance Optimization

### Resource Allocation

```yaml
# k8s resource optimization
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Caching Strategy

```python
# Advanced caching
CACHE_CONFIG = {
    'tool_results': {'ttl': 3600, 'max_size': 1000},
    'model_analysis': {'ttl': 7200, 'max_size': 500},
    'llm_responses': {'ttl': 1800, 'max_size': 2000}
}
```

### Connection Pooling

```python
# Database connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

## Troubleshooting

### Common Deployment Issues

#### Container Issues
```bash
# Check container status
docker ps -a
docker logs modelseed-agent

# Debug container
docker exec -it modelseed-agent /bin/bash
```

#### Kubernetes Issues
```bash
# Check pod status
kubectl get pods -n modelseed
kubectl describe pod <pod-name> -n modelseed
kubectl logs <pod-name> -n modelseed

# Debug networking
kubectl exec -it <pod-name> -n modelseed -- nslookup google.com
```

#### Performance Issues
```bash
# Monitor resource usage
kubectl top pods -n modelseed
docker stats

# Check application metrics
curl http://localhost:8001/metrics
```

## CI/CD and Automation

### GitHub Actions Integration

ModelSEEDagent includes comprehensive CI/CD automation:

#### Release Automation
- **Intelligent Version Bumping** based on conventional commits
- **Automated Changelog Generation** with categorized release notes
- **Comprehensive Validation Pipeline** with security scanning
- **PyPI Publishing** with configurable settings

See [Release Automation Guide](operations/release-automation.md) for complete details.

#### Documentation Automation
- **Automatic Documentation Updates** on code changes
- **Tool Count Tracking** and consistency maintenance
- **Content Duplication Prevention** across all documentation
- **Pre-commit Integration** for quality assurance

See [Documentation Automation Guide](operations/documentation-automation.md) for implementation details.

#### Deployment Pipeline Integration

```yaml
# Example: Integrate with deployment workflows
name: Deploy after Release
on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Production
        run: |
          # Your deployment logic here
          kubectl apply -f k8s/
```

## Next Steps

- **[Release Automation](operations/release-automation.md)**: Set up intelligent release management
- **[Documentation Automation](operations/documentation-automation.md)**: Configure automatic documentation updates
- **[Monitoring Guide](monitoring.md)**: Set up comprehensive monitoring
- **Security Best Practices**: Implement proper security measures
- **[Troubleshooting](troubleshooting.md)**: Resolve common issues
- **[API Documentation](api/overview.md)**: Integrate with existing systems
