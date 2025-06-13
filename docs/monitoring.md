# Monitoring Guide

Comprehensive monitoring setup for ModelSEEDagent in production environments.

## Overview

ModelSEEDagent provides multiple monitoring capabilities:

- **Health Monitoring**: System and service health checks
- **Performance Monitoring**: Resource usage and performance metrics
- **Application Monitoring**: Tool execution and analysis tracking
- **Security Monitoring**: Access control and audit logging
- **Business Monitoring**: Usage analytics and cost tracking

## Health Monitoring

### Built-in Health Checks

```bash
# Basic health check
modelseed-agent health

# Detailed health check with component status
modelseed-agent health --detailed

# Health check with LLM connectivity test
modelseed-agent health --test-llm

# JSON output for monitoring systems
modelseed-agent health --format json
```

### Health Check Endpoints

```python
# health.py - Health check endpoints
from flask import Flask, jsonify
import psutil
import time

app = Flask(__name__)

@app.route('/health')
def health():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    })

@app.route('/health/ready')
def readiness():
    """Readiness probe for Kubernetes"""
    # Check if application can serve requests
    try:
        # Test database connection
        # Test LLM connectivity
        # Check required services
        return jsonify({'status': 'ready'})
    except Exception as e:
        return jsonify({'status': 'not ready', 'error': str(e)}), 503

@app.route('/health/live')
def liveness():
    """Liveness probe for Kubernetes"""
    # Check if application is alive
    return jsonify({
        'status': 'alive',
        'uptime': get_uptime(),
        'memory_usage': psutil.virtual_memory().percent,
        'cpu_usage': psutil.cpu_percent()
    })

@app.route('/health/detailed')
def detailed_health():
    """Comprehensive health information"""
    return jsonify({
        'status': 'healthy',
        'components': {
            'llm_connectivity': check_llm_connectivity(),
            'database': check_database(),
            'file_system': check_file_system(),
            'memory': check_memory(),
            'disk_space': check_disk_space()
        },
        'metrics': get_system_metrics()
    })
```

### Kubernetes Health Checks

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: modelseed-agent
spec:
  template:
    spec:
      containers:
      - name: modelseed-agent
        image: modelseed/modelseed-agent:latest
        ports:
        - containerPort: 8000

        # Liveness probe
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        # Readiness probe
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        # Startup probe
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 30
```

## Performance Monitoring

### System Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import time

# Define metrics
REQUEST_COUNT = Counter('modelseed_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('modelseed_request_duration_seconds', 'Request duration')
ACTIVE_ANALYSES = Gauge('modelseed_active_analyses', 'Number of active analyses')
MEMORY_USAGE = Gauge('modelseed_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('modelseed_cpu_usage_percent', 'CPU usage percentage')

# LLM-specific metrics
LLM_REQUESTS = Counter('modelseed_llm_requests_total', 'LLM API requests', ['provider', 'model'])
LLM_DURATION = Histogram('modelseed_llm_duration_seconds', 'LLM request duration', ['provider'])
LLM_ERRORS = Counter('modelseed_llm_errors_total', 'LLM API errors', ['provider', 'error_type'])

# Tool-specific metrics
TOOL_EXECUTIONS = Counter('modelseed_tool_executions_total', 'Tool executions', ['tool_name'])
TOOL_DURATION = Histogram('modelseed_tool_duration_seconds', 'Tool execution duration', ['tool_name'])
TOOL_ERRORS = Counter('modelseed_tool_errors_total', 'Tool execution errors', ['tool_name'])

def collect_system_metrics():
    """Collect system metrics periodically"""
    while True:
        MEMORY_USAGE.set(psutil.virtual_memory().used)
        CPU_USAGE.set(psutil.cpu_percent())
        time.sleep(30)

# Start metrics server
start_http_server(8001)
```

### Application Performance Monitoring

```python
# performance.py
import time
import functools
from contextlib import contextmanager

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}

    @contextmanager
    def measure(self, operation_name):
        """Context manager for measuring operation duration"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_duration(operation_name, duration)

    def record_duration(self, operation, duration):
        """Record operation duration"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)

        # Update Prometheus metrics
        REQUEST_DURATION.observe(duration)

    def timed(self, operation_name):
        """Decorator for timing function calls"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure(operation_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

# Usage example
monitor = PerformanceMonitor()

@monitor.timed('model_analysis')
def analyze_model(model_path):
    # Analysis implementation
    pass

# Context manager usage
with monitor.measure('fba_execution'):
    result = run_fba(model)
```

### Resource Monitoring

```bash
#!/bin/bash
# monitor_resources.sh

LOG_FILE="/var/log/modelseed/resources.log"

while true; do
    timestamp=$(date -Iseconds)

    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

    # Memory usage
    memory_info=$(free -m | grep "Mem:")
    memory_used=$(echo $memory_info | awk '{print $3}')
    memory_total=$(echo $memory_info | awk '{print $2}')
    memory_percent=$((memory_used * 100 / memory_total))

    # Disk usage
    disk_usage=$(df -h / | tail -1 | awk '{print $5}' | cut -d'%' -f1)

    # Process count
    process_count=$(pgrep -f modelseed-agent | wc -l)

    # Log metrics
    echo "$timestamp,CPU:$cpu_usage,Memory:$memory_percent,Disk:$disk_usage,Processes:$process_count" >> $LOG_FILE

    sleep 60
done
```

## Application Monitoring

### Tool Execution Tracking

```python
# tool_monitoring.py
import json
import time
import logging
from pathlib import Path

class ToolAuditor:
    def __init__(self, audit_dir="logs/tool_audits"):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(exist_ok=True)

    def log_execution(self, tool_name, inputs, outputs, duration, success=True, error=None):
        """Log tool execution details"""
        audit_data = {
            'timestamp': time.time(),
            'tool_name': tool_name,
            'duration': duration,
            'success': success,
            'inputs': self._sanitize_inputs(inputs),
            'outputs': self._sanitize_outputs(outputs),
            'error': str(error) if error else None,
            'memory_usage': self._get_memory_usage(),
            'cpu_usage': self._get_cpu_usage()
        }

        # Save audit log
        audit_file = self.audit_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{tool_name}_{id(audit_data)}.json"
        with open(audit_file, 'w') as f:
            json.dump(audit_data, f, indent=2)

        # Update metrics
        TOOL_EXECUTIONS.labels(tool_name=tool_name).inc()
        TOOL_DURATION.labels(tool_name=tool_name).observe(duration)

        if not success:
            TOOL_ERRORS.labels(tool_name=tool_name).inc()

    def _sanitize_inputs(self, inputs):
        """Remove sensitive data from inputs"""
        # Implementation to sanitize inputs
        return inputs

    def _sanitize_outputs(self, outputs):
        """Remove sensitive data from outputs"""
        # Implementation to sanitize outputs
        return outputs
```

### Agent Workflow Monitoring

```python
# workflow_monitoring.py
class WorkflowMonitor:
    def __init__(self):
        self.workflows = {}

    def start_workflow(self, workflow_id, workflow_type):
        """Start monitoring a workflow"""
        self.workflows[workflow_id] = {
            'start_time': time.time(),
            'type': workflow_type,
            'steps': [],
            'status': 'running'
        }

        ACTIVE_ANALYSES.inc()

    def log_step(self, workflow_id, step_name, duration, success=True):
        """Log a workflow step"""
        if workflow_id in self.workflows:
            self.workflows[workflow_id]['steps'].append({
                'name': step_name,
                'duration': duration,
                'success': success,
                'timestamp': time.time()
            })

    def end_workflow(self, workflow_id, success=True):
        """End workflow monitoring"""
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            workflow['end_time'] = time.time()
            workflow['total_duration'] = workflow['end_time'] - workflow['start_time']
            workflow['status'] = 'completed' if success else 'failed'

            # Archive workflow data
            self._archive_workflow(workflow_id, workflow)

            ACTIVE_ANALYSES.dec()
```

## Logging and Observability

### Structured Logging

```python
# logging_config.py
import logging
import json
import sys
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'process': record.process
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add custom fields
        if hasattr(record, 'tool_name'):
            log_entry['tool_name'] = record.tool_name
        if hasattr(record, 'workflow_id'):
            log_entry['workflow_id'] = record.workflow_id

        return json.dumps(log_entry)

# Configure structured logging
def setup_logging():
    logger = logging.getLogger('modelseed')
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)

    return logger
```

### Log Aggregation

```yaml
# fluentd/fluent.conf
<source>
  @type tail
  path /var/log/modelseed/*.log
  pos_file /var/log/fluentd/modelseed.log.pos
  tag modelseed.*
  format json
  time_key timestamp
  time_format %Y-%m-%dT%H:%M:%S.%L%z
</source>

<filter modelseed.**>
  @type record_transformer
  <record>
    hostname "#{Socket.gethostname}"
    service modelseed-agent
  </record>
</filter>

<match modelseed.**>
  @type elasticsearch
  host elasticsearch.logging.svc.cluster.local
  port 9200
  index_name modelseed-logs
  type_name _doc
</match>
```

## Alerting

### Prometheus Alerting Rules

```yaml
# alerts.yml
groups:
- name: modelseed
  rules:

  # High error rate
  - alert: HighErrorRate
    expr: rate(modelseed_tool_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  # High memory usage
  - alert: HighMemoryUsage
    expr: modelseed_memory_usage_bytes / (1024*1024*1024) > 8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}GB"

  # LLM API failures
  - alert: LLMAPIFailures
    expr: rate(modelseed_llm_errors_total[5m]) > 0.05
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "LLM API failures detected"
      description: "LLM failure rate: {{ $value }} per second"

  # Service down
  - alert: ServiceDown
    expr: up{job="modelseed-agent"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "ModelSEEDagent service is down"
      description: "Service has been down for more than 1 minute"
```

### Alert Manager Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'smtp.company.com:587'
  smtp_from: 'alerts@company.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'admin@company.com'
    subject: 'ModelSEEDagent Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}

  slack_configs:
  - api_url: 'https://hooks.slack.com/services/...'
    channel: '#alerts'
    title: 'ModelSEEDagent Alert'
    text: '{{ .CommonAnnotations.summary }}'
```

## Dashboard Creation

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "ModelSEEDagent Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(modelseed_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, modelseed_request_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, modelseed_request_duration_seconds_bucket)",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "modelseed_memory_usage_bytes / (1024*1024*1024)",
            "legendFormat": "Memory (GB)"
          },
          {
            "expr": "modelseed_cpu_usage_percent",
            "legendFormat": "CPU %"
          }
        ]
      },
      {
        "title": "Tool Execution Success Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(modelseed_tool_executions_total[5m]) - rate(modelseed_tool_errors_total[5m])) / rate(modelseed_tool_executions_total[5m]) * 100",
            "legendFormat": "Success Rate %"
          }
        ]
      }
    ]
  }
}
```

### Custom Monitoring Dashboard

```python
# dashboard.py
import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd

app = dash.Dash(__name__)

def get_metrics_data():
    """Fetch metrics from Prometheus"""
    # Implementation to fetch metrics
    return pd.DataFrame()

app.layout = html.Div([
    html.H1('ModelSEEDagent Monitoring Dashboard'),

    dcc.Graph(
        id='request-rate',
        figure={
            'data': [
                go.Scatter(
                    x=df['timestamp'],
                    y=df['request_rate'],
                    mode='lines',
                    name='Request Rate'
                )
            ],
            'layout': go.Layout(
                title='Request Rate Over Time',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Requests/second'}
            )
        }
    ),

    dcc.Graph(
        id='tool-performance',
        figure={
            'data': [
                go.Bar(
                    x=tool_metrics['tool_name'],
                    y=tool_metrics['avg_duration'],
                    name='Average Duration'
                )
            ],
            'layout': go.Layout(
                title='Tool Performance',
                xaxis={'title': 'Tool'},
                yaxis={'title': 'Duration (seconds)'}
            )
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
```

## Security Monitoring

### Audit Logging

```python
# security_audit.py
import logging
from datetime import datetime

class SecurityAuditor:
    def __init__(self):
        self.audit_logger = logging.getLogger('modelseed.security')

    def log_access(self, user_id, resource, action, success=True):
        """Log access attempts"""
        self.audit_logger.info(
            "Access attempt",
            extra={
                'user_id': user_id,
                'resource': resource,
                'action': action,
                'success': success,
                'timestamp': datetime.utcnow().isoformat(),
                'ip_address': self._get_client_ip()
            }
        )

    def log_api_key_usage(self, api_key_id, provider, success=True):
        """Log API key usage"""
        self.audit_logger.info(
            "API key usage",
            extra={
                'api_key_id': api_key_id,
                'provider': provider,
                'success': success,
                'timestamp': datetime.utcnow().isoformat()
            }
        )

    def log_sensitive_operation(self, operation, user_id, details=None):
        """Log sensitive operations"""
        self.audit_logger.warning(
            "Sensitive operation",
            extra={
                'operation': operation,
                'user_id': user_id,
                'details': details,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
```

### Intrusion Detection

```bash
#!/bin/bash
# intrusion_detection.sh

LOG_FILE="/var/log/modelseed/security.log"
ALERT_THRESHOLD=10

# Monitor failed login attempts
failed_logins=$(grep "authentication failed" $LOG_FILE | grep "$(date +%Y-%m-%d)" | wc -l)

if [ $failed_logins -gt $ALERT_THRESHOLD ]; then
    echo "ALERT: $failed_logins failed login attempts detected today" | \
    mail -s "Security Alert: Multiple Failed Logins" admin@company.com
fi

# Monitor unusual API usage patterns
unusual_api_usage=$(grep "API rate limit" $LOG_FILE | grep "$(date +%Y-%m-%d)" | wc -l)

if [ $unusual_api_usage -gt 5 ]; then
    echo "ALERT: Unusual API usage patterns detected" | \
    mail -s "Security Alert: API Abuse" admin@company.com
fi
```

## Cost Monitoring

### LLM Usage Tracking

```python
# cost_monitoring.py
class CostMonitor:
    def __init__(self):
        self.usage_tracker = {}

    def track_llm_usage(self, provider, model, input_tokens, output_tokens, cost=None):
        """Track LLM API usage and costs"""
        date = datetime.now().date()
        key = f"{provider}_{model}_{date}"

        if key not in self.usage_tracker:
            self.usage_tracker[key] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'requests': 0,
                'cost': 0.0
            }

        self.usage_tracker[key]['input_tokens'] += input_tokens
        self.usage_tracker[key]['output_tokens'] += output_tokens
        self.usage_tracker[key]['requests'] += 1

        if cost:
            self.usage_tracker[key]['cost'] += cost

        # Alert if cost threshold exceeded
        daily_cost = self.usage_tracker[key]['cost']
        if daily_cost > 100:  # $100 daily limit
            self._send_cost_alert(provider, model, daily_cost)

    def _send_cost_alert(self, provider, model, cost):
        """Send cost alert"""
        logging.warning(
            f"Daily cost threshold exceeded: {provider} {model} - ${cost:.2f}"
        )
```

This monitoring setup provides comprehensive observability for ModelSEEDagent in production environments. Regular monitoring and alerting help ensure system reliability, performance, and security.
