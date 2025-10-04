# EDA Backend - Sistema Completo de Análise Exploratória de Dados

## 📊 Visão Geral

O **EDA Backend** é uma API REST robusta construída com FastAPI que oferece funcionalidades completas de análise exploratória de dados (EDA) para arquivos CSV. O sistema combina análise estatística avançada, visualizações interativas, machine learning e persistência de dados em uma solução escalável e pronta para produção.

### 🎯 Características Principais

- **Análise Estatística Completa**: Estatísticas descritivas, correlações, testes estatísticos
- **Visualizações Interativas**: Gráficos estáticos (matplotlib/seaborn) e interativos (plotly)
- **Machine Learning**: Clustering, PCA, detecção de outliers
- **Análise Temporal**: Análise de séries temporais com decomposição e tendências
- **Persistência**: SQLAlchemy assíncrono com suporte a SQLite/PostgreSQL
- **Tempo Real**: WebSockets para notificações e progresso de análises
- **Cloud Storage**: Integração com Cloudflare R2
- **Dupla Arquitetura**: Endpoints básicos e aprimorados para diferentes necessidades

## 🏗️ Arquitetura do Sistema

```mermaid
graph TB
    A[Cliente] --> B[FastAPI Router]
    B --> C[Endpoints Básicos]
    B --> D[Endpoints Aprimorados]
    
    C --> E[Serviços de Análise]
    D --> E
    D --> F[WebSocket Manager]
    D --> G[Database Manager]
    
    E --> H[Visualization Service]
    E --> I[Advanced Stats Service]
    E --> J[Temporal Analysis Service]
    E --> K[Statistical Tests Service]
    E --> L[Data Quality Service]
    
    G --> M[(SQLAlchemy Database)]
    F --> N[Real-time Notifications]
    
    O[Cloudflare R2] --> P[File Storage]
    
    subgraph "Core Services"
    H
    I
    J
    K
    L
    end
```

## 📁 Estrutura do Projeto

```
eda-backend/
├── main.py                    # Ponto de entrada da aplicação
├── pyproject.toml             # Configurações e dependências
├── app/
│   ├── api/
│   │   ├── endpoints.py       # 20+ endpoints básicos (2162 linhas)
│   │   └── enhanced_endpoints.py # 11 endpoints aprimorados
│   ├── core/
│   │   ├── database.py        # Gerenciador de banco de dados
│   │   ├── websocket_manager.py # Gerenciador WebSocket
│   │   ├── cloudflare_r2.py   # Serviço Cloudflare R2
│   │   └── config.py          # Configurações
│   ├── models/
│   │   └── database.py        # Modelos SQLAlchemy
│   ├── services/
│   │   ├── visualization_service.py     # Visualizações
│   │   ├── advanced_stats_service.py    # Estatísticas avançadas
│   │   ├── temporal_analysis_service.py # Análise temporal
│   │   ├── statistical_tests_service.py # Testes estatísticos
│   │   ├── data_quality_service.py      # Qualidade dos dados
│   │   ├── machine_learning_service.py  # Machine Learning
│   │   └── persistent_analyzer.py       # Análise persistente
│   └── repositories/
│       └── analysis_repository.py       # Repositório de análises
```

## 🛠️ Tecnologias Utilizadas

### Backend Core
- **FastAPI**: Framework web assíncrono
- **SQLAlchemy 2.0**: ORM assíncrono
- **Pydantic**: Validação de dados
- **Uvicorn**: Servidor ASGI

### Análise de Dados
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **ydata-profiling**: Profiling automático
- **scikit-learn**: Machine learning
- **scipy**: Estatísticas e testes

### Visualização
- **matplotlib**: Gráficos estáticos
- **seaborn**: Visualizações estatísticas
- **plotly**: Gráficos interativos

### Persistência e Cloud
- **SQLite/PostgreSQL**: Banco de dados
- **Cloudflare R2**: Armazenamento de arquivos
- **boto3**: Cliente AWS S3 compatível

### Tempo Real
- **WebSockets**: Comunicação bidirecional
- **asyncio**: Programação assíncrona

## 🚀 Instalação e Configuração

### Pré-requisitos
- Python 3.8+
- pip ou poetry

### Instalação

```bash
# Clone o repositório
git clone <repository-url>
cd eda-backend

# Instale as dependências
pip install -e .

# Configure as variáveis de ambiente
cp .env.example .env
```

### Configuração do .env

```env
# Cloudflare R2
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_ACCESS_KEY_ID=your_access_key
CLOUDFLARE_SECRET_ACCESS_KEY=your_secret_key
CLOUDFLARE_BUCKET_NAME=your_bucket_name
CLOUDFLARE_R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com

# Database
DATABASE_URL=sqlite+aiosqlite:///./eda.db
```

### Executar o Servidor

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📋 API Endpoints

### 🔄 Health Check

#### GET /health
Verifica o status da aplicação.

```bash
curl -X GET "http://localhost:8000/health"
```

**Resposta:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "database": "connected",
  "services": "operational"
}
```

---

### 📤 Upload e Gerenciamento de Arquivos

#### POST /upload-csv
Upload de arquivo CSV com validação automática.

```bash
curl -X POST "http://localhost:8000/upload-csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.csv"
```

**Resposta:**
```json
{
  "message": "CSV uploaded successfully",
  "filename": "data.csv",
  "size": 1024,
  "columns": ["col1", "col2", "col3"],
  "rows": 100,
  "upload_timestamp": "2024-01-15T10:30:00Z"
}
```

#### GET /csv-info
Informações detalhadas sobre o CSV carregado.

```bash
curl -X GET "http://localhost:8000/csv-info"
```

**Resposta:**
```json
{
  "filename": "data.csv",
  "size": 1024,
  "columns": 10,
  "rows": 100,
  "memory_usage": "8.5 KB",
  "dtypes": {
    "col1": "int64",
    "col2": "float64",
    "col3": "object"
  },
  "null_counts": {
    "col1": 0,
    "col2": 5,
    "col3": 2
  }
}
```

---

### 📊 Análise Estatística

#### GET /statistics
Estatísticas descritivas completas.

```bash
curl -X GET "http://localhost:8000/statistics"
```

**Resposta:**
```json
{
  "count": 100,
  "mean": {
    "col1": 50.5,
    "col2": 25.3
  },
  "std": {
    "col1": 28.9,
    "col2": 15.2
  },
  "min": {
    "col1": 1,
    "col2": 0.5
  },
  "max": {
    "col1": 100,
    "col2": 75.8
  },
  "quartiles": {
    "25%": {"col1": 25.75, "col2": 12.6},
    "50%": {"col1": 50.5, "col2": 25.3},
    "75%": {"col1": 75.25, "col2": 38.1}
  }
}
```

#### GET /correlations
Matriz de correlação entre variáveis numéricas.

```bash
curl -X GET "http://localhost:8000/correlations"
```

**Resposta:**
```json
{
  "correlation_matrix": {
    "col1": {"col1": 1.0, "col2": 0.85},
    "col2": {"col1": 0.85, "col2": 1.0}
  },
  "strong_correlations": [
    {"var1": "col1", "var2": "col2", "correlation": 0.85}
  ]
}
```

---

### 🎨 Visualizações

#### POST /plot/histogram
Gera histograma para uma coluna específica.

```bash
curl -X POST "http://localhost:8000/plot/histogram" \
  -H "Content-Type: application/json" \
  -d '{"column": "col1", "bins": 20}'
```

**Resposta:**
```json
{
  "plot_url": "http://localhost:8000/static/plots/histogram_col1_timestamp.png",
  "plot_type": "histogram",
  "column": "col1",
  "statistics": {
    "bins": 20,
    "range": [1, 100],
    "most_frequent_bin": "[45-50)"
  }
}
```

#### POST /plot/scatter
Gráfico de dispersão entre duas variáveis.

```bash
curl -X POST "http://localhost:8000/plot/scatter" \
  -H "Content-Type: application/json" \
  -d '{"x_column": "col1", "y_column": "col2"}'
```

**Resposta:**
```json
{
  "plot_url": "http://localhost:8000/static/plots/scatter_col1_col2_timestamp.png",
  "plot_type": "scatter",
  "correlation": 0.85,
  "trend": "positive"
}
```

#### GET /plot/correlation-heatmap
Mapa de calor das correlações.

```bash
curl -X GET "http://localhost:8000/plot/correlation-heatmap"
```

---

### 🔬 Análise Avançada

#### GET /advanced-statistics
Estatísticas avançadas incluindo assimetria e curtose.

```bash
curl -X GET "http://localhost:8000/advanced-statistics"
```

**Resposta:**
```json
{
  "skewness": {
    "col1": 0.15,
    "col2": -0.32
  },
  "kurtosis": {
    "col1": 2.8,
    "col2": 3.2
  },
  "normality_tests": {
    "col1": {"statistic": 0.95, "p_value": 0.23, "is_normal": true},
    "col2": {"statistic": 0.89, "p_value": 0.01, "is_normal": false}
  }
}
```

#### POST /clustering
Análise de clustering com K-means.

```bash
curl -X POST "http://localhost:8000/clustering" \
  -H "Content-Type: application/json" \
  -d '{"n_clusters": 3, "columns": ["col1", "col2"]}'
```

**Resposta:**
```json
{
  "n_clusters": 3,
  "cluster_centers": [
    [25.3, 12.1],
    [50.5, 25.3],
    [75.8, 38.7]
  ],
  "inertia": 1234.56,
  "silhouette_score": 0.78,
  "cluster_sizes": [35, 40, 25]
}
```

---

### ⏰ Análise Temporal

#### POST /temporal-analysis
Análise de séries temporais.

```bash
curl -X POST "http://localhost:8000/temporal-analysis" \
  -H "Content-Type: application/json" \
  -d '{"date_column": "date", "value_column": "value", "frequency": "D"}'
```

**Resposta:**
```json
{
  "trend": "increasing",
  "seasonality": "weekly",
  "decomposition": {
    "trend_strength": 0.75,
    "seasonal_strength": 0.45,
    "residual_variance": 123.45
  },
  "forecast": {
    "next_7_days": [100.2, 102.1, 98.7, 105.3, 103.9, 101.4, 99.8],
    "confidence_interval": {
      "lower": [95.1, 97.2, 93.8, 100.1, 98.9, 96.3, 94.7],
      "upper": [105.3, 107.0, 103.6, 110.5, 108.9, 106.5, 104.9]
    }
  }
}
```

---

### 🧪 Testes Estatísticos

#### POST /statistical-tests/normality
Testes de normalidade.

```bash
curl -X POST "http://localhost:8000/statistical-tests/normality" \
  -H "Content-Type: application/json" \
  -d '{"columns": ["col1", "col2"]}'
```

**Resposta:**
```json
{
  "tests": {
    "col1": {
      "shapiro_wilk": {"statistic": 0.95, "p_value": 0.23, "is_normal": true},
      "kolmogorov_smirnov": {"statistic": 0.08, "p_value": 0.45, "is_normal": true},
      "anderson_darling": {"statistic": 0.65, "critical_values": [0.574, 0.653], "significance": "5%"}
    }
  }
}
```

#### POST /statistical-tests/compare-groups
Comparação entre grupos.

```bash
curl -X POST "http://localhost:8000/statistical-tests/compare-groups" \
  -H "Content-Type: application/json" \
  -d '{"group_column": "group", "value_column": "value"}'
```

**Resposta:**
```json
{
  "t_test": {
    "statistic": 2.45,
    "p_value": 0.018,
    "significant": true,
    "confidence_interval": [0.5, 3.2]
  },
  "mann_whitney": {
    "statistic": 1234,
    "p_value": 0.022,
    "significant": true
  },
  "levene_test": {
    "statistic": 1.23,
    "p_value": 0.27,
    "equal_variances": true
  }
}
```

---

### 🔍 Qualidade dos Dados

#### GET /data-quality
Relatório completo de qualidade dos dados.

```bash
curl -X GET "http://localhost:8000/data-quality"
```

**Resposta:**
```json
{
  "completeness": {
    "overall": 0.95,
    "by_column": {
      "col1": 1.0,
      "col2": 0.95,
      "col3": 0.90
    }
  },
  "duplicates": {
    "count": 5,
    "percentage": 0.05,
    "duplicate_rows": [15, 23, 45, 67, 89]
  },
  "outliers": {
    "col1": {"count": 3, "indices": [12, 34, 78]},
    "col2": {"count": 2, "indices": [23, 56]}
  },
  "data_types": {
    "correct": 8,
    "needs_conversion": 2,
    "issues": ["col3 contains mixed types"]
  }
}
```

---

### 🤖 Endpoints Aprimorados

#### WebSocket /ws
Conexão WebSocket para notificações em tempo real.

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Notification:', data);
};
```

**Mensagens recebidas:**
```json
{
  "type": "analysis_started",
  "session_id": "uuid",
  "timestamp": "2024-01-15T10:30:00Z"
}

{
  "type": "analysis_progress",
  "session_id": "uuid",
  "progress": 45,
  "current_step": "generating_visualizations"
}

{
  "type": "analysis_completed",
  "session_id": "uuid",
  "analysis_id": "uuid",
  "results_url": "/enhanced/analysis/uuid"
}
```

#### POST /enhanced/analysis/start
Inicia análise completa com persistência.

```bash
curl -X POST "http://localhost:8000/enhanced/analysis/start" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.csv" \
  -F "analysis_name=My Analysis"
```

**Resposta:**
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "session_id": "660e8400-e29b-41d4-a716-446655440001",
  "status": "running",
  "created_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:35:00Z"
}
```

#### GET /enhanced/analysis/{analysis_id}
Recupera análise persistida.

```bash
curl -X GET "http://localhost:8000/enhanced/analysis/550e8400-e29b-41d4-a716-446655440000"
```

**Resposta:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "My Analysis",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:34:30Z",
  "results": {
    "statistics": {...},
    "correlations": {...},
    "visualizations": {...},
    "data_quality": {...}
  },
  "summary": {
    "total_rows": 1000,
    "total_columns": 15,
    "missing_values": 45,
    "key_insights": [
      "Strong correlation between col1 and col2 (0.85)",
      "col3 shows seasonal patterns",
      "5% duplicate records found"
    ]
  }
}
```

#### GET /enhanced/analysis/{analysis_id}/llm-summary
Resumo otimizado para LLM.

```bash
curl -X GET "http://localhost:8000/enhanced/analysis/550e8400-e29b-41d4-a716-446655440000/llm-summary"
```

**Resposta:**
```json
{
  "dataset_overview": {
    "shape": [1000, 15],
    "memory_usage": "125 KB",
    "data_types": {"numeric": 10, "categorical": 3, "datetime": 2}
  },
  "key_statistics": {
    "numeric_summary": "Mean values range from 25.3 to 150.7",
    "correlations": "Strong positive correlation (0.85) between revenue and profit",
    "missing_data": "4.5% missing values, concentrated in optional fields"
  },
  "insights": [
    "Revenue and profit show strong linear relationship",
    "Seasonal patterns detected in monthly data",
    "Customer satisfaction scores follow normal distribution",
    "5% duplicate records require attention"
  ],
  "recommendations": [
    "Investigate revenue-profit relationship for business optimization",
    "Consider seasonal adjustments in forecasting models",
    "Address data quality issues in customer contact information"
  ]
}
```

---

### ☁️ Cloudflare R2 Integration

#### POST /r2/upload
Upload direto para Cloudflare R2.

```bash
curl -X POST "http://localhost:8000/r2/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.csv" \
  -F "key=datasets/analysis_2024.csv"
```

#### GET /r2/presigned-upload
URL pré-assinada para upload.

```bash
curl -X GET "http://localhost:8000/r2/presigned-upload?key=datasets/new_data.csv&expires_in=3600"
```

**Resposta:**
```json
{
  "upload_url": "https://account.r2.cloudflarestorage.com/bucket/datasets/new_data.csv?...",
  "key": "datasets/new_data.csv",
  "expires_in": 3600,
  "fields": {
    "AWSAccessKeyId": "...",
    "policy": "...",
    "signature": "..."
  }
}
```

#### GET /r2/presigned-download
URL pré-assinada para download.

```bash
curl -X GET "http://localhost:8000/r2/presigned-download?key=datasets/analysis_2024.csv&expires_in=1800"
```

---

## 🔄 Fluxo de Análise Completa

### Básico (Stateless)
```mermaid
sequenceDiagram
    participant C as Cliente
    participant API as FastAPI
    participant S as Serviços
    
    C->>API: POST /upload-csv
    API->>S: Valida e processa CSV
    S-->>API: Dados carregados
    API-->>C: Confirmação upload
    
    C->>API: GET /statistics
    API->>S: Calcula estatísticas
    S-->>API: Resultados
    API-->>C: Estatísticas
    
    C->>API: POST /plot/histogram
    API->>S: Gera visualização
    S-->>API: URL do gráfico
    API-->>C: Link do plot
```

### Aprimorado (Persistente com WebSocket)
```mermaid
sequenceDiagram
    participant C as Cliente
    participant WS as WebSocket
    participant API as FastAPI
    participant DB as Database
    participant S as Serviços
    
    C->>WS: Conecta WebSocket
    WS-->>C: Conexão estabelecida
    
    C->>API: POST /enhanced/analysis/start
    API->>DB: Cria registro de análise
    API->>S: Inicia análise assíncrona
    API-->>C: analysis_id + session_id
    
    S->>WS: Progresso 25%
    WS-->>C: Notificação progresso
    
    S->>WS: Progresso 50%
    WS-->>C: Notificação progresso
    
    S->>DB: Salva resultados
    S->>WS: Análise completa
    WS-->>C: Notificação conclusão
    
    C->>API: GET /enhanced/analysis/{id}
    API->>DB: Recupera análise
    DB-->>API: Dados persistidos
    API-->>C: Resultados completos
```

## 🗄️ Modelo de Dados

### Analysis Model
```python
class Analysis(Base):
    __tablename__ = "analyses"
    
    id: UUID = Column(UUID, primary_key=True)
    name: str = Column(String, nullable=False)
    filename: str = Column(String, nullable=False)
    status: AnalysisStatus = Column(Enum(AnalysisStatus))
    results: dict = Column(JSON)
    summary: dict = Column(JSON)
    visualizations: dict = Column(JSON)
    error_message: Optional[str] = Column(Text)
    created_at: datetime = Column(DateTime)
    completed_at: Optional[datetime] = Column(DateTime)
```

### AnalysisSession Model
```python
class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"
    
    id: UUID = Column(UUID, primary_key=True)
    analysis_id: UUID = Column(UUID, ForeignKey("analyses.id"))
    websocket_id: str = Column(String)
    status: SessionStatus = Column(Enum(SessionStatus))
    progress: int = Column(Integer, default=0)
    current_step: Optional[str] = Column(String)
    created_at: datetime = Column(DateTime)
```

## 🔧 Configuração Avançada

### Variáveis de Ambiente Completas

```env
# Database
DATABASE_URL=sqlite+aiosqlite:///./eda.db
# Para PostgreSQL:
# DATABASE_URL=postgresql+asyncpg://user:password@localhost/eda_db

# Cloudflare R2
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_ACCESS_KEY_ID=your_access_key
CLOUDFLARE_SECRET_ACCESS_KEY=your_secret_key
CLOUDFLARE_BUCKET_NAME=your_bucket_name
CLOUDFLARE_R2_ENDPOINT_URL=https://your_account_id.r2.cloudflarestorage.com

# Application
DEBUG=False
MAX_FILE_SIZE=100MB
ALLOWED_EXTENSIONS=csv,xlsx,tsv
STATIC_FILES_PATH=./static
PLOTS_PATH=./static/plots

# WebSocket
WS_HEARTBEAT_INTERVAL=30
WS_MAX_CONNECTIONS=100

# Analysis
MAX_ANALYSIS_TIME=300
CLEANUP_INTERVAL=3600
AUTO_DELETE_AFTER_DAYS=30
```

### Docker Support

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install -e .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  eda-backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://eda:password@db:5432/eda_db
    depends_on:
      - db
    volumes:
      - ./static:/app/static

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: eda_db
      POSTGRES_USER: eda
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

## 🧪 Testes

### Executar Testes
```bash
# Testes unitários
pytest tests/unit/

# Testes de integração
pytest tests/integration/

# Testes end-to-end
pytest tests/e2e/

# Cobertura
pytest --cov=app tests/
```

### Exemplo de Teste
```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_upload_csv():
    with open("test_data.csv", "rb") as f:
        response = client.post(
            "/upload-csv",
            files={"file": ("test.csv", f, "text/csv")}
        )
    assert response.status_code == 200
    assert "CSV uploaded successfully" in response.json()["message"]

def test_statistics():
    response = client.get("/statistics")
    assert response.status_code == 200
    data = response.json()
    assert "mean" in data
    assert "std" in data
```

## 🚀 Deploy em Produção

### Railway
```bash
# Instalar Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

### Heroku
```bash
# Procfile
web: uvicorn main:app --host 0.0.0.0 --port $PORT

# Deploy
heroku create eda-backend-app
git push heroku main
```

### AWS EC2/ECS
```bash
# Build e push para ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t eda-backend .
docker tag eda-backend:latest <account>.dkr.ecr.us-east-1.amazonaws.com/eda-backend:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/eda-backend:latest
```

## 📊 Métricas e Monitoramento

### Health Checks Detalhados
```python
# Endpoint /health fornece:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "database": "connected",
  "services": "operational",
  "memory_usage": "245 MB",
  "active_connections": 12,
  "uptime": "2d 5h 30m"
}
```

### Logs Estruturados
```python
import logging
import structlog

logger = structlog.get_logger()

# Logs incluem:
logger.info("Analysis started", 
           analysis_id=analysis_id,
           filename=filename,
           user_session=session_id)

logger.error("Analysis failed",
            analysis_id=analysis_id,
            error=str(e),
            duration=time_elapsed)
```

## 🛡️ Segurança

### Validações Implementadas
- **Upload**: Validação de tipo de arquivo e tamanho
- **Input**: Sanitização de parâmetros de entrada
- **Rate Limiting**: Proteção contra abuse de API
- **CORS**: Configuração adequada para frontend
- **Headers**: Cabeçalhos de segurança HTTP

### Exemplo de Configuração de Segurança
```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "*.yourdomain.com"]
)
```

## 🔄 Roadmap Futuro

### Funcionalidades Planejadas
- [ ] **Autenticação JWT**: Sistema completo de usuários
- [ ] **Cache Redis**: Cache inteligente para análises frequentes
- [ ] **Análise ML Avançada**: AutoML e feature engineering
- [ ] **Dashboard Web**: Interface React para visualizações
- [ ] **API GraphQL**: Queries flexíveis complementando REST
- [ ] **Streaming Processing**: Análise em tempo real de dados
- [ ] **Multi-tenancy**: Suporte a múltiplos clientes
- [ ] **Backup/Restore**: Sistema de backup automático

### Melhorias Técnicas
- [ ] **Observabilidade**: OpenTelemetry e Jaeger
- [ ] **Testing**: Cobertura 95%+ com testes automatizados
- [ ] **Performance**: Otimização com Numba/Cython
- [ ] **Scalability**: Kubernetes deployment
- [ ] **Security**: OAuth2 e audit logging
- [ ] **Documentation**: OpenAPI completa com exemplos

## 📞 Suporte e Contribuição

### Como Contribuir
1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Reportar Bugs
Use as issues do GitHub com o template:
```
**Descrição do Bug**
Descrição clara do problema

**Para Reproduzir**
Passos para reproduzir o comportamento

**Comportamento Esperado**
O que deveria acontecer

**Screenshots**
Se aplicável, adicione screenshots

**Informações do Ambiente**
- OS: [e.g. Ubuntu 20.04]
- Python: [e.g. 3.11]
- Versão: [e.g. 1.0.0]
```

### Contato
- **Email**: suporte@eda-backend.com
- **Issues**: GitHub Issues
- **Documentation**: Wiki do projeto

---

## 📜 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

**EDA Backend** - *Transformando dados em insights com análise exploratória completa* 🚀📊

## Estrutura do Projeto

```
eda-backend/
├── app/
│   ├── api/
│   │   └── endpoints.py        # Endpoints da API
│   ├── core/
│   │   └── eda_processor.py    # Processamento EDA
│   └── models/
│       └── responses.py        # Modelos de resposta
├── main.py                     # Aplicação principal
├── pyproject.toml             # Configuração do Poetry
└── README.md                  # Este arquivo
```

## Desenvolvimento

### Formatação de código

```bash
poetry run black .
poetry run isort .
```

### Verificação de código

```bash
poetry run flake8 .
```

### Testes

```bash
poetry run pytest
```