# Cloudflare R2 Upload - Guia de Uso

Este documento explica como usar a funcionalidade de upload para Cloudflare R2 com URLs pré-assinadas.

## Configuração

### 1. Variáveis de Ambiente

Crie um arquivo `.env` baseado no `.env.example` e configure:

```bash
# Cloudflare R2 Credentials
CLOUDFLARE_R2_ACCESS_KEY_ID=seu_access_key_aqui
CLOUDFLARE_R2_SECRET_ACCESS_KEY=sua_secret_key_aqui
CLOUDFLARE_R2_ENDPOINT_URL=https://seu-account-id.r2.cloudflarestorage.com
CLOUDFLARE_R2_BUCKET_NAME=nome-do-seu-bucket
CLOUDFLARE_R2_REGION=auto

# Upload Settings
MAX_FILE_SIZE_MB=100
PRESIGNED_URL_EXPIRATION_SECONDS=3600
```

### 2. Obtenção das Credenciais

1. Acesse o [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Vá para R2 Object Storage
3. Crie um bucket se ainda não tiver
4. Em "Manage R2 API tokens", crie um novo token
5. Configure as permissões necessárias (Read, Write)

## Endpoints Disponíveis

### 1. Verificar Configuração
```http
GET /api/v1/r2/config
```

Retorna o status da configuração do R2.

### 2. Gerar URL Pré-assinada para Upload
```http
POST /api/v1/r2/presigned-upload?filename=arquivo.csv&folder=uploads
```

**Parâmetros:**
- `filename` (obrigatório): Nome do arquivo
- `content_type` (opcional): Tipo MIME do arquivo
- `folder` (opcional): Pasta de destino (padrão: "uploads")

**Resposta:**
```json
{
  "success": true,
  "upload_url": "https://...",
  "method": "PUT",
  "file_key": "uploads/2024/10/01/uuid.csv",
  "content_type": "text/csv",
  "expires_in": 3600,
  "max_file_size_mb": 100,
  "expires_at": "2024-10-01T15:00:00",
  "headers": {
    "Content-Type": "text/csv",
    "x-amz-meta-original-filename": "arquivo.csv",
    "x-amz-meta-upload-timestamp": "2024-10-01T12:00:00"
  }
}
```

### 3. Upload do Arquivo

Use a URL retornada para fazer upload via PUT:

```bash
# Exemplo com curl
curl -X PUT "https://upload-url..." \
  -H "Content-Type: text/csv" \
  -H "x-amz-meta-original-filename: arquivo.csv" \
  --data-binary @arquivo.csv
```

```javascript
// Exemplo em JavaScript
const response = await fetch(upload_url, {
  method: 'PUT',
  headers: headers,
  body: file
});
```

### 4. Obter Informações do Arquivo
```http
GET /api/v1/r2/file-info/{file_key}
```

### 5. Gerar URL de Download
```http
GET /api/v1/r2/presigned-download/{file_key}
```

### 6. Listar Arquivos
```http
GET /api/v1/r2/files?folder=uploads&limit=50
```

### 7. Deletar Arquivo
```http
DELETE /api/v1/r2/file/{file_key}
```

## Fluxo Completo de Upload

### Passo 1: Obter URL Pré-assinada
```bash
curl -X POST "http://localhost:8000/api/v1/r2/presigned-upload?filename=dados.csv"
```

### Passo 2: Upload do Arquivo
```bash
# Use os dados retornados do passo 1
curl -X PUT "https://upload-url..." \
  -H "Content-Type: text/csv" \
  -H "x-amz-meta-original-filename: dados.csv" \
  --data-binary @dados.csv
```

### Passo 3: Verificar Upload
```bash
curl "http://localhost:8000/api/v1/r2/file-info/uploads/2024/10/01/uuid.csv"
```

### Passo 4: Processar Arquivo (Futuro)
```bash
curl -X POST "http://localhost:8000/api/v1/r2/upload-and-process?file_key=uploads/2024/10/01/uuid.csv"
```

## Exemplos em Diferentes Linguagens

### Python
```python
import requests
from pathlib import Path

# 1. Obter URL pré-assinada
response = requests.post(
    "http://localhost:8000/api/v1/r2/presigned-upload",
    params={"filename": "dados.csv"}
)
upload_data = response.json()

# 2. Fazer upload
with open("dados.csv", "rb") as file_data:
    headers = upload_data["headers"]
    response = requests.put(
        upload_data["upload_url"], 
        data=file_data, 
        headers=headers
    )
```

### JavaScript (Browser)
```javascript
// 1. Obter URL pré-assinada
const response = await fetch('/api/v1/r2/presigned-upload?filename=dados.csv', {
  method: 'POST'
});
const uploadData = await response.json();

// 2. Fazer upload
await fetch(uploadData.upload_url, {
  method: 'PUT',
  headers: uploadData.headers,
  body: fileInput.files[0]
});
```

## Vantagens das URLs Pré-assinadas

1. **Segurança**: Credenciais não expostas no frontend
2. **Performance**: Upload direto para R2, sem passar pelo servidor
3. **Escalabilidade**: Reduz carga no servidor da API
4. **Controle**: Limites de tamanho e tempo configuráveis
5. **Flexibilidade**: Suporte a qualquer tipo de arquivo
6. **Compatibilidade**: Usa PUT requests (suportado pelo R2)

## Limitações e Considerações

- URLs expiram em 1 hora por padrão
- Tamanho máximo de 100MB por padrão
- Arquivos são organizados por data automaticamente
- Nomes únicos (UUID) evitam conflitos
- **Importante**: R2 não suporta presigned POST, apenas PUT/GET

## Instalação das Dependências

```bash
# Instalar dependências
poetry add boto3 botocore pydantic-settings

# Ou com pip
pip install boto3 botocore pydantic-settings
```