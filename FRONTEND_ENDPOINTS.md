# Frontend API Endpoints Reference

## Overview

Your frontend UI needs to connect to **two services**:

1. **Python Backend** (`http://localhost:8000` or deployed URL)
   - Main chat endpoint
   - Document upload
   - Health checks

2. **Mastra Cloud** (if using Mastra Studio/Cloud)
   - Agent endpoints
   - Workflow endpoints

## Python Backend Endpoints

### Base URL
- **Local**: `http://localhost:8000`
- **Production**: `https://your-python-api.railway.app` (or your deployed URL)

### 1. Chat Endpoint (Main)

**POST** `/chat`

Send messages and get responses.

**Request:**
```json
{
  "message": "How many audits are there?",
  "conversation_id": "optional-conversation-id",
  "context_type": "excel" | "document" | null,
  "resource_id": "optional-resource-id",
  "thread_id": "optional-thread-id",
  "metadata": { "key": "value" }
}
```

**Response:**
```json
{
  "response": "There are 37 audits in the dataset...",
  "conversation_id": "uuid-here",
  "context_used": "excel",
  "sources": null
}
```

**Example:**
```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'How many audits are there?'
  })
});
const data = await response.json();
console.log(data.response);
```

### 2. Health Check

**GET** `/health`

Check if backend is running.

**Response:**
```json
{
  "status": "healthy",
  "service": "Audit Assistant API",
  "version": "1.0.0",
  "context_files_loaded": 3,
  "documents_loaded": 0
}
```

### 3. Context Summary

**GET** `/context/summary`

Get information about loaded Excel context files.

**Response:**
```json
{
  "context_files": ["audits", "issues", "workpapers"],
  "summary": "AUDIT SYSTEM CONTEXT...",
  "details": {
    "audits": {
      "rows": 37,
      "columns": 20,
      "column_names": ["Audit Title", "Status", ...]
    }
  }
}
```

### 4. Upload Document

**POST** `/upload`

Upload a document (PDF, DOCX, TXT) for analysis.

**Request:** `multipart/form-data` with file

**Response:**
```json
{
  "document_id": "uuid-here",
  "filename": "document.pdf",
  "original_name": "document.pdf",
  "size_bytes": 12345,
  "size_mb": 0.01,
  "upload_time": "2026-01-22T...",
  "summary": "Document summary...",
  "file_type": ".pdf"
}
```

**Example:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/upload', {
  method: 'POST',
  body: formData
});
const data = await response.json();
```

### 5. List Documents

**GET** `/documents`

Get list of all uploaded documents.

**Response:**
```json
{
  "documents": [
    {
      "filename": "document.pdf",
      "size_bytes": 12345,
      "size_mb": 0.01,
      "modified": "2026-01-22T...",
      "file_type": ".pdf"
    }
  ],
  "count": 1
}
```

### 6. Delete Document

**DELETE** `/documents/{document_id}`

Delete an uploaded document.

**Response:**
```json
{
  "message": "Document {document_id} deleted successfully"
}
```

## Mastra Cloud Endpoints

If you deploy to Mastra Cloud, your agent will be available at:

### Agent Generate

**POST** `https://your-project.mastra.cloud/api/agents/audit-chatbot/generate`

**Request:**
```json
{
  "messages": [
    { "role": "user", "content": "How many audits are there?" }
  ]
}
```

**Response:**
```json
{
  "text": "There are 37 audits...",
  "agent_id": "audit-chatbot"
}
```

## Frontend Integration Example

### React/Next.js Example

```typescript
// Chat component
const [messages, setMessages] = useState([]);
const [loading, setLoading] = useState(false);

const sendMessage = async (message: string) => {
  setLoading(true);
  
  try {
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
      }),
    });
    
    const data = await response.json();
    
    setMessages(prev => [
      ...prev,
      { role: 'user', content: message },
      { role: 'assistant', content: data.response }
    ]);
  } catch (error) {
    console.error('Error:', error);
  } finally {
    setLoading(false);
  }
};
```

### Vanilla JavaScript Example

```javascript
async function chat(message) {
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });
  
  const data = await response.json();
  return data.response;
}

// Usage
chat('How many audits are there?').then(response => {
  console.log(response);
});
```

## CORS Configuration

Your Python backend already has CORS enabled:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

This means your frontend can call the API from any domain.

## Environment Variables for Frontend

Create a `.env` file in your frontend project:

```env
REACT_APP_API_URL=http://localhost:8000
# Or in production:
REACT_APP_API_URL=https://your-python-api.railway.app
```

Then use in your code:
```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
```

## Summary

**Essential endpoints for frontend:**
1. ✅ `POST /chat` - Main chat endpoint
2. ✅ `GET /health` - Health check
3. ✅ `POST /upload` - Upload documents
4. ✅ `GET /documents` - List documents
5. ✅ `GET /context/summary` - Get context info

**Optional endpoints:**
- `DELETE /documents/{id}` - Delete documents

All endpoints return JSON and support CORS! 🚀
