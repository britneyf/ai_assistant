# API Endpoints for Frontend UI

## Quick Reference

Your frontend needs these endpoints to connect to your AI assistant:

## Python Backend Endpoints

**Base URL:** `http://localhost:8000` (local) or `https://your-api.railway.app` (production)

### 1. Chat (Main Endpoint) ⭐

**POST** `/chat`

Send a message and get a response.

```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'How many audits are there?',
    conversation_id: 'optional-conversation-id',
    context_type: 'excel' | 'document' | null,
    resource_id: 'optional-resource-id',
    thread_id: 'optional-thread-id'
  })
});

const data = await response.json();
// data.response = "There are 37 audits..."
// data.conversation_id = "uuid"
// data.context_used = "excel"
```

### 2. Health Check

**GET** `/health`

Check if backend is running.

```javascript
const response = await fetch('http://localhost:8000/health');
const data = await response.json();
// { status: "healthy", context_files_loaded: 3, ... }
```

### 3. Upload Document

**POST** `/upload`

Upload a file (PDF, DOCX, TXT) for analysis.

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/upload', {
  method: 'POST',
  body: formData
});

const data = await response.json();
// { document_id: "uuid", summary: "...", ... }
```

### 4. List Documents

**GET** `/documents`

Get all uploaded documents.

```javascript
const response = await fetch('http://localhost:8000/documents');
const data = await response.json();
// { documents: [...], count: 1 }
```

### 5. Context Summary

**GET** `/context/summary`

Get information about loaded audit data.

```javascript
const response = await fetch('http://localhost:8000/context/summary');
const data = await response.json();
// { context_files: ["audits", "issues", "workpapers"], ... }
```

## Mastra Cloud Endpoints (If Deployed)

**Base URL:** `https://your-project.mastra.cloud`

### Agent Generate

**POST** `/api/agents/audit-chatbot/generate`

```javascript
const response = await fetch('https://your-project.mastra.cloud/api/agents/audit-chatbot/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [
      { role: 'user', content: 'How many audits are there?' }
    ]
  })
});

const data = await response.json();
// { text: "There are 37 audits...", agent_id: "audit-chatbot" }
```

## Frontend Integration Example

### React Hook Example

```typescript
import { useState } from 'react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async (message: string, conversationId?: string) => {
    setLoading(true);
    
    try {
      const response = await fetch(`${API_URL}/chat`, {
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
      
      return data.conversation_id;
    } catch (error) {
      console.error('Error:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  return { messages, sendMessage, loading };
}
```

### Complete Frontend Example

```typescript
// ChatComponent.tsx
import { useState } from 'react';

const API_URL = 'http://localhost:8000';

export function ChatComponent() {
  const [message, setMessage] = useState('');
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!message.trim()) return;
    
    setLoading(true);
    
    // Add user message to UI
    setMessages(prev => [...prev, { role: 'user', content: message }]);
    const userMessage = message;
    setMessage('');
    
    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          conversation_id: conversationId,
        }),
      });
      
      const data = await response.json();
      
      // Add assistant response to UI
      setMessages(prev => [...prev, { role: 'assistant', content: data.response }]);
      setConversationId(data.conversation_id);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={msg.role}>
            {msg.content}
          </div>
        ))}
      </div>
      <input
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && handleSend()}
        disabled={loading}
      />
      <button onClick={handleSend} disabled={loading}>
        {loading ? 'Sending...' : 'Send'}
      </button>
    </div>
  );
}
```

## Environment Variables for Frontend

Create `.env` in your frontend project:

```env
REACT_APP_API_URL=http://localhost:8000
# Or in production:
REACT_APP_API_URL=https://your-python-api.railway.app
```

## CORS

Your Python backend already has CORS enabled, so your frontend can call it from any domain.

## Summary

**Essential endpoints:**
- ✅ `POST /chat` - Main chat endpoint
- ✅ `GET /health` - Health check
- ✅ `POST /upload` - Upload documents

**Optional endpoints:**
- `GET /documents` - List documents
- `GET /context/summary` - Get context info
- `DELETE /documents/{id}` - Delete documents

All endpoints return JSON and support CORS! 🚀
