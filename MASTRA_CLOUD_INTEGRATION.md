# Integrating with Mastra Cloud

## Current Situation

- **Your Python Agent**: Working locally at `http://localhost:8000`
- **Mastra Cloud**: Expects TypeScript/JavaScript projects
- **Your Mastra Cloud Project**: `ai_assistant` (3 agents, 1 workflow)

## Option 1: Deploy Python as API Service (Recommended)

Deploy your Python FastAPI service to a cloud platform, then call it from Mastra Cloud.

### Step 1: Deploy Python Service

Deploy to any cloud platform:
- **Railway**: `railway up`
- **Render**: Connect GitHub repo
- **Fly.io**: `fly deploy`
- **Heroku**: `git push heroku main`
- **AWS/GCP/Azure**: Deploy container

### Step 2: Create Mastra Agent that Calls Your API

In Mastra Cloud, create a TypeScript agent that calls your Python API:

```typescript
// src/mastra/agents/auditAgent.ts
import { Agent } from "@mastra/core/agent";
import { Tool } from "@mastra/core/tool";

// Tool to call your Python API
const auditQueryTool = new Tool({
  name: "queryAuditAPI",
  description: "Query audit data from Python API service",
  execute: async ({ query }: { query: string }) => {
    const response = await fetch("https://your-python-api.railway.app/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: query }),
    });
    const data = await response.json();
    return data.response;
  },
});

export const auditAgent = new Agent({
  id: "audit-agent",
  name: "Audit Assistant",
  instructions: `You are an audit assistant that queries audit data.
Use the queryAuditAPI tool to get information about audits, issues, and workpapers.`,
  model: "openai/gpt-4o-mini",
  tools: { auditQueryTool },
});
```

### Step 3: Register in Mastra

```typescript
// src/mastra/index.ts
import { Mastra } from "@mastra/core";
import { auditAgent } from "./agents/auditAgent";

export const mastra = new Mastra({
  agents: { auditAgent },
});
```

## Option 2: Port to TypeScript (Native Mastra Cloud)

Convert your Python agent to TypeScript for native Mastra Cloud deployment.

### Step 1: Create TypeScript Agent

```typescript
// src/mastra/agents/auditAgent.ts
import { Agent } from "@mastra/core/agent";
import { createPandasDataframeTool } from "@mastra/tools"; // If available
// Or create custom tool for Excel data

export const auditAgent = new Agent({
  id: "audit-agent",
  name: "Audit Research Assistant",
  instructions: `You are a helpful audit research assistant that analyzes audit data.
Use the provided tools to find relevant information about audits, issues, and workpapers.`,
  model: "openai/gpt-4o-mini",
  tools: {
    // Add tools for querying Excel data
    queryAuditData: auditDataTool,
  },
});
```

### Step 2: Deploy to Mastra Cloud

```bash
# In your Mastra project
mastra deploy
```

## Option 3: Hybrid Approach (Best of Both Worlds)

Keep Python for data processing, use Mastra for orchestration.

1. **Python Service**: Handles Excel data, complex processing
2. **Mastra Agent**: Orchestrates, provides UI, manages workflows
3. **Communication**: HTTP API between them

## Quick Start: Option 1 (API Bridge)

### 1. Deploy Your Python Service

```bash
# Example with Railway
railway login
railway init
railway up
```

### 2. Get Your API URL

After deployment, you'll get a URL like:
- `https://your-app.railway.app`
- `https://your-app.onrender.com`

### 3. Create Mastra Agent

In your Mastra Cloud project, create an agent that calls your Python API:

```typescript
import { Agent } from "@mastra/core/agent";
import { Tool } from "@mastra/core/tool";

const pythonAPITool = new Tool({
  name: "queryPythonAuditAPI",
  description: "Query the Python audit assistant API",
  execute: async ({ message }: { message: string }) => {
    const PYTHON_API_URL = process.env.PYTHON_API_URL || "https://your-python-api.railway.app";
    
    const response = await fetch(`${PYTHON_API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    
    const data = await response.json();
    return data.response;
  },
});

export const auditAgent = new Agent({
  id: "audit-agent",
  name: "Audit Assistant",
  instructions: `You are an audit assistant. Use the queryPythonAuditAPI tool to answer questions about audits, issues, and workpapers.`,
  model: "openai/gpt-4o-mini",
  tools: { pythonAPITool },
});
```

### 4. Add Environment Variable

In Mastra Cloud project settings:
- Add `PYTHON_API_URL` environment variable
- Set it to your deployed Python API URL

### 5. Deploy to Mastra Cloud

```bash
mastra deploy
```

## Recommended Approach

**Option 1 (API Bridge)** is recommended because:
- ✅ Keep your Python code (no rewrite needed)
- ✅ Leverage Mastra Cloud UI and workflows
- ✅ Easy to maintain and update
- ✅ Best of both worlds

## Next Steps

1. **Deploy Python service** to Railway/Render/Fly.io
2. **Create Mastra agent** that calls your Python API
3. **Deploy to Mastra Cloud** using `mastra deploy`
4. **Test** in Mastra Cloud Studio

Your Python agent will be accessible through Mastra Cloud! 🚀
