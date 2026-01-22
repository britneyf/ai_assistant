# Deploying to Mastra Cloud

## Overview

Your Python audit assistant can be integrated with Mastra Cloud by deploying it as an API service and creating a Mastra agent that calls it.

## Quick Steps

### 1. Deploy Python Service

Choose a platform:
- **Railway** (easiest): `railway up`
- **Render**: Connect GitHub repo
- **Fly.io**: `fly deploy`
- **Heroku**: `git push heroku main`

### 2. Get Your API URL

After deployment, you'll get a URL like:
- `https://your-app.railway.app`
- `https://your-app.onrender.com`

### 3. Create Mastra Agent

In your Mastra Cloud project (`ai_assistant`), create:

**File: `src/mastra/agents/auditAgent.ts`**
```typescript
import { Agent } from "@mastra/core/agent";
import { Tool } from "@mastra/core/tool";

const queryAuditAPITool = new Tool({
  name: "queryAuditAPI",
  description: "Query audit data from Python API",
  execute: async ({ message }: { message: string }) => {
    const PYTHON_API_URL = process.env.PYTHON_API_URL;
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
  instructions: "You are an audit assistant. Use queryAuditAPI to answer questions.",
  model: "openai/gpt-4o-mini",
  tools: { queryAuditAPI: queryAuditAPITool },
});
```

### 4. Register Agent

**File: `src/mastra/index.ts`**
```typescript
import { Mastra } from "@mastra/core";
import { auditAgent } from "./agents/auditAgent";

export const mastra = new Mastra({
  agents: { auditAgent },
});
```

### 5. Set Environment Variable

In Mastra Cloud project settings:
- Add `PYTHON_API_URL` = `https://your-python-api.railway.app`

### 6. Deploy

```bash
mastra deploy
```

## Result

Your Python audit assistant will be accessible through Mastra Cloud! You can:
- Use it in Mastra Studio
- Call it from workflows
- Access it via Mastra Cloud API
- Use it in your Mastra Cloud project

## Files Included

- `mastra_cloud_agent_example.ts` - Example Mastra agent code
- `deploy_to_railway.md` - Railway deployment guide
- `MASTRA_CLOUD_INTEGRATION.md` - Detailed integration guide
