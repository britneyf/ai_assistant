# Agent Works Locally But Not in Cloud - Debugging Guide

## ✅ Confirmed: Agent Configuration is Correct
- Agent shows up in local Studio ✅
- Agent is properly registered ✅
- This is a **Cloud deployment issue**, not a code issue

## Common Cloud vs Local Differences

### 1. Import Path Extensions
**Local:** TypeScript/Node might auto-resolve `.js` extensions  
**Cloud:** Build process might require explicit extensions OR might not handle them

**Try:** Remove `.js` extensions from imports:
```typescript
// Instead of:
import { auditChatbotAgent } from './agents/audit-chatbot-agent.js';

// Try:
import { auditChatbotAgent } from './agents/audit-chatbot-agent';
```

### 2. Environment Variables
**Local:** Reads from `.env` file  
**Cloud:** Must be set in Mastra Cloud Settings

**Check:**
- `OPENAI_API_KEY` - Required
- `PYTHON_API_URL` - If using Python backend

### 3. Build Process
**Local:** `npm run dev` uses development mode  
**Cloud:** Uses production build (`mastra build`)

**Check deployment logs for:**
- TypeScript compilation errors
- Module resolution errors
- Missing dependencies

### 4. File Paths/Case Sensitivity
**Local:** macOS is case-insensitive  
**Cloud:** Linux build might be case-sensitive

**Verify:**
- File names match exactly (case-sensitive)
- Directory structure matches

### 5. Dependencies
**Local:** `node_modules` might have different versions  
**Cloud:** Fresh install from `package.json`

**Check:**
- All dependencies in `package.json`
- Version compatibility

## 🔍 Debugging Steps

### Step 1: Check Deployment Logs
1. Go to Mastra Cloud → Deployments
2. Click latest deployment
3. Look for:
   - ❌ Build errors
   - ❌ TypeScript errors
   - ❌ Import/module errors
   - ❌ Missing dependencies

### Step 2: Test API Endpoint
```bash
# Test if agent is accessible via API
curl https://sparse-salmon-rainbow.mastra.cloud/api/agents
```

**If this works:** Agent is deployed, Studio UI issue  
**If this fails:** Build/deployment issue

### Step 3: Compare Local vs Cloud
**Local:**
```bash
npm run dev
# Check: http://localhost:4111/api/agents
```

**Cloud:**
```bash
# Check: https://sparse-salmon-rainbow.mastra.cloud/api/agents
```

### Step 4: Try Removing .js Extensions
The `.js` extensions might work locally but fail in Cloud build.

**Change:**
```typescript
// src/mastra/index.ts
import { auditChatbotAgent } from './agents/audit-chatbot-agent';  // Remove .js

// src/mastra/agents/audit-chatbot-agent.ts  
import { auditChatTool } from '../tools/audit-chat-tool';  // Remove .js
```

## 🎯 Most Likely Causes

1. **Import path with `.js` extension** - Works locally, fails in Cloud build
2. **Build error in deployment logs** - Check for TypeScript/module errors
3. **Missing environment variable** - `OPENAI_API_KEY` not set in Cloud
4. **Build process difference** - Cloud uses production build, local uses dev

## ✅ Quick Fix to Try

Remove `.js` extensions from imports and push:

```typescript
// src/mastra/index.ts
import { auditChatbotAgent } from './agents/audit-chatbot-agent';  // No .js

// src/mastra/agents/audit-chatbot-agent.ts
import { auditChatTool } from '../tools/audit-chat-tool';  // No .js
```

Then push and check deployment logs.
