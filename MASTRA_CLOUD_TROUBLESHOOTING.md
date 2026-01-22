# Mastra Cloud - Agent Not Showing in Studio

## Issue: Agent not appearing in Mastra Cloud Studio

If your agent doesn't show up at:
`https://cloud.mastra.ai/britneys-team/dashboard/projects/sparse-salmon-rainbow/studio/agents`

## Common Causes & Solutions

### 1. Build/Deployment Issues

**Check deployment logs:**
1. Go to your Mastra Cloud project dashboard
2. Click on "Deployments" tab
3. Check the latest deployment logs for errors

**Common build errors:**
- TypeScript compilation errors
- Missing dependencies
- Import path issues

**Solution:**
```bash
# Test build locally first
npm run build  # or npx mastra build

# Check for TypeScript errors
npx tsc --noEmit
```

### 2. Agent Registration Key Mismatch

**Your current setup:**
```typescript
// src/mastra/index.ts
export const mastra = new Mastra({
  agents: { auditChatbotAgent }, // Object key: 'auditChatbotAgent'
});
```

**Agent definition:**
```typescript
// src/mastra/agents/audit-chatbot-agent.ts
export const auditChatbotAgent = new Agent({
  id: 'audit-chatbot',  // Agent ID
  name: 'AI Assistant', // Display name
});
```

**Important:** Mastra Cloud Studio may display agents by:
- The **object key** in the `agents` object (`auditChatbotAgent`)
- The agent's **`id`** property (`audit-chatbot`)
- The agent's **`name`** property (`AI Assistant`)

### 3. Export/Import Issues

**Verify the agent is properly exported:**
```typescript
// src/mastra/agents/audit-chatbot-agent.ts
export const auditChatbotAgent = new Agent({ ... }); // ✅ Must be exported
```

**Verify the agent is properly imported:**
```typescript
// src/mastra/index.ts
import { auditChatbotAgent } from './agents/audit-chatbot-agent'; // ✅ Must match export name
```

### 4. Environment Variables Missing

**Check Mastra Cloud environment variables:**
1. Go to your project settings
2. Check "Environment Variables"
3. Ensure these are set:
   - `OPENAI_API_KEY` ✅
   - `PYTHON_API_URL` (if using Python backend) ✅

### 5. Mastra Instance Not Properly Exported

**Verify `mastra` is exported:**
```typescript
// src/mastra/index.ts
export const mastra = new Mastra({ ... }); // ✅ Must be exported
```

### 6. File Structure Issues

**Required structure:**
```
src/mastra/
  ├── index.ts                    # ✅ Must export mastra
  ├── agents/
  │   └── audit-chatbot-agent.ts  # ✅ Must export agent
  └── tools/
      └── audit-chat-tool.ts      # ✅ Must export tool
```

### 7. Deployment Not Complete

**Check deployment status:**
1. Go to Mastra Cloud dashboard
2. Check if deployment is "Active" (green)
3. If it's still building, wait for it to complete
4. If it failed, check the error logs

### 8. Studio Cache Issues

**Try:**
1. Hard refresh the Studio page (Cmd+Shift+R or Ctrl+Shift+R)
2. Clear browser cache
3. Log out and log back in

## Verification Steps

### Step 1: Test Locally

```bash
# Start dev server
npm run dev

# Open Studio
# http://localhost:4111

# Check if agent appears in Studio
```

### Step 2: Check API Endpoint

```bash
# Test if agent is accessible via API
curl http://localhost:4111/api/agents

# Should return:
# {
#   "auditChatbotAgent": {
#     "id": "audit-chatbot",
#     "name": "AI Assistant",
#     ...
#   }
# }
```

### Step 3: Test Agent Generation

```bash
curl -X POST http://localhost:4111/api/agents/auditChatbotAgent/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "How many audits are there?" }
    ]
  }'
```

### Step 4: Check Mastra Cloud API

```bash
# Replace with your actual Mastra Cloud URL
curl https://your-project.mastra.cloud/api/agents

# Should return your agents
```

## Quick Fix: Ensure Proper Registration

Make sure your `src/mastra/index.ts` looks exactly like this:

```typescript
import { Mastra } from '@mastra/core/mastra';
import { auditChatbotAgent } from './agents/audit-chatbot-agent';

export const mastra = new Mastra({
  agents: { auditChatbotAgent }, // ✅ Agent registered here
});
```

## Still Not Working?

1. **Check Mastra Cloud deployment logs** for specific errors
2. **Verify the project root** is set correctly in Mastra Cloud (should be `./` or `src/mastra`)
3. **Check if TypeScript is compiling** without errors
4. **Verify all dependencies** are installed (`npm install`)
5. **Check Mastra Cloud status page** for service issues

## Expected Behavior

Once properly deployed, you should see:
- Agent listed in Studio at: `/studio/agents`
- Agent accessible via API: `/api/agents/auditChatbotAgent`
- Agent name displayed as: "AI Assistant"
