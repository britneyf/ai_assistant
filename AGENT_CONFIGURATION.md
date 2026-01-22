# Agent Configuration Guide

## ✅ Your Agent is Properly Configured

Your `auditChatbotAgent` is correctly set up according to [Mastra's agent documentation](https://mastra.ai/docs/agents/overview):

### Agent Structure

```typescript
// src/mastra/agents/audit-chatbot-agent.ts
export const auditChatbotAgent = new Agent({
  id: 'audit-chatbot',           // ✅ Unique identifier
  name: 'AI Assistant',          // ✅ Display name
  instructions: '...',            // ✅ System instructions
  model: 'openai/gpt-4o-mini',   // ✅ Model configuration
  tools: { auditChatTool },      // ✅ Tools registered
  memory: new Memory(),          // ✅ Memory enabled
});
```

### Registration

```typescript
// src/mastra/index.ts
export const mastra = new Mastra({
  agents: { auditChatbotAgent }, // ✅ Agent registered
  // ... other config
});
```

## 🔍 Troubleshooting "Agents are not configured yet"

If you see this message in Mastra Studio or Cloud, try these steps:

### 1. Verify Dev Server is Running

```bash
npm run dev
```

You should see:
```
✓ Initial bundle complete
◇ Starting Mastra dev server...
```

### 2. Check for Build Errors

Look for any TypeScript or import errors in the terminal output.

### 3. Verify Agent Export

The agent must be:
- ✅ Exported from `src/mastra/agents/audit-chatbot-agent.ts`
- ✅ Imported in `src/mastra/index.ts`
- ✅ Registered in the `Mastra` constructor

### 4. Check Environment Variables

Ensure `.env` has:
```bash
OPENAI_API_KEY=your-key-here
```

### 5. Verify File Structure

```
src/mastra/
  ├── index.ts                    # Mastra config (exports mastra)
  ├── agents/
  │   └── audit-chatbot-agent.ts  # Agent definition
  └── tools/
      └── audit-chat-tool.ts      # Tool definition
```

### 6. Restart Dev Server

Sometimes a restart helps:
```bash
# Stop the server (Ctrl+C)
npm run dev
```

## 🚀 Testing Your Agent

### Via Mastra Studio

1. Open http://localhost:4111
2. Select "AI Assistant" from the agents list
3. Try a query: "How many audits are there?"

### Via API

```bash
curl -X POST http://localhost:4111/api/agents/auditChatbotAgent/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "How many audits are there?" }
    ]
  }'
```

### Via Code

```typescript
import { mastra } from './src/mastra/index.js';

const agent = mastra.getAgent('auditChatbotAgent');
const response = await agent.generate('How many audits are there?');
console.log(response.text);
```

## 📝 Common Issues

### Issue: "Cannot find module"
**Solution**: Ensure all imports use correct paths and file extensions match your TypeScript config.

### Issue: "Agent not found"
**Solution**: Verify the agent ID matches exactly:
- Agent definition: `id: 'audit-chatbot'`
- Registration: `agents: { auditChatbotAgent }`
- Access: `mastra.getAgent('auditChatbotAgent')` (uses the object key, not the id)

### Issue: "Tool not found"
**Solution**: Ensure the tool is:
- Properly exported from the tools file
- Imported in the agent file
- Passed to the agent: `tools: { auditChatTool }`

## ✅ Verification Checklist

- [ ] Agent created with `new Agent({ ... })`
- [ ] Agent exported from agent file
- [ ] Agent imported in `index.ts`
- [ ] Agent registered in `Mastra` constructor
- [ ] Dev server starts without errors
- [ ] Agent appears in Mastra Studio
- [ ] Python backend is running (for tool to work)
- [ ] Environment variables are set

## 📚 Reference

- [Mastra Agents Documentation](https://mastra.ai/docs/agents/overview)
- [Using Tools](https://mastra.ai/docs/agents/using-tools)
- [Agent Memory](https://mastra.ai/docs/agents/memory)
