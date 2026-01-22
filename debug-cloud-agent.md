# Debug: Agent Not Showing in Mastra Cloud Studio

## Settings are correct ✅
- Project Root: `./`
- Mastra Directory: `src/mastra`

## Next Steps to Debug

### 1. Check Deployment Logs

In Mastra Cloud:
1. Go to **Deployments** tab
2. Click on the latest deployment
3. Check for any errors in the build logs

**Look for:**
- TypeScript compilation errors
- Import/module errors
- Missing dependencies
- Build failures

### 2. Test API Endpoint Directly

Test if the agent is actually deployed:

```bash
# Replace with your actual Mastra Cloud URL
curl https://sparse-salmon-rainbow.mastra.cloud/api/agents
```

**Expected response:**
```json
{
  "auditChatbotAgent": {
    "id": "audit-chatbot",
    "name": "AI Assistant",
    ...
  }
}
```

If this returns your agent, it's deployed but Studio might have a UI issue.

### 3. Check Environment Variables

In Mastra Cloud Settings → Environment Variables:

**Required:**
- `OPENAI_API_KEY` ✅
- `PYTHON_API_URL` (if using Python backend) ✅

**Missing env vars can cause silent failures.**

### 4. Verify Agent Export

Your `src/mastra/index.ts` should have:

```typescript
export const mastra = new Mastra({
  agents: { auditChatbotAgent },
});
```

**Check:**
- ✅ `mastra` is exported (not just `const mastra`)
- ✅ `auditChatbotAgent` is imported correctly
- ✅ Agent is in the `agents` object

### 5. Check for TypeScript Errors

Run locally to catch errors:

```bash
cd /Users/bforsyth/Desktop/mastra_ai_assistant
npm run dev
```

If this fails locally, it will fail in Cloud too.

### 6. Studio Cache Issue

Try:
1. Hard refresh Studio: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows)
2. Clear browser cache
3. Try incognito/private window
4. Log out and log back in

### 7. Check Agent Registration Key

Mastra Cloud Studio might be looking for agents by:
- Object key: `auditChatbotAgent`
- Agent ID: `audit-chatbot`
- Agent name: `AI Assistant`

Your current setup:
```typescript
// Registration
agents: { auditChatbotAgent }  // Object key

// Agent definition
id: 'audit-chatbot'            // Agent ID
name: 'AI Assistant'            // Display name
```

### 8. Verify Build Output

Check if Mastra can find your config:

```bash
# Test locally
npm run dev

# Should start without errors
# Should show agent in local Studio at http://localhost:4111
```

## Most Likely Issues

1. **Build failed silently** - Check deployment logs
2. **Missing environment variable** - `OPENAI_API_KEY` not set
3. **TypeScript error** - Check build logs for compilation errors
4. **Import path issue** - Agent file not found during build
5. **Studio UI cache** - Try hard refresh

## Quick Test Commands

```bash
# Test local build
npm run dev

# Test API (replace with your URL)
curl https://sparse-salmon-rainbow.mastra.cloud/api/agents

# Check if agent is accessible
curl -X POST https://sparse-salmon-rainbow.mastra.cloud/api/agents/auditChatbotAgent/generate \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "test"}]}'
```

## What to Check First

1. **Deployment logs** - Are there any errors?
2. **API endpoint** - Does `/api/agents` return your agent?
3. **Environment variables** - Are they all set?

Share what you find in the deployment logs, and we can fix the specific issue!
