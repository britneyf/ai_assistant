# Mastra Cloud Settings - Correct Configuration

## ✅ Required Settings for Your Project

Based on your project structure, here are the **exact settings** you need in Mastra Cloud:

### Project Settings

1. **Project Root**: `./`
   - This should point to the root of your repository
   - Your `src/mastra/index.ts` is at: `./src/mastra/index.ts` (relative to root)

2. **Mastra Directory**: `src/mastra`
   - This tells Mastra where to find your `index.ts` file
   - Mastra will look for `src/mastra/index.ts`

3. **Branch**: `main`
   - ✅ Already correct

4. **Install Command**: (usually auto-detected)
   - Should be: `npm install` or `pnpm install`

5. **Project Port**: (usually auto-detected)
   - Default: `4111`

## 🔍 How to Verify

### Step 1: Check Your Project Structure

Your project should look like this:
```
mastra_ai_assistant/
├── src/
│   └── mastra/
│       ├── index.ts          ← Mastra config (exports mastra)
│       ├── agents/
│       │   └── audit-chatbot-agent.ts
│       └── tools/
│           └── audit-chat-tool.ts
├── package.json
├── .env
└── ...
```

### Step 2: Verify Settings in Mastra Cloud

1. Go to: https://cloud.mastra.ai/britneys-team/dashboard/projects/sparse-salmon-rainbow/settings
2. Check these values:

**Project Root:**
- ✅ Should be: `./`
- ❌ NOT: `src/` or `src/mastra/`

**Mastra Directory:**
- ✅ Should be: `src/mastra`
- ❌ NOT: `./` or empty

### Step 3: Test Locally First

Before deploying, test that Mastra can find your config:

```bash
cd /Users/bforsyth/Desktop/mastra_ai_assistant
npm run dev
```

If this works locally, the same structure should work in Mastra Cloud.

## 🐛 Common Issues

### Issue 1: Project Root is `src/` instead of `./`

**Symptom:** Mastra can't find `package.json` or `node_modules`

**Fix:** Change Project Root to `./`

### Issue 2: Mastra Directory is wrong

**Symptom:** "Cannot find module" or "Mastra instance not found"

**Fix:** Set Mastra Directory to `src/mastra`

### Issue 3: Agent not showing in Studio

**Possible causes:**
1. Build failed - check deployment logs
2. TypeScript errors - check build output
3. Agent not exported - verify `export const mastra` in `index.ts`

## ✅ Quick Checklist

- [ ] Project Root = `./`
- [ ] Mastra Directory = `src/mastra`
- [ ] Branch = `main`
- [ ] `src/mastra/index.ts` exists and exports `mastra`
- [ ] `src/mastra/index.ts` imports and registers `auditChatbotAgent`
- [ ] Environment variables set: `OPENAI_API_KEY`, `PYTHON_API_URL`
- [ ] Deployment status is "Active" (green)

## 🔄 After Changing Settings

1. **Save** the settings
2. **Trigger a new deployment** (or wait for auto-deploy if enabled)
3. **Check deployment logs** to verify it builds successfully
4. **Wait for deployment to complete** (usually 2-3 minutes)
5. **Refresh Studio** and check if agent appears

## 📝 Your Current Configuration

Based on your project:

```typescript
// src/mastra/index.ts
export const mastra = new Mastra({
  agents: { auditChatbotAgent }, // ✅ Agent registered
  // ... other config
});
```

This should work with:
- **Project Root**: `./`
- **Mastra Directory**: `src/mastra`

## 🚀 Next Steps

1. **Verify settings** in Mastra Cloud dashboard
2. **Update if needed** (Project Root = `./`, Mastra Directory = `src/mastra`)
3. **Trigger new deployment** if you changed settings
4. **Check deployment logs** for any errors
5. **Test in Studio** once deployment completes
