# Deploy to Mastra Cloud - Step by Step

## Overview

You'll deploy **two components**:
1. **Python Backend** → Deploy to Railway/Render/etc.
2. **Mastra TypeScript Agent** → Deploy to Mastra Cloud

## Step 1: Deploy Python Backend

### Quick Deploy to Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Deploy
cd /Users/bforsyth/Desktop/mastra_ai_assistant
railway init
railway up
```

After deployment, Railway gives you a URL like:
```
https://your-app.railway.app
```

**Save this URL!** You'll need it for Mastra Cloud.

### Set Environment Variables in Railway

In Railway dashboard, add:
- `OPENAI_API_KEY` = your OpenAI key
- `OPENAI_MODEL` = `gpt-4o-mini`

## Step 2: Deploy to Mastra Cloud

### Option A: Connect GitHub Repo (Recommended)

1. **Push to GitHub:**
   ```bash
   cd /Users/bforsyth/Desktop/mastra_ai_assistant
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/your-username/mastra_ai_assistant.git
   git push -u origin main
   ```

2. **In Mastra Cloud:**
   - Go to https://cloud.mastra.ai
   - Click "Create Project"
   - Select "Create from GitHub"
   - Connect your GitHub account
   - Select your `mastra_ai_assistant` repository
   - Click "Import"

3. **Configure Settings:**
   - **Project Root**: `./` (root of repo)
   - **Branch**: `main`
   - **Environment Variables**:
     - `PYTHON_API_URL` = `https://your-app.railway.app`
     - `OPENAI_API_KEY` = your OpenAI key

4. **Enable Deployments:**
   - Go to "Deployment" in sidebar
   - Click "Enable Deployments"
   - Mastra Cloud will build and deploy automatically

### Option B: Deploy from Local

If you have Mastra Cloud CLI access:

```bash
cd /Users/bforsyth/Desktop/mastra_ai_assistant

# Set environment variable
export PYTHON_API_URL=https://your-app.railway.app

# Deploy
npm run deploy
```

## Step 3: Verify Deployment

After deployment, your agent will be available at:
```
https://your-project.mastra.cloud/api/agents/audit-chatbot/generate
```

Test it:
```bash
curl -X POST https://your-project.mastra.cloud/api/agents/audit-chatbot/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "How many audits are there?" }
    ]
  }'
```

## Environment Variables Summary

### For Python Backend (Railway/Render/etc.):
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional, defaults to gpt-4o-mini)

### For Mastra Cloud:
- `PYTHON_API_URL` - Your deployed Python backend URL
- `OPENAI_API_KEY` - Your OpenAI key
- `MASTRA_CLOUD_ACCESS_TOKEN` - Auto-set by Mastra Cloud

## What Gets Deployed

- ✅ Your TypeScript agent (`audit-chatbot-agent.ts`)
- ✅ Your tool (`audit-chat-tool.ts`)
- ✅ Mastra configuration (`src/mastra/index.ts`)
- ✅ All dependencies from `package.json`

The Python backend stays separate and is called via API.

## Next Steps

1. ✅ Python backend deployed
2. ✅ Mastra agent deployed to Mastra Cloud
3. 🎉 Your agent is live and accessible!

Use Mastra Cloud Studio to test, or call the API endpoints directly.
