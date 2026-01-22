# Deploy Now - Quick Guide

## Step 1: Deploy Python Backend to Railway

### Install Railway CLI (if not installed)
```bash
npm install -g @railway/cli
```

### Login and Deploy
```bash
cd /Users/bforsyth/Desktop/mastra_ai_assistant
railway login
railway init
railway up
```

Railway will:
- Detect Python project
- Install dependencies from `requirements.txt`
- Start server with `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Give you a URL like `https://your-app.railway.app`

### Set Environment Variables in Railway

In Railway dashboard:
1. Go to your project
2. Click "Variables"
3. Add:
   - `OPENAI_API_KEY` = your OpenAI key
   - `OPENAI_MODEL` = `gpt-4o-mini`

**Save the Railway URL** - you'll need it for Mastra Cloud!

## Step 2: Push to GitHub

```bash
cd /Users/bforsyth/Desktop/mastra_ai_assistant
git add .
git commit -m "Initial commit - Mastra AI Assistant"
git branch -M main

# Create repo on GitHub first, then:
git remote add origin https://github.com/your-username/mastra_ai_assistant.git
git push -u origin main
```

## Step 3: Deploy to Mastra Cloud

1. Go to https://cloud.mastra.ai
2. Click "Create Project"
3. Select "Create from GitHub"
4. Connect GitHub account
5. Select `mastra_ai_assistant` repository
6. Configure:
   - **Project Root**: `./`
   - **Branch**: `main`
   - **Environment Variables**:
     - `PYTHON_API_URL` = `https://your-app.railway.app` (from Railway)
     - `OPENAI_API_KEY` = your OpenAI key
7. Click "Create Project"
8. Go to "Deployment" → "Enable Deployments"

## Step 4: Test

Once deployed, test your agent:

```bash
curl -X POST https://your-project.mastra.cloud/api/agents/audit-chatbot/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "How many audits are there?" }
    ]
  }'
```

## What Gets Deployed

**Railway (Python Backend):**
- FastAPI server
- All Python dependencies
- Excel context files (from test_files/)
- Runs at: `https://your-app.railway.app`

**Mastra Cloud (TypeScript Agent):**
- Your agent (`audit-chatbot-agent.ts`)
- Your tool (`audit-chat-tool.ts`)
- Mastra configuration
- Runs at: `https://your-project.mastra.cloud`

The agent calls the Python backend via the `PYTHON_API_URL` environment variable.

## Quick Commands

```bash
# Deploy Python backend
railway up

# Check Railway status
railway status

# View Railway logs
railway logs

# Deploy Mastra agent (after pushing to GitHub)
# Done via Mastra Cloud dashboard
```

## Troubleshooting

### Railway deployment fails
- Check `requirements.txt` has all dependencies
- Verify `railway.json` is correct
- Check Railway logs: `railway logs`

### Mastra Cloud can't connect to Python backend
- Verify `PYTHON_API_URL` is set correctly
- Test Python API: `curl https://your-app.railway.app/health`
- Check Railway is running

### Agent not found in Mastra Cloud
- Verify agent is exported in `src/mastra/index.ts`
- Check build logs in Mastra Cloud dashboard
- Ensure TypeScript compiles without errors
