# Deploy to Mastra Cloud

## Overview

You have two components to deploy:
1. **Python Backend** (FastAPI) - Your audit logic
2. **Mastra TypeScript Agent** - Calls Python backend, deployed to Mastra Cloud

## Step 1: Deploy Python Backend

Deploy your Python FastAPI service first. Choose a platform:

### Option A: Railway (Easiest)

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize and deploy
cd /Users/bforsyth/Desktop/mastra_ai_assistant
railway init
railway up
```

### Option B: Render

1. Connect your GitHub repo to Render
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables:
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL=gpt-4o-mini`

### Option C: Fly.io

```bash
fly launch
fly deploy
```

After deployment, you'll get a URL like:
- `https://your-app.railway.app`
- `https://your-app.onrender.com`
- `https://your-app.fly.dev`

**Save this URL** - you'll need it for Mastra Cloud!

## Step 2: Deploy to Mastra Cloud

### Option A: Deploy from Local Project

1. **Set environment variable in Mastra Cloud:**
   - Go to your Mastra Cloud project settings
   - Add environment variable: `PYTHON_API_URL`
   - Set value to your deployed Python API URL (e.g., `https://your-app.railway.app`)

2. **Deploy:**
   ```bash
   cd /Users/bforsyth/Desktop/mastra_ai_assistant
   npm run deploy
   ```
   
   Or if you have Mastra Cloud CLI:
   ```bash
   mastra deploy
   ```

### Option B: Connect GitHub Repo to Mastra Cloud

1. In Mastra Cloud dashboard, connect your GitHub repo
2. Set the project root to: `/Users/bforsyth/Desktop/mastra_ai_assistant` (or the repo path)
3. Add environment variables:
   - `PYTHON_API_URL` = your deployed Python API URL
   - `OPENAI_API_KEY` = your OpenAI key
4. Deploy from Mastra Cloud dashboard

## Step 3: Verify Deployment

After deployment, your agent will be available at:
```
https://your-project.mastra.cloud/api/agents/audit-chatbot/generate
```

## Environment Variables for Mastra Cloud

Add these in your Mastra Cloud project settings:

- `PYTHON_API_URL` - Your deployed Python backend URL (required)
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `MASTRA_RESOURCE_ID` - Optional, for runtime context
- `MASTRA_THREAD_ID` - Optional, for conversation threads

## Testing

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

## Troubleshooting

### Agent can't connect to Python backend
- Verify `PYTHON_API_URL` is set correctly in Mastra Cloud
- Check Python backend is running and accessible
- Test Python API directly: `curl https://your-python-api.railway.app/health`

### Deployment fails
- Check TypeScript compilation errors
- Verify all dependencies are in `package.json`
- Check Mastra Cloud logs for errors
