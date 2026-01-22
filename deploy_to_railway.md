# Quick Deploy to Railway (For Mastra Cloud Integration)

## Step 1: Install Railway CLI

```bash
npm i -g @railway/cli
```

## Step 2: Login to Railway

```bash
railway login
```

## Step 3: Initialize Project

```bash
cd /Users/bforsyth/Desktop/mastra_ai_assistant
railway init
```

## Step 4: Create railway.json (Optional)

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

## Step 5: Set Environment Variables

```bash
railway variables set OPENAI_API_KEY=your-key-here
railway variables set OPENAI_MODEL=gpt-4o-mini
```

## Step 6: Deploy

```bash
railway up
```

## Step 7: Get Your URL

After deployment, Railway will give you a URL like:
`https://your-app.railway.app`

Use this URL in your Mastra Cloud agent!
