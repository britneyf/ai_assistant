# 🚀 Deployment Steps - Follow These Now

## Part 1: Deploy Python Backend to Railway (5 minutes)

### Option A: Using Railway Web Interface (Easiest)

1. **Go to Railway**: https://railway.app
2. **Sign up/Login** (use GitHub)
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
   - If repo not connected: Click "Configure GitHub App"
   - Select your repository: `mastra_ai_assistant`
5. **Railway will auto-detect Python** and deploy
6. **Add Environment Variables**:
   - Click on your service → "Variables" tab
   - Add:
     ```
     OPENAI_API_KEY = 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
     OPENAI_MODEL = gpt-4o-mini
     ```
7. **Get your URL**: 
   - Click "Settings" → "Generate Domain"
   - Copy the URL (e.g., `https://mastra-ai-assistant.railway.app`)
   - **SAVE THIS URL** - you'll need it for Mastra Cloud!

### Option B: Using Railway CLI

```bash
# Install Railway CLI (may need sudo)
npm install -g @railway/cli

# Or use npx (no install needed)
npx @railway/cli login
npx @railway/cli init
npx @railway/cli up
```

## Part 2: Push to GitHub (2 minutes)

```bash
cd /Users/bforsyth/Desktop/mastra_ai_assistant

# Create GitHub repo first at: https://github.com/new
# Then run:

git add .
git commit -m "Initial commit - Mastra AI Assistant"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mastra_ai_assistant.git
git push -u origin main
```

**Replace `YOUR_USERNAME` with your GitHub username!**

## Part 3: Deploy to Mastra Cloud (5 minutes)

1. **Go to Mastra Cloud**: https://cloud.mastra.ai
2. **Sign up/Login** (use GitHub)
3. **Click "Create Project"**
4. **Select "Create from GitHub"**
5. **Select repository**: `mastra_ai_assistant`
6. **Configure Project**:
   - **Project Root**: `./` (root of repo)
   - **Branch**: `main`
7. **Add Environment Variables**:
   ```
   PYTHON_API_URL = https://your-railway-url.railway.app
   OPENAI_API_KEY = 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
   ```
   **Replace `your-railway-url` with your actual Railway URL!**
8. **Click "Create Project"**
9. **Enable Deployments**:
   - Go to "Deployments" tab
   - Click "Enable Deployments"
   - Wait for build to complete (~2-3 minutes)

## Part 4: Test Your Deployment

Once both are deployed, test:

```bash
# Test Python backend
curl https://your-railway-url.railway.app/health

# Test Mastra agent
curl -X POST https://your-mastra-project.mastra.cloud/api/agents/audit-chatbot/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "How many audits are there?" }
    ]
  }'
```

## ✅ You're Done!

Your agent is now live at:
- **Python Backend**: `https://your-railway-url.railway.app`
- **Mastra Agent**: `https://your-mastra-project.mastra.cloud/api/agents/audit-chatbot/generate`

## 📝 Quick Reference

**Railway URL**: _____________________________  
**Mastra Cloud URL**: _____________________________  
**GitHub Repo**: https://github.com/YOUR_USERNAME/mastra_ai_assistant
