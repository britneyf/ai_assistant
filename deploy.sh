#!/bin/bash

# Deployment Helper Script
# This script helps you deploy to Railway and Mastra Cloud

echo "🚀 Mastra AI Assistant - Deployment Helper"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git not initialized. Run: git init"
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
echo "📦 Current branch: $CURRENT_BRANCH"
echo ""

# Check if remote exists
if git remote get-url origin >/dev/null 2>&1; then
    REMOTE_URL=$(git remote get-url origin)
    echo "✅ GitHub remote: $REMOTE_URL"
    echo ""
    echo "Pushing to GitHub..."
    git push -u origin $CURRENT_BRANCH
else
    echo "⚠️  No GitHub remote found."
    echo ""
    echo "📝 To connect to GitHub:"
    echo "1. Create a new repository at: https://github.com/new"
    echo "2. Then run:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/mastra_ai_assistant.git"
    echo "   git push -u origin $CURRENT_BRANCH"
    echo ""
    exit 1
fi

echo ""
echo "✅ Code pushed to GitHub!"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. Deploy Python Backend to Railway:"
echo "   → Go to: https://railway.app"
echo "   → New Project → Deploy from GitHub"
echo "   → Select: mastra_ai_assistant"
echo "   → Add env vars: OPENAI_API_KEY, OPENAI_MODEL"
echo "   → Copy your Railway URL"
echo ""
echo "2. Deploy to Mastra Cloud:"
echo "   → Go to: https://cloud.mastra.ai"
echo "   → Create Project → From GitHub"
echo "   → Select: mastra_ai_assistant"
echo "   → Add env var: PYTHON_API_URL = (your Railway URL)"
echo ""
echo "📖 See DEPLOY_STEPS.md for detailed instructions"
echo ""
