#!/bin/bash

# Script to check Mastra Cloud deployment status

echo "🔍 Checking Mastra Cloud Deployment..."
echo "======================================"
echo ""

# Check if we can access the agents API
PROJECT_URL="https://sparse-salmon-rainbow.mastra.cloud"

echo "📡 Testing API endpoint..."
echo "URL: ${PROJECT_URL}/api/agents"
echo ""

response=$(curl -s -o /dev/null -w "%{http_code}" "${PROJECT_URL}/api/agents" 2>/dev/null)

if [ "$response" = "200" ]; then
    echo "✅ API is accessible (HTTP $response)"
    echo ""
    echo "📋 Available agents:"
    curl -s "${PROJECT_URL}/api/agents" | jq '.' 2>/dev/null || curl -s "${PROJECT_URL}/api/agents"
else
    echo "❌ API not accessible (HTTP $response)"
    echo ""
    echo "Possible issues:"
    echo "1. Deployment not complete - check Mastra Cloud dashboard"
    echo "2. Deployment failed - check build logs"
    echo "3. Wrong project URL - verify in Mastra Cloud settings"
fi

echo ""
echo "🔗 Mastra Cloud Dashboard:"
echo "https://cloud.mastra.ai/britneys-team/dashboard/projects/sparse-salmon-rainbow"
echo ""
echo "Next steps:"
echo "1. Check Deployment tab for build status"
echo "2. Check Settings for environment variables"
echo "3. Verify project root is set to './' or 'src/mastra'"
