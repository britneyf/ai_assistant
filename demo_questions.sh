#!/bin/bash
# Demo script showing multiple audit questions

API_URL="http://localhost:8000/chat"

echo "=========================================="
echo "Mastra AI Assistant - Audit Questions Demo"
echo "=========================================="
echo ""

questions=(
    "How many audits are there?"
    "What audits are currently in progress?"
    "What issues are there?"
    "What workpapers are pending?"
    "List all audit managers"
    "What is the status of AML Audit?"
    "How many issues are high priority?"
    "What audits started in 2025?"
)

for i in "${!questions[@]}"; do
    question="${questions[$i]}"
    echo "Question $((i+1)): $question"
    echo "----------------------------------------"
    
    response=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"$question\"}")
    
    answer=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['response'][:300])")
    
    echo "Answer: $answer..."
    echo ""
done

echo "=========================================="
echo "Demo complete! Server is running at $API_URL"
echo "=========================================="
