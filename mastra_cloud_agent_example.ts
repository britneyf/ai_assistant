/**
 * Example Mastra Cloud Agent that calls your Python API
 * 
 * Place this in your Mastra Cloud project:
 * src/mastra/agents/auditAgent.ts
 */

import { Agent } from "@mastra/core/agent";
import { Tool } from "@mastra/core/tool";

// Tool that calls your deployed Python API
const queryAuditAPITool = new Tool({
  name: "queryAuditAPI",
  description: "Query audit data from the Python audit assistant API. Use this to answer questions about audits, issues, and workpapers.",
  execute: async ({ message }: { message: string }) => {
    // Get your Python API URL from environment variable
    const PYTHON_API_URL = process.env.PYTHON_API_URL || "https://your-python-api.railway.app";
    
    try {
      const response = await fetch(`${PYTHON_API_URL}/chat`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          message,
          // Optional: Add Mastra runtime context
          resource_id: "mastra-cloud",
          thread_id: "mastra-session",
        }),
      });
      
      if (!response.ok) {
        throw new Error(`API request failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data.response;
    } catch (error) {
      return `Error calling audit API: ${error instanceof Error ? error.message : String(error)}`;
    }
  },
});

// Export the agent
export const auditAgent = new Agent({
  id: "audit-agent",
  name: "Audit Research Assistant",
  instructions: `You are a helpful audit research assistant that analyzes audit data, issues, and workpapers.

Use the queryAuditAPI tool to find relevant information from the audit database.
Provide accurate, well-supported answers based on the retrieved content.
Focus on the specific content available and acknowledge if you cannot find sufficient information.

When answering:
- Be specific about audit titles, statuses, dates, and managers
- Reference issue ratings and priorities when relevant
- Include workpaper status and dates when available
- Use natural language, not technical jargon`,
  model: "openai/gpt-4o-mini",
  tools: { queryAuditAPI: queryAuditAPITool },
});
