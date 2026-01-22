import { createTool } from '@mastra/core/tools';
import { z } from 'zod';

/**
 * Tool for querying the Python FastAPI backend for audit data and document Q&A
 * 
 * This tool calls the Python backend (main.py) which has all the audit logic:
 * - Excel context querying using pandas agents
 * - Document Q&A using RAG
 * - Natural language processing
 */
export const auditChatTool = createTool({
  id: 'audit-chat',
  description: `Query audit data from Excel context files or ask questions about uploaded documents.
  
  Use this tool for:
  - Querying audit information (audits, issues, workpapers)
  - Asking questions about uploaded documents
  - Getting audit insights and summaries
  
  The backend has access to:
  - Excel context files: audits, issues, workpapers
  - Uploaded documents (PDF, DOCX, TXT) for Q&A`,
  inputSchema: z.object({
    query: z.string().describe('The user\'s question about audits, issues, workpapers, or documents'),
    contextType: z.enum(['excel', 'document', 'auto']).optional().describe('Type of context to query (excel for audit data, document for uploaded docs, auto to detect automatically)'),
  }),
  outputSchema: z.object({
    response: z.string().describe('The response from the audit system'),
    contextUsed: z.string().describe('Which context was used (excel, document, both, or none)'),
    sources: z.array(z.string()).nullable().optional().describe('Source document IDs if querying documents'),
  }),
  execute: async ({ query, contextType = 'auto' }) => {
    try {
      // Use environment variable for API URL (works locally and in production)
      const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';
      
      const response = await fetch(`${PYTHON_API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: query,
          context_type: contextType,
          // Optional: Add Mastra runtime context if available
          resource_id: process.env.MASTRA_RESOURCE_ID,
          thread_id: process.env.MASTRA_THREAD_ID,
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.statusText} (${response.status})`);
      }

      const data = await response.json();
      
      return {
        response: data.response || 'No response generated',
        contextUsed: data.context_used || 'unknown',
        sources: data.sources || null,
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      const apiUrl = process.env.PYTHON_API_URL || 'http://localhost:8000';
      return {
        response: `I'm having trouble connecting to the audit system backend at ${apiUrl}. Please ensure:
1. The Python backend is running (python3 -m uvicorn main:app --reload)
2. The PYTHON_API_URL environment variable is set correctly if deployed
3. The backend has the Excel context files loaded in test_files/

Error: ${errorMessage}`,
        contextUsed: 'none',
        sources: null,
      };
    }
  },
});
