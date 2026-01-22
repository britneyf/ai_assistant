/**
 * Quick verification script to ensure the agent is properly configured
 * Run with: npx tsx verify-agent.ts
 */

import { mastra } from './src/mastra/index.js';

console.log('🔍 Verifying Mastra Agent Configuration...\n');

try {
  // Check if mastra instance exists
  if (!mastra) {
    throw new Error('Mastra instance not found');
  }

  // Get the agent
  const agent = mastra.getAgent('auditChatbotAgent');
  
  if (!agent) {
    throw new Error('Agent "auditChatbotAgent" not found');
  }

  console.log('✅ Agent Configuration:');
  console.log(`   ID: ${agent.id}`);
  console.log(`   Name: ${agent.name}`);
  console.log(`   Model: ${typeof agent.model === 'string' ? agent.model : 'dynamic'}`);
  console.log(`   Tools: ${Object.keys(agent.tools || {}).length} tool(s)`);
  console.log(`   Memory: ${agent.memory ? 'enabled' : 'disabled'}`);
  
  console.log('\n✅ Agent is properly configured!');
  console.log('\n📋 Next Steps:');
  console.log('   1. Make sure Python backend is running: python3 -m uvicorn main:app --reload');
  console.log('   2. Start Mastra dev server: npm run dev');
  console.log('   3. Open Mastra Studio: http://localhost:4111');
  
} catch (error) {
  console.error('❌ Error:', error instanceof Error ? error.message : String(error));
  process.exit(1);
}
