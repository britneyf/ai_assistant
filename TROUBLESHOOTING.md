# Troubleshooting: Mastra Dev Server

## Why is `npm run dev` taking so long?

The first time you run `npx mastra dev`, it:
1. Downloads the Mastra CLI (can take 1-2 minutes)
2. Compiles TypeScript files
3. Starts the dev server
4. Opens Studio

**This is normal for the first run!** Subsequent runs will be faster.

## Quick Fixes

### Option 1: Wait it out (Recommended)
Just let it run - it should complete in 1-3 minutes. You'll see:
```
Studio available at http://localhost:4111
```

### Option 2: Check if it's actually running
Open a new terminal and check:
```bash
curl http://localhost:4111
```

If it responds, Studio is ready!

### Option 3: Run directly without npm script
```bash
npx mastra@latest dev
```

### Option 4: Check for errors
Look at the terminal output - if there are errors, they'll show what's wrong.

## Common Issues

### "mastra: command not found"
✅ Fixed - we're using `npx mastra` now

### "Cannot find module"
Run: `npm install`

### Port 4111 already in use
Kill the process:
```bash
lsof -ti:4111 | xargs kill
```

### TypeScript errors
Check `src/mastra/index.ts` and agent files for syntax errors

## Alternative: Test Python Backend Directly

While waiting for Mastra, you can test your Python backend:

```bash
# In another terminal
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many audits are there?"}'
```

This confirms your Python backend is working!

## Expected Timeline

- **First run**: 2-5 minutes (downloading, compiling)
- **Subsequent runs**: 10-30 seconds (just starting server)

If it's been more than 5 minutes, there might be an issue. Check the terminal for error messages.
