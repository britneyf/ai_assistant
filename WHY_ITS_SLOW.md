# Why `npm run dev` is Taking Long

## This is Normal! ⏱️

The first time you run `npx mastra dev`, it needs to:

1. **Download Mastra CLI** (~30-60 seconds)
   - Downloads from npm registry
   - First time only

2. **Compile TypeScript** (~30-60 seconds)
   - Compiles `src/mastra/**/*.ts` files
   - Checks for errors

3. **Start Dev Server** (~10-30 seconds)
   - Initializes Mastra framework
   - Loads agents and tools
   - Starts HTTP server

4. **Open Studio** (instant)
   - Opens at `http://localhost:4111`

**Total time: 2-5 minutes on first run** ✅

## What You Should See

In your terminal, you'll eventually see:
```
✓ Compiled successfully
Studio available at http://localhost:4111
```

## While You Wait

You can test your Python backend is working:

```bash
# In another terminal
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many audits are there?"}'
```

If this works, your backend is ready! ✅

## After First Run

Subsequent runs will be **much faster** (10-30 seconds) because:
- CLI is cached
- TypeScript only recompiles changed files
- Server starts faster

## If It's Been More Than 5 Minutes

1. **Check terminal output** - Look for error messages
2. **Check if port 4111 is in use:**
   ```bash
   lsof -i :4111
   ```
3. **Try stopping and restarting:**
   - Press `Ctrl+C` to stop
   - Run `npm run dev` again

## Quick Test

Once it's running, open:
```
http://localhost:4111
```

You should see Mastra Studio! 🎉
