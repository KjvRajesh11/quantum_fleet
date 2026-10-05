#!/usr/bin/env bash
# Quantum Fleet Shutdown Script

echo "⚓ Stopping Quantum Fleet services on ports 8000 & 3000..."

# Kill process on port 8000
lsof -ti :8000 | xargs kill -9 2>/dev/null && echo "✓ Stopped FastAPI Backend (port 8000)" || echo "Port 8000 already stopped"

# Kill process on port 3000
lsof -ti :3000 | xargs kill -9 2>/dev/null && echo "✓ Stopped Next.js Frontend (port 3000)" || echo "Port 3000 already stopped"

echo "All Quantum Fleet services shut down."
