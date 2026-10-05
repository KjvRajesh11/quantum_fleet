#!/usr/bin/env bash
# Quantum Fleet Local Startup Script
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "⚓ Starting Quantum Fleet Decision Support System"
echo "=========================================================="

# 1. Kill any existing instances on ports 8000 and 3000
echo "[1/3] Checking ports 8000 & 3000..."
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :3000 | xargs kill -9 2>/dev/null || true

# 2. Launch FastAPI Backend on port 8000
echo "[2/3] Launching FastAPI Backend on http://localhost:8000..."
PYTHONPATH=backend "$PROJECT_DIR/.venv/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "      Backend started (PID: $BACKEND_PID, logs: backend.log)"

# Wait for backend health
echo "      Waiting for backend initialization..."
for i in {1..20}; do
  if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "      ✓ Backend is ONLINE (http://localhost:8000)"
    break
  fi
  sleep 0.5
done

# 3. Launch Next.js Frontend on port 3000
echo "[3/3] Launching Next.js Frontend on http://localhost:3000..."
cd "$PROJECT_DIR/frontend"
npm run start -- -p 3000 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "      Frontend started (PID: $FRONTEND_PID, logs: frontend.log)"

# Wait for frontend health
for i in {1..20}; do
  if curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo "      ✓ Frontend is ONLINE (http://localhost:3000)"
    break
  fi
  sleep 0.5
done

echo ""
echo "=========================================================="
echo "🚀 Quantum Fleet is fully operational:"
echo "   - Dashboard UI:     http://localhost:3000"
echo "   - QPSO Planner:     http://localhost:3000/optimize"
echo "   - IMO CII & WtW:    http://localhost:3000/cii-analytics"
echo "   - Solver Benchmark: http://localhost:3000/benchmark"
echo "   - FastAPI Swagger:  http://localhost:8000/docs"
echo "=========================================================="
echo "To stop both services at any time, run: ./stop.sh"
