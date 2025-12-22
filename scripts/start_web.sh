#!/bin/bash
# Start the Research Paper Companion AI Web Application

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "=============================================="
echo "  Research Paper Companion AI"
echo "  Starting Web Application..."
echo "=============================================="
echo -e "${NC}"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/../venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Please run setup first.${NC}"
    exit 1
fi

# Activate virtual environment
source "$PROJECT_ROOT/../venv/bin/activate"

echo -e "${GREEN}Starting FastAPI Backend on http://localhost:8000${NC}"
echo ""

# Start backend in background
cd "$PROJECT_ROOT/api"
python server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

echo -e "${GREEN}Starting Next.js Frontend on http://localhost:3000${NC}"
echo ""

# Start frontend
cd "$PROJECT_ROOT/web"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

npm run dev &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}=============================================="
echo "  Application Started Successfully!"
echo "=============================================="
echo -e "${NC}"
echo ""
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Handle shutdown
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}Done!${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
