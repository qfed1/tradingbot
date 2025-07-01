#!/bin/bash
echo "🛑 EMERGENCY STOP - BULLETPROOF LIVE TRADER"
echo "==========================================="
echo ""

# Stop the trading process
echo "🔪 Stopping trading process..."
pkill -f BULLETPROOF_LIVE_TRADER.py
sleep 2

# Kill tmux session
echo "🔪 Stopping tmux session..."
tmux kill-session -t live_trading 2>/dev/null || true

echo "✅ All trading stopped successfully"
echo "📊 Final logs:"
tail -10 bulletproof_live.log 2>/dev/null || echo "No logs found"
echo ""
echo "To restart: ./LAUNCH_LIVE_TONIGHT.sh"
