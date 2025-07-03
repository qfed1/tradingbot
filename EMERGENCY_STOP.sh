#!/bin/bash
echo "🚨 EMERGENCY STOP INITIATED"
echo "Stopping all trading processes..."

# Kill all bulletproof trading processes
pkill -f "BULLETPROOF" || echo "No BULLETPROOF processes found"
pkill -f "ENHANCED_BULLETPROOF" || echo "No ENHANCED_BULLETPROOF processes found"

# Kill the tmux session
tmux kill-session -t bulletproof_trading 2>/dev/null || echo "No tmux session found"

echo "✅ Emergency stop completed"
echo "All trading activity has been halted"
