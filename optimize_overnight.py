#!/usr/bin/env python3
"""
Overnight Optimization Process
Monitors trading, optimizes strategy parameters, and applies changes automatically
"""
import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("overnight_optimizer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("overnight_optimizer")

class OvernightOptimizer:
    def __init__(self, 
                 log_file="trading_log.txt",
                 report_dir="reports",
                 monitor_interval=300,
                 optimization_interval=1800,
                 restart_on_optimize=True):
        
        self.log_file = log_file
        self.report_dir = report_dir
        self.monitor_interval = monitor_interval  # seconds
        self.optimization_interval = optimization_interval  # seconds
        self.restart_on_optimize = restart_on_optimize
        self.last_optimization = 0
        self.process_id = None
        self.optimization_count = 0
        
        os.makedirs(report_dir, exist_ok=True)
        
    def _get_current_process_id(self):
        """Find the process ID of the running live_trader.py"""
        try:
            result = subprocess.run(
                "ps aux | grep 'python live_trader.py' | grep -v grep",
                shell=True, capture_output=True, text=True
            )
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                return int(lines[0].split()[1])
        except Exception as e:
            logger.error(f"Error finding process ID: {e}")
        return None
    
    def _restart_trading_process(self, api_key, symbols, risk_pct, param_file=None):
        """Stop and restart the trading process with optimized parameters"""
        # First, stop the existing process if running
        current_pid = self._get_current_process_id()
        if current_pid:
            logger.info(f"Stopping current trading process (PID: {current_pid})")
            try:
                os.kill(current_pid, signal.SIGTERM)
                time.sleep(5)  # Give it time to shut down
            except Exception as e:
                logger.error(f"Error stopping process: {e}")
        
        # Archive the current log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if os.path.exists(self.log_file):
            archive_log = f"{self.log_file}.{timestamp}"
            try:
                shutil.copy2(self.log_file, archive_log)
                logger.info(f"Archived trading log to {archive_log}")
            except Exception as e:
                logger.error(f"Error archiving log: {e}")
        
        # Start new process
        cmd = ["python", "live_trader.py", 
               "--api_key", api_key,
               "--risk_pct", str(risk_pct)]
        
        # Add symbols
        for symbol in symbols:
            cmd.extend(["--symbols", symbol])
            
        # Add optimized parameters file if available
        if param_file and os.path.exists(param_file):
            cmd.extend(["--config", param_file])
            
        try:
            with open(self.log_file, 'w') as log_file:
                process = subprocess.Popen(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT
                )
            logger.info(f"Started new trading process with PID: {process.pid}")
            return process.pid
        except Exception as e:
            logger.error(f"Error starting new process: {e}")
            return None
    
    def _run_monitoring(self):
        """Run the monitoring script to collect trading data"""
        logger.info("Running monitoring cycle")
        try:
            subprocess.run([
                "python", "monitor.py",
                "--log_file", self.log_file,
                "--interval", "1"  # Just run once
            ], capture_output=True, text=True)
        except Exception as e:
            logger.error(f"Error running monitor: {e}")
    
    def _run_optimization(self):
        """Run the optimization process"""
        logger.info("Running optimization cycle")
        
        trades_csv = os.path.join(self.report_dir, "trades.csv")
        
        # Skip if we don't have trade data yet
        if not os.path.exists(trades_csv):
            logger.warning("No trade data available for optimization")
            return False
        
        # Check trade count
        try:
            with open(trades_csv, 'r') as f:
                # Count lines excluding header
                line_count = sum(1 for _ in f) - 1
            
            if line_count < 5:
                logger.info(f"Only {line_count} trades - waiting for more data before optimization")
                return False
        except Exception as e:
            logger.error(f"Error checking trade count: {e}")
            return False
        
        # Run optimizer
        output_file = f"optimized_params_{self.optimization_count}.json"
        try:
            result = subprocess.run([
                "python", "strategy_optimizer.py",
                "--trades_file", trades_csv,
                "--output_file", output_file
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Optimization failed: {result.stderr}")
                return False
                
            logger.info(f"Optimization completed: {result.stdout}")
            self.optimization_count += 1
            return output_file
        except Exception as e:
            logger.error(f"Error running optimizer: {e}")
            return False
    
    def run(self, api_key, symbols, risk_pct):
        """Run the overnight optimization process"""
        logger.info("Starting overnight optimization process")
        logger.info(f"Monitoring log: {self.log_file}")
        logger.info(f"Symbols: {symbols}")
        
        try:
            while True:
                # Run monitoring to collect data
                self._run_monitoring()
                
                # Check if it's time to optimize
                current_time = time.time()
                if current_time - self.last_optimization >= self.optimization_interval:
                    logger.info("Running scheduled optimization")
                    param_file = self._run_optimization()
                    self.last_optimization = current_time
                    
                    if param_file and self.restart_on_optimize:
                        # Restart trading with new parameters
                        logger.info(f"Restarting trading with optimized parameters from {param_file}")
                        new_pid = self._restart_trading_process(api_key, symbols, risk_pct, param_file)
                        if new_pid:
                            self.process_id = new_pid
                
                # Wait before next monitoring cycle
                time.sleep(self.monitor_interval)
        
        except KeyboardInterrupt:
            logger.info("Optimization process stopped by user")
        except Exception as e:
            logger.error(f"Error in optimizer: {e}")
        finally:
            # Run one final optimization and generate reports
            final_param_file = self._run_optimization()
            if final_param_file:
                logger.info(f"Final optimized parameters saved to {final_param_file}")

def main():
    parser = argparse.ArgumentParser(description="Overnight Strategy Optimizer")
    parser.add_argument("--api_key", required=True, help="CoinAPI API key")
    parser.add_argument("--symbols", required=True, nargs="+", 
                        help="Symbols to trade (e.g. BITSTAMP_SPOT_BTC_USD)")
    parser.add_argument("--risk_pct", type=float, default=0.01, 
                        help="Risk percentage per trade (default: 0.01)")
    parser.add_argument("--monitor_interval", type=int, default=300,
                        help="Monitoring interval in seconds (default: 300)")
    parser.add_argument("--optimization_interval", type=int, default=1800,
                        help="Optimization interval in seconds (default: 1800)")
    parser.add_argument("--no_restart", action="store_false", dest="restart",
                        help="Don't restart trading process after optimization")
    args = parser.parse_args()
    
    optimizer = OvernightOptimizer(
        monitor_interval=args.monitor_interval,
        optimization_interval=args.optimization_interval,
        restart_on_optimize=args.restart
    )
    optimizer.run(args.api_key, args.symbols, args.risk_pct)

if __name__ == "__main__":
    main() 