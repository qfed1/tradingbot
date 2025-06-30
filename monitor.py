#!/usr/bin/env python3
"""
Trading Strategy Monitor
Analyzes trading signals and performance from log files
"""
import argparse
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("monitor_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("monitor")

class TradingMonitor:
    def __init__(self, log_file):
        self.log_file = log_file
        self.last_position = 0
        self.trades = []
        self.signals = defaultdict(list)
        self.symbols = set()
        self.candle_counts = defaultdict(int)
        self.performance_metrics = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0.0,
            "max_drawdown": 0.0,
            "by_symbol": defaultdict(lambda: {"trades": 0, "wins": 0, "profit": 0.0})
        }
        
    def parse_log(self):
        """Parse the trading log file for signals and trades"""
        if not os.path.exists(self.log_file):
            logger.error(f"Log file not found: {self.log_file}")
            return False
            
        with open(self.log_file, 'r') as f:
            f.seek(self.last_position)
            new_content = f.read()
            self.last_position = f.tell()
            
        # Parse trades
        trade_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) INFO: (.*?) \| (BUY|SELL) (\w+) @ ([\d\.]+)(.*)"
        for match in re.finditer(trade_pattern, new_content):
            timestamp, _, side, symbol, price, details = match.groups()
            self.symbols.add(symbol)
            
            # Extract PNL if available
            pnl = None
            pnl_match = re.search(r"P&L: ([-\d\.]+)%", details)
            if pnl_match:
                pnl = float(pnl_match.group(1)) / 100
                
            # Extract reason
            reason_match = re.search(r"Reason: ([^|]+)", details)
            reason = reason_match.group(1).strip() if reason_match else "Unknown"
                
            trade = {
                "timestamp": datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f"),
                "symbol": symbol,
                "side": side,
                "price": float(price),
                "reason": reason,
                "pnl": pnl
            }
            
            self.trades.append(trade)
            
            # Update performance metrics if this is an exit
            if pnl is not None:
                self.performance_metrics["total_trades"] += 1
                self.performance_metrics["by_symbol"][symbol]["trades"] += 1
                
                if pnl > 0:
                    self.performance_metrics["wins"] += 1
                    self.performance_metrics["by_symbol"][symbol]["wins"] += 1
                    
                self.performance_metrics["profit"] += pnl
                self.performance_metrics["by_symbol"][symbol]["profit"] += pnl
        
        # Parse signal data
        signal_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) INFO: (.+)"
        for match in re.finditer(signal_pattern, new_content):
            timestamp, message = match.groups()
            
            # Track candle counts
            if "Not enough data for" in message:
                symbol = re.search(r"Not enough data for ([^,]+)", message).group(1)
                self.symbols.add(symbol)
                self.candle_counts[symbol] += 1
                
            # Track VPVR signals
            if "VPVR" in message:
                symbol = re.search(r"(BUY|SELL) ([^ ]+)", message).group(2)
                self.signals[symbol].append({
                    "timestamp": datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f"),
                    "message": message
                })
                
        return True
    
    def analyze_performance(self):
        """Analyze trading performance from collected data"""
        if not self.trades:
            return "No trades to analyze yet"
            
        win_rate = 0
        if self.performance_metrics["total_trades"] > 0:
            win_rate = self.performance_metrics["wins"] / self.performance_metrics["total_trades"] * 100
            
        results = [
            f"===== PERFORMANCE METRICS =====",
            f"Total Trades: {self.performance_metrics['total_trades']}",
            f"Win Rate: {win_rate:.2f}%",
            f"Total Profit: {self.performance_metrics['profit']*100:.2f}%",
            f"",
            f"Performance by Symbol:"
        ]
        
        for symbol, metrics in self.performance_metrics["by_symbol"].items():
            if metrics["trades"] > 0:
                symbol_win_rate = metrics["wins"] / metrics["trades"] * 100
                results.append(f"  {symbol}: {metrics['trades']} trades, {symbol_win_rate:.2f}% win rate, {metrics['profit']*100:.2f}% profit")
                
        return "\n".join(results)
    
    def suggest_optimizations(self):
        """Suggest strategy optimizations based on performance"""
        if not self.trades:
            return "No trades to analyze yet"
            
        suggestions = ["===== OPTIMIZATION SUGGESTIONS ====="]
        
        # Analyze win rate by signal type
        signal_performance = defaultdict(lambda: {"count": 0, "wins": 0, "profit": 0.0})
        for trade in self.trades:
            if trade["pnl"] is not None:
                signal_type = "trendline" if "trendline" in trade["reason"] else "zone" if "zone" in trade["reason"] else "other"
                signal_performance[signal_type]["count"] += 1
                
                if trade["pnl"] > 0:
                    signal_performance[signal_type]["wins"] += 1
                    
                signal_performance[signal_type]["profit"] += trade["pnl"]
        
        # Make suggestions based on signal performance
        for signal_type, metrics in signal_performance.items():
            if metrics["count"] > 0:
                win_rate = metrics["wins"] / metrics["count"] * 100
                suggestions.append(f"Signal Type: {signal_type}")
                suggestions.append(f"  Trades: {metrics['count']}, Win Rate: {win_rate:.2f}%, Profit: {metrics['profit']*100:.2f}%")
                
                if win_rate < 40:
                    suggestions.append(f"  SUGGEST: Consider removing or adjusting {signal_type} signals")
                elif win_rate > 60:
                    suggestions.append(f"  SUGGEST: Consider increasing weight of {signal_type} signals")
        
        # Analyze symbol-specific performance
        for symbol, metrics in self.performance_metrics["by_symbol"].items():
            if metrics["trades"] < 3:
                continue
                
            win_rate = metrics["wins"] / metrics["trades"] * 100
            suggestions.append(f"\nSymbol: {symbol}")
            
            if win_rate < 40:
                suggestions.append(f"  Low Win Rate ({win_rate:.2f}%) - Consider adjusting parameters")
            elif metrics["profit"] < 0:
                suggestions.append(f"  Negative Profit ({metrics['profit']*100:.2f}%) - Review stop loss placement")
            elif metrics["profit"] > 0.05:
                suggestions.append(f"  Strong Performer ({metrics['profit']*100:.2f}%) - Consider increasing position size")
        
        return "\n".join(suggestions)
    
    def generate_reports(self, report_dir="reports"):
        """Generate performance reports and charts"""
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
            
        if not self.trades:
            logger.info("No trades to generate reports")
            return
        
        # Create trades dataframe
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(os.path.join(report_dir, "trades.csv"), index=False)
        
        # Generate profit chart if we have exits
        exits_df = trades_df[trades_df["pnl"].notna()]
        if not exits_df.empty:
            exits_df["cumulative_pnl"] = exits_df["pnl"].cumsum()
            
            plt.figure(figsize=(10, 6))
            plt.plot(exits_df["timestamp"], exits_df["cumulative_pnl"] * 100)
            plt.title("Cumulative Profit/Loss (%)")
            plt.xlabel("Time")
            plt.ylabel("Cumulative P&L (%)")
            plt.grid(True)
            plt.savefig(os.path.join(report_dir, "cumulative_pnl.png"))
            plt.close()
            
            # Generate win/loss chart
            win_loss = exits_df.groupby("symbol")["pnl"].agg(["count", lambda x: (x > 0).sum(), lambda x: x.sum()])
            win_loss.columns = ["trades", "wins", "profit"]
            win_loss["win_rate"] = win_loss["wins"] / win_loss["trades"] * 100
            
            plt.figure(figsize=(10, 6))
            win_loss["win_rate"].plot(kind="bar")
            plt.title("Win Rate by Symbol")
            plt.xlabel("Symbol")
            plt.ylabel("Win Rate (%)")
            plt.grid(True, axis="y")
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "win_rate.png"))
            plt.close()
        
        logger.info(f"Reports generated and saved to {report_dir}")
    
    def run(self, interval=60):
        """Run the monitor continuously"""
        logger.info(f"Starting monitor with interval {interval} seconds")
        try:
            while True:
                if self.parse_log():
                    performance = self.analyze_performance()
                    suggestions = self.suggest_optimizations()
                    
                    logger.info(f"\n{performance}")
                    logger.info(f"\n{suggestions}")
                    
                    # Generate reports every hour
                    if datetime.now().minute == 0:
                        self.generate_reports()
                        
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitor stopped")
            self.generate_reports()

def main():
    parser = argparse.ArgumentParser(description="Trading Strategy Monitor")
    parser.add_argument("--log_file", default="trading_log.txt", help="Path to trading log file")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    args = parser.parse_args()
    
    monitor = TradingMonitor(args.log_file)
    monitor.run(args.interval)

if __name__ == "__main__":
    main() 