#!/usr/bin/env python3
"""
Strategy Optimizer
Analyzes trading performance and suggests parameter adjustments
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("optimizer_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("optimizer")

class StrategyOptimizer:
    def __init__(self, trades_file, config_file=None):
        self.trades_file = trades_file
        self.config_file = config_file
        self.trades_df = None
        self.current_params = self._load_current_params()
        self.suggested_params = {}
        self.symbols = []
        self.performance_metrics = {}
    
    def _load_current_params(self) -> Dict[str, Any]:
        """Load current strategy parameters"""
        # Default parameters from live_trader.py
        params = {
            "reconnect_delay": 5,
            "max_candles": 500,
            "vpvr_window": 200,
            "num_bins": 50, 
            "top_n_peaks": 3,
            "top_n_valleys": 3,
            "trend_lookback": 100,
            "entry_threshold_pct": 1.0,  # Within 1%
            "confirmation_threshold_pct": 2.0,  # Within 2%
            "stop_loss_pct": 2.0  # 2% below/above entry
        }
        
        # Load from config file if it exists
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved_params = json.load(f)
                params.update(saved_params)
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                
        return params
    
    def load_trades(self) -> bool:
        """Load trades from CSV file"""
        if not os.path.exists(self.trades_file):
            logger.error(f"Trades file not found: {self.trades_file}")
            return False
            
        try:
            self.trades_df = pd.read_csv(self.trades_file)
            self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
            self.symbols = self.trades_df['symbol'].unique().tolist()
            return True
        except Exception as e:
            logger.error(f"Error loading trades file: {e}")
            return False
    
    def calculate_metrics(self) -> None:
        """Calculate performance metrics from trades"""
        if self.trades_df is None or self.trades_df.empty:
            logger.warning("No trades to analyze")
            return
            
        # Filter to completed trades (those with PnL)
        completed = self.trades_df[self.trades_df['pnl'].notna()].copy()
        if completed.empty:
            logger.warning("No completed trades to analyze")
            return
            
        # Overall metrics
        wins = completed['pnl'] > 0
        self.performance_metrics['overall'] = {
            'total_trades': len(completed),
            'win_rate': wins.mean() * 100,
            'profit': completed['pnl'].sum() * 100,  # As percentage
            'avg_win': completed.loc[wins, 'pnl'].mean() * 100 if wins.any() else 0,
            'avg_loss': completed.loc[~wins, 'pnl'].mean() * 100 if (~wins).any() else 0,
        }
        
        # By symbol
        for symbol in self.symbols:
            symbol_trades = completed[completed['symbol'] == symbol]
            if symbol_trades.empty:
                continue
                
            symbol_wins = symbol_trades['pnl'] > 0
            self.performance_metrics[symbol] = {
                'total_trades': len(symbol_trades),
                'win_rate': symbol_wins.mean() * 100 if len(symbol_trades) > 0 else 0,
                'profit': symbol_trades['pnl'].sum() * 100,
                'avg_win': symbol_trades.loc[symbol_wins, 'pnl'].mean() * 100 if symbol_wins.any() else 0,
                'avg_loss': symbol_trades.loc[~symbol_wins, 'pnl'].mean() * 100 if (~symbol_wins).any() else 0,
            }
            
        # By reason type
        reason_types = []
        for reason in completed['reason']:
            if 'trendline' in reason.lower():
                reason_types.append('trendline')
            elif 'zone' in reason.lower():
                reason_types.append('zone')
            else:
                reason_types.append('other')
                
        completed['reason_type'] = reason_types
        
        for reason_type in ['trendline', 'zone', 'other']:
            type_trades = completed[completed['reason_type'] == reason_type]
            if type_trades.empty:
                continue
                
            type_wins = type_trades['pnl'] > 0
            self.performance_metrics[f'reason_{reason_type}'] = {
                'total_trades': len(type_trades),
                'win_rate': type_wins.mean() * 100 if len(type_trades) > 0 else 0,
                'profit': type_trades['pnl'].sum() * 100,
                'avg_win': type_trades.loc[type_wins, 'pnl'].mean() * 100 if type_wins.any() else 0,
                'avg_loss': type_trades.loc[~type_wins, 'pnl'].mean() * 100 if (~type_wins).any() else 0,
            }
    
    def optimize_parameters(self) -> Dict[str, Any]:
        """Generate optimized parameters based on performance metrics"""
        if not self.performance_metrics:
            return self.current_params.copy()
            
        optimized = self.current_params.copy()
        
        # 1. Optimize VPVR parameters based on overall performance
        overall = self.performance_metrics.get('overall', {})
        
        if overall.get('win_rate', 0) < 45:
            # If win rate is low, try to be more selective with entries
            optimized['top_n_peaks'] = max(1, optimized['top_n_peaks'] - 1)
            optimized['top_n_valleys'] = max(1, optimized['top_n_valleys'] - 1)
            optimized['entry_threshold_pct'] = min(0.8, optimized['entry_threshold_pct'])
            logger.info("Low win rate detected - making entry conditions more selective")
            
        elif overall.get('win_rate', 0) > 60:
            # If win rate is high, can be slightly less selective to get more trades
            optimized['entry_threshold_pct'] = min(1.5, optimized['entry_threshold_pct'] * 1.1)
            logger.info("High win rate detected - slightly relaxing entry conditions")
            
        # 2. Optimize based on reason type performance
        trendline_metrics = self.performance_metrics.get('reason_trendline', {})
        zone_metrics = self.performance_metrics.get('reason_zone', {})
        
        # If one confirmation type is performing significantly better
        if (trendline_metrics.get('win_rate', 0) > zone_metrics.get('win_rate', 0) + 15 and 
            trendline_metrics.get('total_trades', 0) >= 3):
            # Trendlines are performing much better
            optimized['confirmation_threshold_pct'] = max(1.0, optimized['confirmation_threshold_pct'] * 0.8)
            optimized['trend_lookback'] = min(150, int(optimized['trend_lookback'] * 1.1))
            logger.info("Trendline signals performing better - optimizing for trendline detection")
            
        elif (zone_metrics.get('win_rate', 0) > trendline_metrics.get('win_rate', 0) + 15 and 
              zone_metrics.get('total_trades', 0) >= 3):
            # Zones are performing much better
            optimized['confirmation_threshold_pct'] = max(1.0, optimized['confirmation_threshold_pct'] * 0.8)
            logger.info("Zone signals performing better - optimizing for S/R zone detection")
            
        # 3. Optimize stop loss based on average loss size
        if overall.get('avg_loss', 0) < -2.0:
            # Losses are too big, tighten stops
            optimized['stop_loss_pct'] = max(1.0, optimized['stop_loss_pct'] * 0.9)
            logger.info("Large average losses detected - tightening stop loss")
        elif overall.get('avg_win', 0) > 2.0 * abs(overall.get('avg_loss', 0)):
            # Good risk:reward, can widen stops slightly to reduce whipsaws
            optimized['stop_loss_pct'] = min(3.0, optimized['stop_loss_pct'] * 1.1)
            logger.info("Good risk:reward ratio - slightly widening stop loss")
            
        # 4. Optimize VPVR window based on activity
        recent_count = 0
        if self.trades_df is not None:
            recent_cutoff = datetime.now() - timedelta(hours=4)
            recent_count = len(self.trades_df[self.trades_df['timestamp'] > recent_cutoff])
            
        if recent_count <= 2:
            # Very few trades, may need to look at larger window
            optimized['vpvr_window'] = min(300, int(optimized['vpvr_window'] * 1.2))
            logger.info("Few recent trades - increasing VPVR window")
        elif recent_count >= 10:
            # Many trades, may benefit from more focused analysis
            optimized['vpvr_window'] = max(100, int(optimized['vpvr_window'] * 0.9))
            logger.info("Many recent trades - decreasing VPVR window for more focused analysis")
            
        # 5. Make symbol-specific adjustments
        symbol_adjustments = {}
        for symbol in self.symbols:
            if symbol not in self.performance_metrics:
                continue
                
            metrics = self.performance_metrics[symbol]
            if metrics.get('total_trades', 0) < 3:
                continue
                
            # Create symbol-specific adjustment if needed
            if metrics.get('win_rate', 0) < 40:
                # Poor performance, needs custom parameters
                symbol_adjustments[symbol] = {
                    'entry_threshold_pct': max(0.5, optimized['entry_threshold_pct'] * 0.8),
                    'confirmation_threshold_pct': max(1.0, optimized['confirmation_threshold_pct'] * 0.8),
                    'stop_loss_pct': max(1.0, optimized['stop_loss_pct'] * 0.8)
                }
                logger.info(f"Creating custom parameters for underperforming symbol: {symbol}")
            elif metrics.get('win_rate', 0) > 65:
                # Great performance, could be more aggressive
                symbol_adjustments[symbol] = {
                    'entry_threshold_pct': min(2.0, optimized['entry_threshold_pct'] * 1.2),
                    'stop_loss_pct': min(3.0, optimized['stop_loss_pct'] * 1.2)
                }
                logger.info(f"Creating custom parameters for high-performing symbol: {symbol}")
        
        if symbol_adjustments:
            optimized['symbol_specific'] = symbol_adjustments
        
        return optimized
    
    def save_optimized_params(self, output_file: str) -> None:
        """Save optimized parameters to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.suggested_params, f, indent=2)
            logger.info(f"Optimized parameters saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving optimized parameters: {e}")
    
    def run(self, output_file: str = "optimized_params.json") -> None:
        """Run the optimization process"""
        logger.info("Starting strategy optimization...")
        
        if not self.load_trades():
            return
            
        self.calculate_metrics()
        
        if not self.performance_metrics:
            logger.warning("No performance metrics available for optimization")
            return
            
        self.suggested_params = self.optimize_parameters()
        
        # Log optimization results
        logger.info("\n===== OPTIMIZATION RESULTS =====")
        logger.info("Performance Metrics:")
        overall = self.performance_metrics.get('overall', {})
        logger.info(f"  Total Trades: {overall.get('total_trades', 0)}")
        logger.info(f"  Win Rate: {overall.get('win_rate', 0):.2f}%")
        logger.info(f"  Total P&L: {overall.get('profit', 0):.2f}%")
        
        logger.info("\nParameter Changes:")
        for key, value in self.suggested_params.items():
            if key == 'symbol_specific':
                logger.info(f"  {key}: Symbol-specific adjustments created")
                for symbol, params in value.items():
                    logger.info(f"    {symbol}: {params}")
            elif key in self.current_params and value != self.current_params[key]:
                logger.info(f"  {key}: {self.current_params[key]} -> {value}")
                
        # Save the optimized parameters
        self.save_optimized_params(output_file)

def main():
    parser = argparse.ArgumentParser(description="Strategy Optimizer")
    parser.add_argument("--trades_file", default="reports/trades.csv",
                        help="Path to trades CSV file")
    parser.add_argument("--config_file", default=None,
                        help="Path to current configuration file (optional)")
    parser.add_argument("--output_file", default="optimized_params.json",
                        help="Path to output optimized parameters")
    args = parser.parse_args()
    
    optimizer = StrategyOptimizer(args.trades_file, args.config_file)
    optimizer.run(args.output_file)

if __name__ == "__main__":
    main() 