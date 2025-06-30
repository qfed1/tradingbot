#!/usr/bin/env python3
"""
Live Swing Trading Strategy using CoinAPI WebSocket
Based on VPVR + trendline/structure analysis
"""
import argparse
import json
import logging
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import websocket
from numba import njit, prange

# ─── Logger Setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger("live_trader")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(handler)

# ─── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "reconnect_delay": 5,    # seconds
    "max_candles": 500,      # max candles to store per symbol
    "vpvr_window": 200,      # number of candles to use for VPVR calculation
    "num_bins": 50,          # number of bins for VPVR histogram
    "top_n_peaks": 3,        # number of top volume peaks to consider
    "top_n_valleys": 3,      # number of top volume valleys to consider
    "trend_lookback": 100,   # number of candles to use for trendline detection
    "entry_threshold_pct": 1.0,      # within 1%
    "confirmation_threshold_pct": 2.0,  # within 2%
    "stop_loss_pct": 2.0     # 2% below/above entry
}

# ─── Data Structures ────────────────────────────────────────────────────────────
@dataclass
class Trade:
    """Represents a trade (entry or exit)"""
    symbol: str
    side: str
    price: float
    time: datetime
    reason: str
    stop_loss: float = None
    pnl: float = None

@dataclass
class Position:
    """Represents a current position"""
    symbol: str
    side: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    entry_reason: str
    size: float = 1.0

# ─── VPVR and Trendline Functions ─────────────────────────────────────────────
@njit(cache=True, fastmath=True, parallel=True)
def calc_vpvr_numba_optimized(
    low_arr: np.ndarray, high_arr: np.ndarray, vol_arr: np.ndarray,
    num_bins: int, low_val: float, high_val: float
) -> Tuple[np.ndarray, float]:
    """Calculate VPVR histogram using Numba for performance"""
    height = (high_val - low_val) / num_bins
    inv_h = 1.0 / height
    vol_hist = np.zeros(num_bins, dtype=np.float64)
    for i in prange(len(low_arr)):
        l, h, v = low_arr[i], high_arr[i], vol_arr[i]
        if h <= low_val or l >= high_val or h - l <= 0:
            continue
        cl = max(l, low_val); ch = min(h, high_val)
        start = int((cl - low_val) * inv_h); end = int((ch - low_val) * inv_h)
        bar_h = h - l; factor = v / bar_h
        for b in range(max(0, start), min(num_bins, end + 1)):
            bin_low = low_val + b * height
            bin_high = bin_low + height
            overlap = min(bin_high, ch) - max(bin_low, cl)
            if overlap > 0:
                vol_hist[b] += factor * overlap
    return vol_hist, height

def detect_trendlines(prices: np.ndarray, window: int = 100) -> Tuple[Optional[float], Optional[float]]:
    """
    Detect support and resistance trendlines from price data
    Returns: (support_slope, support_intercept), (resistance_slope, resistance_intercept)
    """
    if len(prices) < window:
        return None, None
    
    # Use recent window of prices
    recent_prices = prices[-window:]
    x = np.arange(len(recent_prices))
    
    # Find local highs and lows
    highs = []
    lows = []
    
    for i in range(1, len(recent_prices) - 1):
        if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
            highs.append((i, recent_prices[i]))
        if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
            lows.append((i, recent_prices[i]))
    
    # Need at least 2 points to form a line
    if len(highs) >= 2:
        resistance_x = np.array([p[0] for p in highs])
        resistance_y = np.array([p[1] for p in highs])
        resistance_slope, resistance_intercept = np.polyfit(resistance_x, resistance_y, 1)
    else:
        resistance_slope, resistance_intercept = None, None
    
    if len(lows) >= 2:
        support_x = np.array([p[0] for p in lows])
        support_y = np.array([p[1] for p in lows])
        support_slope, support_intercept = np.polyfit(support_x, support_y, 1)
    else:
        support_slope, support_intercept = None, None
    
    return (support_slope, support_intercept), (resistance_slope, resistance_intercept)

def identify_support_resistance_zones(df: pd.DataFrame, window: int = 20, threshold: float = 0.01) -> List[float]:
    """Identify horizontal support/resistance zones based on price clusters"""
    zones = []
    
    # Use closing prices for simplicity
    prices = df['close'].values
    
    # Find price clusters
    for i in range(len(prices) - window + 1):
        price_window = prices[i:i+window]
        mean_price = np.mean(price_window)
        
        # Check if prices cluster around this mean
        cluster_count = np.sum(np.abs(price_window - mean_price) / mean_price < threshold)
        
        if cluster_count > window * 0.7:  # If 70% of prices are within threshold
            zones.append(mean_price)
    
    # Merge nearby zones
    if not zones:
        return []
    
    zones = sorted(zones)
    merged_zones = [zones[0]]
    
    for zone in zones[1:]:
        if abs(zone - merged_zones[-1]) / merged_zones[-1] < threshold:
            # Update with average if zones are close
            merged_zones[-1] = (merged_zones[-1] + zone) / 2
        else:
            merged_zones.append(zone)
    
    return merged_zones

# ─── Trading Strategy ────────────────────────────────────────────────────────
class VPVRTrendStrategy:
    def __init__(self, symbols: List[str], risk_pct: float = 0.01, config: Dict = None):
        """Initialize the strategy"""
        self.symbols = symbols
        self.risk_pct = risk_pct
        self.candles: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, Position] = {}
        self.trades_history: List[Trade] = []
        self.sr_zones: Dict[str, List[float]] = {symbol: [] for symbol in symbols}
        
        # Load configuration parameters
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        # Symbol-specific configurations
        self.symbol_config: Dict[str, Dict] = {}
        if config and 'symbol_specific' in config:
            self.symbol_config = config['symbol_specific']
            logger.info(f"Loaded symbol-specific configurations for: {list(self.symbol_config.keys())}")
        
        logger.info(f"Strategy initialized with parameters: {self.config}")
        
    def _get_param(self, param_name: str, symbol: str = None) -> any:
        """Get parameter value, checking symbol-specific config first"""
        if symbol and symbol in self.symbol_config and param_name in self.symbol_config[symbol]:
            return self.symbol_config[symbol][param_name]
        return self.config.get(param_name, DEFAULT_CONFIG.get(param_name))
    
    def update_candles(self, symbol: str, ohlcv_data: dict) -> None:
        """Update candle data with new OHLCV information"""
        if symbol not in self.candles:
            self.candles[symbol] = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            self.candles[symbol].set_index('time', inplace=True)
        
        # Convert to pandas timestamp
        time_start = pd.Timestamp(ohlcv_data['time_period_start'])
        
        # Check if we already have this candle
        if time_start in self.candles[symbol].index:
            # Update existing candle
            self.candles[symbol].loc[time_start] = [
                ohlcv_data['price_open'],
                ohlcv_data['price_high'],
                ohlcv_data['price_low'],
                ohlcv_data['price_close'],
                ohlcv_data['volume_traded']
            ]
        else:
            # Add new candle
            new_row = pd.DataFrame(
                [[ohlcv_data['price_open'], 
                  ohlcv_data['price_high'], 
                  ohlcv_data['price_low'], 
                  ohlcv_data['price_close'], 
                  ohlcv_data['volume_traded']]],
                index=[time_start],
                columns=['open', 'high', 'low', 'close', 'volume']
            )
            self.candles[symbol] = pd.concat([self.candles[symbol], new_row])
            
            # Sort by time and keep only the most recent MAX_CANDLES
            max_candles = self._get_param('max_candles')
            self.candles[symbol] = self.candles[symbol].sort_index().iloc[-max_candles:]
            
            # Update support/resistance zones every 10 candles
            if len(self.candles[symbol]) % 10 == 0:
                self.sr_zones[symbol] = identify_support_resistance_zones(self.candles[symbol])
            
            # Process new candle for trading signals
            self.process_new_candle(symbol)
    
    def process_new_candle(self, symbol: str) -> None:
        """Process a new candle for trading signals"""
        # Skip if we don't have enough data
        vpvr_window = self._get_param('vpvr_window', symbol)
        if len(self.candles[symbol]) < vpvr_window:
            logger.info(f"Not enough data for {symbol}, waiting for more candles...")
            return
        
        df = self.candles[symbol]
        
        # Check for entry signals if we're not in a position
        if symbol not in self.positions:
            self._check_entry_signals(symbol, df)
        # Check for exit signals if we're in a position
        else:
            self._check_exit_signals(symbol, df)
    
    def _check_entry_signals(self, symbol: str, df: pd.DataFrame) -> None:
        """Check for entry signals based on VPVR and trendlines"""
        # Get parameters for this symbol
        vpvr_window = self._get_param('vpvr_window', symbol)
        num_bins = self._get_param('num_bins', symbol)
        top_n_peaks = self._get_param('top_n_peaks', symbol)
        top_n_valleys = self._get_param('top_n_valleys', symbol)
        entry_threshold_pct = self._get_param('entry_threshold_pct', symbol) / 100
        confirmation_threshold_pct = self._get_param('confirmation_threshold_pct', symbol) / 100
        
        # Calculate VPVR
        window_data = df.iloc[-vpvr_window:]
        lows, highs, vols = window_data['low'].values, window_data['high'].values, window_data['volume'].values
        vmin, vmax = float(lows.min()), float(highs.max())
        
        profile, height = calc_vpvr_numba_optimized(
            lows, highs, vols, num_bins, vmin, vmax
        )
        
        # Find peaks and valleys
        peaks = sorted([vmin + (b + 0.5)*height for b in 
                       np.argpartition(profile, -top_n_peaks)[-top_n_peaks:]], reverse=True)
        valleys = sorted([vmin + (b + 0.5)*height for b in 
                         np.argpartition(profile, top_n_valleys)[:top_n_valleys]])
        
        # Get current price and previous close
        current_price = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # Detect trendlines
        trend_lookback = self._get_param('trend_lookback', symbol)
        support_line, resistance_line = detect_trendlines(df['close'].values, trend_lookback)
        
        # Check for long entry
        buy_levels = [v for v in valleys if v > prev_close]
        if buy_levels:
            buy_trigger = min(buy_levels)
            
            # Check if price is near buy trigger
            if abs(current_price - buy_trigger) / buy_trigger < entry_threshold_pct:
                
                # Confirm with trendline or support zone
                confirmed = False
                reason = ""
                
                # Check support trendline
                if support_line is not None:
                    support_slope, support_intercept = support_line
                    current_idx = len(df) - 1
                    support_value = support_slope * (current_idx - trend_lookback + len(df[-trend_lookback:])) + support_intercept
                    
                    if abs(current_price - support_value) / support_value < confirmation_threshold_pct:
                        confirmed = True
                        reason = f"VPVR valley at {buy_trigger:.2f} with trendline support at {support_value:.2f}"
                
                # Check support zones
                if not confirmed and self.sr_zones[symbol]:
                    for zone in self.sr_zones[symbol]:
                        if abs(current_price - zone) / zone < confirmation_threshold_pct:
                            confirmed = True
                            reason = f"VPVR valley at {buy_trigger:.2f} with S/R zone support at {zone:.2f}"
                            break
                
                # If confirmed, enter long position
                if confirmed:
                    # Calculate stop loss (below nearest support or 2% below entry)
                    stop_loss_pct = self._get_param('stop_loss_pct', symbol) / 100
                    if support_line is not None:
                        stop_loss = min(support_value * (1 - stop_loss_pct/2), current_price * (1 - stop_loss_pct))
                    else:
                        stop_loss = current_price * (1 - stop_loss_pct)
                    
                    # Enter position
                    self._enter_position(symbol, "BUY", current_price, stop_loss, reason)
        
        # Check for short entry
        sell_levels = [p for p in peaks if p < prev_close]
        if sell_levels:
            sell_trigger = max(sell_levels)
            
            # Check if price is near sell trigger
            if abs(current_price - sell_trigger) / sell_trigger < entry_threshold_pct:
                
                # Confirm with trendline or resistance zone
                confirmed = False
                reason = ""
                
                # Check resistance trendline
                if resistance_line is not None:
                    resistance_slope, resistance_intercept = resistance_line
                    current_idx = len(df) - 1
                    resistance_value = resistance_slope * (current_idx - trend_lookback + len(df[-trend_lookback:])) + resistance_intercept
                    
                    if abs(current_price - resistance_value) / resistance_value < confirmation_threshold_pct:
                        confirmed = True
                        reason = f"VPVR peak at {sell_trigger:.2f} with trendline resistance at {resistance_value:.2f}"
                
                # Check resistance zones
                if not confirmed and self.sr_zones[symbol]:
                    for zone in self.sr_zones[symbol]:
                        if abs(current_price - zone) / zone < confirmation_threshold_pct:
                            confirmed = True
                            reason = f"VPVR peak at {sell_trigger:.2f} with S/R zone resistance at {zone:.2f}"
                            break
                
                # If confirmed, enter short position
                if confirmed:
                    # Calculate stop loss (above nearest resistance or 2% above entry)
                    stop_loss_pct = self._get_param('stop_loss_pct', symbol) / 100
                    if resistance_line is not None:
                        stop_loss = max(resistance_value * (1 + stop_loss_pct/2), current_price * (1 + stop_loss_pct))
                    else:
                        stop_loss = current_price * (1 + stop_loss_pct)
                    
                    # Enter position
                    self._enter_position(symbol, "SELL", current_price, stop_loss, reason)
    
    def _check_exit_signals(self, symbol: str, df: pd.DataFrame) -> None:
        """Check for exit signals based on trendlines and stop loss"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        current_price = df['close'].iloc[-1]
        
        # Check stop loss
        if (position.side == "BUY" and current_price <= position.stop_loss) or \
           (position.side == "SELL" and current_price >= position.stop_loss):
            self._exit_position(symbol, current_price, "Stop loss hit")
            return
        
        # Check trendline break
        support_line, resistance_line = detect_trendlines(df['close'].values)
        
        if position.side == "BUY" and support_line is not None:
            support_slope, support_intercept = support_line
            current_idx = len(df) - 1
            support_value = support_slope * (current_idx - TREND_LOOKBACK + len(df[-TREND_LOOKBACK:])) + support_intercept
            
            # Exit long if price breaks below support trendline
            if current_price < support_value:
                self._exit_position(symbol, current_price, "Support trendline broken")
        
        elif position.side == "SELL" and resistance_line is not None:
            resistance_slope, resistance_intercept = resistance_line
            current_idx = len(df) - 1
            resistance_value = resistance_slope * (current_idx - TREND_LOOKBACK + len(df[-TREND_LOOKBACK:])) + resistance_intercept
            
            # Exit short if price breaks above resistance trendline
            if current_price > resistance_value:
                self._exit_position(symbol, current_price, "Resistance trendline broken")
    
    def _enter_position(self, symbol: str, side: str, price: float, stop_loss: float, reason: str) -> None:
        """Enter a new position"""
        # Create a new position
        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            entry_time=datetime.now(timezone.utc),
            stop_loss=stop_loss,
            entry_reason=reason
        )
        
        # Log the trade
        self.execute_trade(symbol, side, price, reason, stop_loss)
    
    def _exit_position(self, symbol: str, price: float, reason: str) -> None:
        """Exit an existing position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.side == "BUY":
            pnl = (price - position.entry_price) / position.entry_price
            exit_side = "SELL"
        else:  # SELL
            pnl = (position.entry_price - price) / position.entry_price
            exit_side = "BUY"
        
        # Log the trade
        self.execute_trade(symbol, exit_side, price, reason, None, pnl)
        
        # Remove the position
        del self.positions[symbol]
    
    def execute_trade(self, symbol: str, side: str, price: float, reason: str, 
                     stop_loss: float = None, pnl: float = None) -> None:
        """Execute a trade (log to console for now)"""
        trade = Trade(
            symbol=symbol,
            side=side,
            price=price,
            time=datetime.now(timezone.utc),
            reason=reason,
            stop_loss=stop_loss,
            pnl=pnl
        )
        
        # Add to trade history
        self.trades_history.append(trade)
        
        # Format PnL string if available
        pnl_str = f" | P&L: {pnl*100:.2f}%" if pnl is not None else ""
        stop_str = f" | Stop: {stop_loss:.2f}" if stop_loss is not None else ""
        
        # Log the trade
        logger.info(f"{trade.time} | {side} {symbol} @ {price:.2f}{stop_str} | Reason: {reason}{pnl_str}")

# ─── WebSocket Client ────────────────────────────────────────────────────────
class CoinAPIWebSocketClient:
    def __init__(self, api_key: str, symbols: List[str], strategy: VPVRTrendStrategy):
        """Initialize the WebSocket client"""
        self.api_key = api_key
        self.symbols = symbols
        self.strategy = strategy
        self.ws = None
        self.is_connected = False
        self.reconnect_required = False
        self.reconnect_timer = None
    
    def start(self):
        """Start the WebSocket connection"""
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp("wss://ws.coinapi.io/v1/",
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close,
                                         on_open=self.on_open)
        
        # Start WebSocket connection in a separate thread
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
        
        # Main loop to handle reconnections
        try:
            while True:
                if self.reconnect_required:
                    logger.info("Reconnecting to WebSocket...")
                    self.reconnect_required = False
                    
                    if self.ws:
                        self.ws.close()
                    
                    time.sleep(RECONNECT_DELAY)
                    
                    self.ws = websocket.WebSocketApp("wss://ws.coinapi.io/v1/",
                                                   on_message=self.on_message,
                                                   on_error=self.on_error,
                                                   on_close=self.on_close,
                                                   on_open=self.on_open)
                    
                    wst = threading.Thread(target=self.ws.run_forever)
                    wst.daemon = True
                    wst.start()
                
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.ws.close()
    
    def on_open(self, ws):
        """Called when the WebSocket connection is established"""
        logger.info("WebSocket connection established")
        self.is_connected = True
        
        # Prepare symbol IDs for subscription
        symbol_ids = [f"{symbol}$" for symbol in self.symbols]
        
        # Send hello message to subscribe to OHLCV data
        hello_msg = {
            "type": "hello",
            "apikey": self.api_key,
            "heartbeat": True,
            "subscribe_data_type": ["ohlcv", "trade"],
            "subscribe_filter_symbol_id": symbol_ids,
            "subscribe_filter_period_id": ["1MIN"]
        }
        
        ws.send(json.dumps(hello_msg))
        logger.info(f"Subscribed to {len(self.symbols)} symbols: {', '.join(self.symbols)}")
    
    def on_message(self, ws, message):
        """Called when a message is received from the WebSocket"""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if data["type"] == "ohlcv":
                symbol = data["symbol_id"]
                self.strategy.update_candles(symbol, data)
            
            elif data["type"] == "error":
                logger.error(f"Error from WebSocket: {data['message']}")
                self.reconnect_required = True
            
            elif data["type"] == "reconnect":
                logger.info(f"Reconnect message received: {data}")
                self.reconnect_required = True
            
            elif data["type"] == "hearbeat":
                # Just a heartbeat, ignore
                pass
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            logger.error(f"Message: {message}")
    
    def on_error(self, ws, error):
        """Called when a WebSocket error occurs"""
        logger.error(f"WebSocket error: {error}")
        self.reconnect_required = True
    
    def on_close(self, ws, close_status_code, close_msg):
        """Called when the WebSocket connection is closed"""
        logger.info(f"WebSocket connection closed: {close_status_code} {close_msg}")
        self.is_connected = False
        self.reconnect_required = True

# ─── Main Function ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VPVR+Trendline Live Trading Strategy")
    parser.add_argument("--api_key", required=True, help="CoinAPI API key")
    parser.add_argument("--symbols", required=True, nargs="+", 
                        help="Symbols to trade (e.g. BITSTAMP_SPOT_BTC_USD)")
    parser.add_argument("--risk_pct", type=float, default=0.01, 
                        help="Risk percentage per trade (default: 0.01)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configuration JSON file with strategy parameters")
    args = parser.parse_args()
    
    # Load configuration if specified
    config = None
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    # Initialize strategy
    strategy = VPVRTrendStrategy(args.symbols, args.risk_pct, config)
    
    # Initialize WebSocket client
    client = CoinAPIWebSocketClient(args.api_key, args.symbols, strategy)
    
    # Start trading
    logger.info(f"Starting live trading for symbols: {', '.join(args.symbols)}")
    client.start()

if __name__ == "__main__":
    main() 