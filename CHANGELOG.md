# Changelog

All notable changes to the Strategy Finder project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-24

### Added - Phase 2: Robustness & Out-of-Sample Evaluation

#### Walk-Forward Out-of-Sample (OOS) Evaluation
- **`--oos` flag**: Enable walk-forward out-of-sample evaluation
- **`--train_days`**: Training window size in days (default: 180)
- **`--test_days`**: Test window size in days (default: 30) 
- **`--top_k_eval`**: Number of top strategies to evaluate on test data (default: 5)
- **Sequential sliding windows**: Automatic creation of overlapping train/test periods
- **OOS metrics aggregation**: Mean P&L, win rate, and drawdown across all test windows
- **`runs/oos_summary.json`**: Comprehensive OOS evaluation results and statistics

#### Parameter Stability & Robustness Testing
- **`run_stability_test()`**: Test strategy robustness with parameter perturbations
- **`--stability_perturbs`**: Number of parameter perturbations to test (default: 10)  
- **Gaussian noise perturbation**: Small random variations (5% of parameter range)
- **Stability scoring**: Variance-based penalty for unstable strategies
- **Automatic stability analysis**: Run on best strategies after optimization

#### Performance & Parallelization
- **`--workers`**: Parallel processing with ProcessPoolExecutor
- **Multi-core optimization**: Distribute strategy evaluations across CPU cores
- **Serialization safety**: Proper handling of multiprocessing data structures
- **Graceful fallback**: Sequential mode when parallel processing unavailable
- **Memory efficiency**: Optimized for large parameter searches

#### Enhanced Artifacts & Reproducibility  
- **Structured runs directory**: Per-run metadata and results in `runs/`
- **`<uid>_meta.json`**: Run metadata with timestamps, seeds, and parameters
- **`<uid>_results.json`**: Detailed strategy results with trade ledgers
- **`<uid>_best_trades.json`**: Best strategy trade records
- **Deterministic execution**: Enhanced seed control for all random operations
- **Artifact management**: Automatic cleanup and organization

### Enhanced - Phase 1 Improvements Carried Forward

#### Risk Controls (Maintained)
- **MAX_NOTIONAL_PCT**: Notional cap enforcement (prevents hidden leverage)
- **MIN_TRADES**: Minimum trades filter for strategy validation
- **Enhanced position sizing**: Conflict resolution between constraints

#### CLI Interface Improvements
- **Additional parameters**: `--oos`, `--train_days`, `--test_days`, `--top_k_eval`
- **Performance options**: `--workers`, `--stability_perturbs`
- **Artifact control**: `--save_runs_dir`
- **Backward compatibility**: All Phase 1 parameters maintained

### Technical Improvements

#### Code Quality & Testing
- **45 comprehensive tests**: Unit tests for all major functionality
- **Phase 2 test coverage**: OOS evaluation, stability, parallel processing
- **Integration tests**: End-to-end validation of Phase 1 and Phase 2 features
- **CI/CD pipeline**: GitHub Actions workflow with multi-Python version testing
- **Error handling**: Robust exception handling for edge cases

#### Documentation & Usability
- **Enhanced help text**: Clear parameter descriptions and usage examples
- **Comprehensive logging**: Detailed progress reporting and status updates
- **Performance metrics**: Timing and efficiency reporting
- **Example commands**: Ready-to-use CLI examples in README

### Architecture Changes

#### Modular Design
- **OOS evaluation module**: Separate functions for window creation and evaluation
- **Stability testing module**: Independent parameter perturbation and analysis
- **Artifact management**: Centralized save/load functionality
- **Parallel processing**: Clean separation of serial and parallel execution paths

#### Data Structures
- **Enhanced result objects**: Additional metadata for OOS and stability metrics
- **Serialization compatibility**: All data structures support multiprocessing
- **JSON serialization**: Proper handling of timestamps and numpy arrays
- **Backward compatibility**: Phase 1 result formats maintained

### Performance Improvements
- **Parallel execution**: Up to 4x speedup with multi-core processing
- **Memory optimization**: Efficient handling of large parameter searches
- **I/O efficiency**: Batch artifact saving and structured file organization
- **Algorithmic improvements**: Optimized normalization and scoring functions

## [1.0.0] - 2025-01-24 - Phase 1: Core Risk Controls & Trade Ledger

### Added - Phase 1: Professional Risk Management

#### Enhanced Position Sizing
- **Notional cap (MAX_NOTIONAL_PCT)**: Prevent hidden leverage by capping position size to % of equity
- **`compute_qty_from_risk()`**: Professional position sizing with multiple constraints
- **Conflict resolution**: Handle cases where minimum trade size conflicts with notional cap  
- **Decimal precision**: Configurable quantity rounding (QTY_DECIMALS)

#### Detailed Trade Ledger
- **Comprehensive trade records**: Replace simple P&L list with detailed trade objects
- **Trade fields**: entry_time, exit_time, side, entry_price, exit_price, qty, pnl, reason, fees, slippage, notional_capped
- **CSV export**: `best_trades.csv` with full trade details
- **JSON format**: Structured trade data in results

#### Robustness Validation  
- **Minimum trades filter (MIN_TRADES)**: Strategies with insufficient trades marked invalid
- **Validity scoring**: Invalid strategies receive -1e6 score to exclude from ranking
- **CLI parameter**: `--min_trades` for configurable thresholds

#### Enhanced Error Handling
- **CSV validation**: Robust loading with proper error messages for malformed data
- **Simulation safety**: Try/catch blocks with graceful degradation
- **Input validation**: Check for zero/negative values, NaN, infinity
- **Detailed error reporting**: Clear messages for troubleshooting

### Enhanced CLI Interface
- **New parameters**: `--min_trades`, `--max_notional_pct`
- **Enhanced output**: Validity status, notional violations count
- **Improved summary**: Professional-grade strategy reporting
- **Warning system**: Clear guidance when no valid strategies found

### Technical Infrastructure
- **Unit tests**: 21 comprehensive tests covering CSV loading, VWAP, position sizing
- **Test coverage**: Edge cases, error conditions, numerical precision
- **File structure**: Organized codebase with proper Python package structure
- **Documentation**: Inline code documentation and usage examples

### Initial Release
- **Multi-timeframe strategy**: Supply/demand zones with EMA filters and volume confirmation
- **Professional simulation**: Realistic slippage, fees, and position sizing
- **Parameter optimization**: Random sampling with hill-climbing refinement
- **Result visualization**: Equity curves and trade analysis
- **CSV data support**: Bitcoin/USDT 1-minute OHLCV data processing

---

## Assumptions Made During Implementation

### Data Assumptions
- **Timestamp format**: Supports both millisecond integers and datetime strings
- **OHLCV structure**: Standard OHLC + volume columns required
- **Data quality**: Assumes reasonably clean data with minimal gaps
- **Frequency**: Optimized for 1-minute cryptocurrency data

### Strategy Assumptions  
- **Risk-free rate**: Not considered in strategy metrics (appropriate for crypto)
- **Slippage model**: Constant percentage (0.05%) per side, reasonable for liquid crypto pairs
- **Fee structure**: Flat 0.04% per side (typical crypto exchange taker fees)
- **Position sizing**: No fractional shares, respects exchange quantity precision rules

### OOS Evaluation Assumptions
- **Window sizing**: Default 180-day training / 30-day testing suitable for crypto volatility
- **Overlap strategy**: Test window size used as step size for walk-forward analysis
- **Minimum data**: Requires at least 100 bars training, 50 bars testing per window
- **Evaluation metric**: Net P&L used as primary OOS ranking criterion

### Performance Assumptions
- **Parallel processing**: Assumes ProcessPoolExecutor availability (standard Python 3.2+)
- **Memory usage**: Designed for datasets up to several million bars on typical hardware
- **CPU utilization**: Worker count should not exceed available CPU cores for optimal performance
- **I/O performance**: Artifact saving assumes reasonable disk write speeds

### Security & Privacy Notes
- **Sample data**: Included test data is synthetic/anonymized - no real trading data exposed
- **No credentials**: System does not store or transmit any API keys or trading credentials  
- **Local execution**: All processing occurs locally - no external API calls required
- **Artifact privacy**: Saved artifacts contain only strategy parameters and synthetic results