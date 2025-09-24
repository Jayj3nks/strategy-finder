# Professional Strategy Finder

A professional-grade quantitative trading strategy finder for cryptocurrency markets, featuring robust risk controls, out-of-sample validation, and comprehensive backtesting capabilities.

## 🎯 Features

### Phase 1: Core Risk Management ✅
- **Notional Cap Enforcement**: Prevent hidden leverage with configurable position limits
- **Detailed Trade Ledger**: Comprehensive trade records with fees, slippage, and exit reasons
- **Minimum Trades Validation**: Filter strategies requiring sufficient trade count for statistical significance
- **Professional Position Sizing**: Advanced risk management with conflict resolution

### Phase 2: Robustness & OOS Evaluation ✅  
- **Walk-Forward Out-of-Sample**: Rigorous evaluation on unseen data using sliding windows
- **Parameter Stability Testing**: Robustness analysis through parameter perturbation
- **Parallel Processing**: Multi-core optimization for faster parameter searches
- **Comprehensive Artifacts**: Structured result storage with metadata and reproducibility

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd strategy-finder

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```bash
# Simple strategy optimization
python strategy_finder.py --csv sample_BTCUSDT_1m.csv --budget 200 --seed 42

# Quick test run
python strategy_finder.py --csv sample_BTCUSDT_1m.csv --budget 40 --quick --workers 2

# Out-of-sample evaluation
python strategy_finder.py --csv sample_BTCUSDT_1m.csv --oos --train_days 90 --test_days 14 --top_k_eval 5

# Parallel processing
python strategy_finder.py --csv sample_BTCUSDT_1m.csv --budget 100 --workers 4 --seed 42
```

### Running Tests
```bash
# Run all tests
pytest -v

# Run specific test modules  
pytest tests/test_oos_evaluation.py -v
pytest tests/test_notional_cap.py -v

# Run integration tests
python test_phase1_integration.py
python test_phase2_integration.py
```

## 📊 Strategy Overview

The strategy finder implements a multi-timeframe supply and demand zone strategy:

- **Higher Timeframe Filter**: 4-hour EMA trend direction
- **Mid Timeframe Pullback**: 30-minute EMA counter-trend for entries  
- **Supply/Demand Zones**: Pivot-based zones with ATR sizing
- **Structure Confirmation**: 5-minute swing highs/lows and volume
- **Risk Management**: ATR-based stops with R:R targets, VWAP exits

## 🔧 Configuration

### CLI Parameters

#### Basic Parameters
- `--csv`: Path to OHLCV CSV file (required)
- `--budget`: Number of backtests to run (default: 200)
- `--seed`: Random seed for reproducibility (default: 42)
- `--quick`: Reduced budget and parameter ranges for testing

#### Risk Management
- `--min_trades`: Minimum trades required for valid strategy (default: 50)
- `--max_notional_pct`: Maximum position size as % of equity (default: 1.0)

#### Out-of-Sample Evaluation
- `--oos`: Enable walk-forward OOS evaluation
- `--train_days`: Training window size in days (default: 180)
- `--test_days`: Test window size in days (default: 30)
- `--top_k_eval`: Top strategies to evaluate OOS (default: 5)

#### Performance & Stability
- `--workers`: Number of parallel workers (default: 1)
- `--stability_perturbs`: Parameter perturbations for stability (default: 10)
- `--save_runs_dir`: Directory for run artifacts (default: "runs")

### CSV Format

Required columns:
- `open_time` or `timestamp`: Unix milliseconds or datetime string
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume

Example:
```csv
open_time,open,high,low,close,volume
1672531200000,41000.0,41100.0,40900.0,41050.0,125.5
1672531260000,41050.0,41150.0,40950.0,41100.0,98.2
```

## 📁 Output Files

### Standard Outputs
- `search_results.csv`: All strategy results with parameters and metrics
- `best_equity.png`: Equity curve visualization for best strategy
- `best_trades.csv`: Detailed trade ledger for best strategy

### Enhanced Artifacts (runs/ directory)
- `<run_uid>_meta.json`: Run metadata with timestamps and configuration
- `<run_uid>_results.json`: Complete strategy results with trade ledgers
- `<run_uid>_best_trades.json`: Best strategy trade records
- `oos_summary.json`: Out-of-sample evaluation summary (when using --oos)

## 🧪 Testing

### Test Coverage
- **45 comprehensive tests** covering all major functionality
- **Unit tests**: CSV loading, VWAP calculation, position sizing, OOS evaluation, stability testing
- **Integration tests**: End-to-end validation of Phase 1 and Phase 2 features
- **Parallel processing tests**: Multiprocessing compatibility and performance
- **Error handling tests**: Edge cases and graceful degradation

### Test Categories
- `tests/test_load_csv.py`: Data loading and validation
- `tests/test_vwap.py`: VWAP calculation accuracy  
- `tests/test_notional_cap.py`: Position sizing and risk controls
- `tests/test_oos_evaluation.py`: Out-of-sample window creation and evaluation
- `tests/test_stability.py`: Parameter perturbation and stability analysis
- `tests/test_parallel.py`: Parallel processing functionality

## 🏗️ Architecture

### Core Components
```
strategy_finder.py          # Main application
├── Data Loading            # CSV parsing and validation
├── Indicators              # Multi-timeframe technical indicators  
├── Strategy Logic          # Supply/demand zone implementation
├── Risk Management         # Position sizing with notional caps
├── Search Algorithm        # Parameter optimization with perturbations
├── OOS Evaluation         # Walk-forward validation framework
├── Stability Testing     # Parameter robustness analysis
├── Parallel Processing   # Multi-core optimization
└── Artifact Management   # Result storage and organization
```

### Key Classes & Functions
- `load_csv()`: Robust CSV loading with error handling
- `simulate_trades()`: Core backtesting engine with detailed trade ledger
- `compute_qty_from_risk()`: Professional position sizing with constraints
- `run_search()`: Parameter optimization with optional parallelization
- `run_oos_evaluation()`: Walk-forward out-of-sample validation
- `run_stability_test()`: Parameter perturbation analysis
- `save_run_artifacts()`: Comprehensive result storage

## ⚡ Performance

### Optimization Features
- **Parallel Processing**: Up to 4x speedup with multi-core execution
- **Memory Efficiency**: Optimized for large parameter searches
- **Vectorized Calculations**: NumPy-based indicator computations
- **Efficient I/O**: Batch artifact saving and JSON serialization

### Benchmarks (Typical Hardware)
- **Sequential**: ~50 strategies/minute on single core
- **Parallel (4 cores)**: ~180 strategies/minute  
- **Memory Usage**: ~100MB for 1M bars + 1000 strategies
- **Storage**: ~1MB artifacts per 100 strategies

## 🛡️ Risk Controls

### Position Sizing
- **Notional Cap**: Configurable maximum position size (default: 100% equity)
- **Minimum Trade Size**: Ensures meaningful position sizes
- **Decimal Precision**: Respects exchange quantity rules (8 decimals)
- **Conflict Resolution**: Handles competing constraints gracefully

### Validation
- **Minimum Trades**: Requires statistical significance (default: 50 trades)
- **Data Quality**: Robust handling of missing/invalid data
- **Parameter Bounds**: Enforced ranges prevent invalid configurations
- **Error Recovery**: Graceful handling of simulation failures

### Realism
- **Slippage**: 0.05% per side (realistic for liquid crypto pairs)
- **Fees**: 0.04% per side (typical exchange taker fees)  
- **Lookback Bias**: Proper indicator calculation without future data
- **Execution Logic**: Realistic fill conditions and timing

## 🔍 Out-of-Sample Validation

### Walk-Forward Analysis
The `--oos` mode implements rigorous out-of-sample validation:

1. **Window Creation**: Overlapping train/test periods across data history
2. **Strategy Optimization**: Find top strategies on training data  
3. **OOS Evaluation**: Test top strategies on unseen future data
4. **Aggregation**: Combine results across all test periods
5. **Ranking**: Rank strategies by aggregated OOS performance

### Benefits
- **Reduces Overfitting**: Tests on truly unseen data
- **Realistic Performance**: Better estimates of future returns
- **Robustness**: Identifies consistently performing strategies
- **Statistical Validity**: Multiple test periods increase confidence

## 📈 Example Results

### Basic Optimization
```bash
python strategy_finder.py --csv data.csv --budget 200 --seed 42

# Output:
PHASE 2 Professional Strategy Finder
Seed: 42 | Min trades: 50 | Max notional: 100.0%
Loaded 525600 rows (2023-01-01 to 2023-12-31)

Search complete: 157 valid strategies, 43 invalid (< 50 trades)
Top strategy: UID 1247, P&L: $2,847.33, Win Rate: 68.4%, Trades: 89
```

### Out-of-Sample Evaluation
```bash
python strategy_finder.py --csv data.csv --oos --train_days 90 --test_days 14

# Output:
OUT-OF-SAMPLE EVALUATION SUMMARY
============================================================
Number of Windows: 12
Strategies Tested: 60
Valid Strategies: 45

Aggregated OOS Metrics:
  Mean P&L: $1,245.67 ± $892.34
  Mean Win Rate: 62.3%
  Mean Drawdown: 8.7%

Best OOS Strategy:
  P&L: $3,102.45
  Win Rate: 71.2%
  Trades: 76
  Max Drawdown: 5.4%
```

## 🚨 Important Disclaimers

### Trading Risk Warning
- **Past Performance**: Backtested results do not guarantee future performance
- **Market Risk**: All trading involves substantial risk of loss
- **Paper Trading**: Recommend forward testing before live implementation
- **No Financial Advice**: This tool is for educational and research purposes only

### Technical Limitations  
- **Data Dependencies**: Results quality depends on input data accuracy
- **Market Regime**: Strategies may not work across all market conditions
- **Execution Differences**: Real trading may have different slippage/fees
- **Overfitting Risk**: Even with OOS validation, overfitting remains possible

## 📚 Development

### Contributing
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`  
7. Open Pull Request

### Code Quality
- **Testing**: Maintain >95% test coverage
- **Documentation**: Update README and docstrings
- **Style**: Follow PEP 8 guidelines (checked by flake8)
- **Type Hints**: Use type annotations for new functions

### Release Process
1. Update version in `__version__`
2. Update `CHANGELOG.md` with new features
3. Run full test suite including integration tests
4. Tag release: `git tag -a v2.0.0 -m "Release v2.0.0"`
5. Push tags: `git push origin --tags`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Pandas/NumPy**: Data processing and numerical computations
- **Matplotlib**: Visualization and charting
- **pytest**: Comprehensive testing framework
- **tqdm**: Progress bars for long-running operations

---

**⚠️ Remember**: This is a research tool. Always validate strategies through paper trading before risking real capital.