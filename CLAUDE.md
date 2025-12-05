# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BOCPD (Bayesian Online Changepoint Detection) is a Python library for real-time changepoint detection in time series data. It implements the Adams & MacKay (2007) algorithm using Bayesian conjugate models with analytical updates.

## Development Commands

### Running the Demo
```bash
uv run python main.py
```
Executes a demonstration on synthetic data with changepoints at t=100 and t=200.

### Code Quality
```bash
uv run black bocpd/        # Format code
uv run ruff check bocpd/   # Lint code
uv run mypy bocpd/         # Type checking
```

### Testing
```bash
uv run pytest tests/              # Run all tests
uv run pytest tests/test_foo.py   # Run specific test file
uv run pytest tests/ -v           # Verbose output
```

### Installing Dependencies
```bash
uv add <package>                  # Add runtime dependency
uv add --optional viz <package>   # Add to viz optional group
uv add --optional dev <package>   # Add to dev optional group
```

## Architecture

### Three-Layer Design

The library is structured in three layers, each with abstract base classes for extensibility:

**1. Model Layer (`bocpd/models/`)**
- `PredictiveModel` (ABC): Defines interface for Bayesian conjugate models
- `GaussianModel`: Implements Normal-Inverse-Gamma conjugate prior
- Key principle: **Immutability** - `update()` returns new instances, never modifies in-place
- This allows each run length to maintain independent model states

**2. Hazard Function Layer (`bocpd/hazards/`)**
- `HazardFunction` (ABC): Models changepoint occurrence probability
- `ConstantHazard`: Constant hazard rate (exponential distribution of run lengths)
- Future: `GeometricHazard`, `DiscreteHazard`

**3. Detector Layer (`bocpd/detector.py`)**
- `BOCPD`: Core algorithm maintaining run length distribution
- Internal state: `run_length_dist` (log probabilities), `models` (one per run length)
- Memory efficiency: `max_run_length` parameter truncates distribution

### Key Algorithm Details

**BOCPD Update Process:**
1. Compute predictive probability for each run length hypothesis
2. Calculate growth probabilities (no changepoint: r → r+1)
3. Calculate changepoint probability (r → 0)
4. Combine and normalize using Bayes' theorem
5. Update each run length's model with new observation

**Numerical Stability:**
- All probability computations in log space
- Uses `log_sum_exp` trick to prevent overflow/underflow
- Critical: Save `prev_run_length_dist` before updating for correct prediction probability calculation

### Changepoint Detection Strategy

The most reliable detection approach is:
```python
run_length = result["most_likely_run_length"]
prob = result["run_length_dist"][run_length]
if run_length == 1 and prob > 0.5:
    # Changepoint detected
```

**Why not `run_length == 0`?**
- Changepoints manifest as high probability at r=1 (one step after occurrence)
- P(r=0) is always equal to the hazard rate by construction
- Looking for r=1 with high probability is more robust

## Extending the Library

### Adding a New Conjugate Model

1. Create `bocpd/models/your_model.py`
2. Inherit from `PredictiveModel`
3. Implement required methods:
   - `fit_empirical(data)`: Estimate hyperparameters from data
   - `predict(x)`: Return log probability of predictive distribution
   - `update(x)`: Return NEW instance with updated hyperparameters
   - `copy()`, `get_params()`, `set_params()`
4. Add to `bocpd/models/__init__.py`

**Critical:** `update()` must return a new instance (immutability principle).

### Adding a New Hazard Function

1. Create `bocpd/hazards/your_hazard.py`
2. Inherit from `HazardFunction`
3. Implement `compute(r)` returning hazard probability at run length r
4. Optionally override `compute_log()`, `compute_log_survival()` for efficiency
5. Add to `bocpd/hazards/__init__.py`

## Important Implementation Notes

### Model Immutability
Models must be immutable. The `update()` method creates a new instance because BOCPD maintains one model per run length hypothesis. Modifying in place would corrupt the algorithm.

### Empirical Bayes Initialization
`fit_empirical()` sets weakly informative priors based on sample statistics:
- For `GaussianModel`: Uses sample mean/variance with kappa0=1, alpha0=2
- Purpose: Provides reasonable starting hyperparameters without strong assumptions

### Run Length Distribution
- Stored in log space (`run_length_dist`)
- Index i represents run length i
- P(r=0) is changepoint probability (always equals hazard rate)
- Most likely run length: `argmax(run_length_dist)`

## Common Pitfalls

1. **Don't use `run_length_dist` directly after update**
   - Save `prev_run_length_dist` before computing new distribution
   - Needed for accurate prediction log probability

2. **Changepoint detection threshold**
   - Simple `changepoint_prob > 0.5` often fails
   - Better: Check if `most_likely_run_length == 1` with high probability

3. **Hazard rate interpretation**
   - `lambda_ = 0.01` means expected run length of 100, not 1% detection rate
   - Expected run length = 1 / lambda_

4. **Memory management**
   - Set `max_run_length` for long-running detection
   - Without it, memory grows linearly with time
