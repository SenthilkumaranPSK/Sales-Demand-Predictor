"""Application constants."""

# Confidence interval z-scores for normal distribution
# 50% CI: ±0.674 std, 80% CI: ±1.282 std, 90% CI: ±1.645 std
CONFIDENCE_Z_SCORES = {
    "50%": 0.674,
    "80%": 1.282,
    "90%": 1.645,
}

# Quantile conversion factors for Amazon Forecast
# Amazon Forecast provides p10, p50, p90 quantiles
# 50% CI: approximated as p50 ± 0.33 * (p90 - p10)
# 80% CI: directly p10 to p90
# 90% CI: approximated as p50 ± 0.82 * (p90 - p10)
QUANTILE_CONVERSION_FACTORS = {
    "ci_50_factor": 0.33,   # 50% CI width as fraction of 80% CI width
    "ci_90_factor": 0.82,   # 90% CI width as fraction of 80% CI width
}

# Forecast configuration defaults
DEFAULT_FORECAST_HORIZON_MAX = 90
DEFAULT_FORECAST_HORIZON_MIN = 1

# Model training defaults
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0

# AWS service timeouts (seconds)
S3_TIMEOUT = 30
FORECAST_API_TIMEOUT = 60
RDS_TIMEOUT = 10

# Rate limiting defaults
DEFAULT_RATE_LIMIT_PER_MINUTE = 1000
DEFAULT_MAX_CONCURRENT_REQUESTS = 100

# Health check thresholds
HEALTH_CHECK_TIMEOUT_MS = 1000
HEALTH_CHECK_DEGRADED_THRESHOLD_MS = 500

# Model artifact cache TTL (seconds)
MODEL_CACHE_TTL = 3600  # 1 hour

# API versioning
API_VERSION = "1.0.0"
API_V1_PREFIX = "/api/v1"