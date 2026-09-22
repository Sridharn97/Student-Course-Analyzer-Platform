"""App configuration and optional dependencies."""

PAGE_OPTIONS = [
    "Main Dashboard",
    "ML Models Comparison",
    "At-Risk Detection",
    "Course Recommendations",
    "Report Generator",
]

try:
    import xgboost as xgb  # noqa: F401
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    from sklearn.neural_network import MLPRegressor  # noqa: F401
    NEURAL_NET_AVAILABLE = True
except ImportError:
    NEURAL_NET_AVAILABLE = False
    MLPRegressor = None
