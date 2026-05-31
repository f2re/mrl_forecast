"""Central project constants for radar reflectivity nowcasting."""

FORECAST_STEP_MINUTES = 15
DEFAULT_INPUT_LENGTH = 4
DEFAULT_TARGET_LENGTH = 4

SUPPORTED_TARGET_LENGTHS = {
    "1h": 4,
    "2h": 8,
    "3h": 12,
}

MIN_DBZ = 0.0
MAX_DBZ = 70.0

PRODUCT_NAME = "experimental_radar_reflectivity_nowcast"
NOT_OFFICIAL_WARNING = True
