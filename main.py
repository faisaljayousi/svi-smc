import logging
import os
from datetime import datetime

from src.config_loader import load_config
from src.diagnostics import (
    diagnostic_cost_surface,
    diagnostic_instability,
    diagnostic_vanna,
)

# Setup artifact paths
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports/logs", exist_ok=True)

# Setup Logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"reports/logs/session_{timestamp}.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting SVI Stability Diagnostics.")

    # Load Configuration
    cfg = load_config(os.path.join("config", "params.yaml"))

    # Run Diagnostics
    diagnostic_instability(cfg)
    diagnostic_vanna(cfg)
    diagnostic_cost_surface(cfg)

    logger.info("Stability diagnostics complete.")


if __name__ == "__main__":
    main()
