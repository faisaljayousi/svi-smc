import functools
import logging
import time
from datetime import datetime

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def diagnostic_report(name: str):
    """
    Decorator to wrap diagnostic functions.
    Handles timing, logging, and automatic plot saving.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"--- Starting Diagnostic: {name} ---")
            start_time = time.perf_counter()

            # Execute the diagnostic function
            # We assume diagnostic functions return the plt.Figure object
            fig = func(*args, **kwargs)

            end_time = time.perf_counter()
            duration = end_time - start_time

            # Metadata for saving
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"reports/figures/{name.lower().replace(' ', '_')}_{timestamp}.png"
            )

            if isinstance(fig, plt.Figure):
                fig.savefig(filename, dpi=300, bbox_inches="tight")
                plt.close(fig)  # Prevent memory leaks in large loops
                logger.info(f"Successfully saved diagnostic plot to {filename}")

            logger.info(f"Finished {name} in {duration:.2f} seconds.")
            return fig

        return wrapper

    return decorator
