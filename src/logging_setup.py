import logging
import mlflow
from packaging.version import Version
from .tracking import init_tracking

logger = logging.getLogger("prompt_xpert")
logging.basicConfig(level=logging.INFO)

def configure_logging_and_mlflow():
    try:
        # Initialize tracking (DagsHub or local) before enabling autologging
        init_tracking()
        if Version(mlflow.__version__) < Version("2.18.0"):
            logger.warning("MLflow >= 2.18.0 is recommended for DSPy tracing support.")
        try:
            mlflow.dspy.autolog(
                log_compiles=True,
                log_evals=True,
                log_traces_from_compile=True,
                log_traces_from_eval=True,
            )
            logger.info("Enabled mlflow.dspy.autolog.")
        except Exception as _e:
            logger.warning(f"Failed to enable mlflow.dspy.autolog: {_e}.")
    except Exception as _e:
        logger.warning(f"Could not check MLflow version or enable dspy autolog: {_e}")
    return logger
