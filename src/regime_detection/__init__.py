from enum import Enum
import logging

logger = logging.getLogger('regime_detection')

class RegimeDetectorType(Enum):
    KMEANS = "kmeans"
    HMM = "hmm"
    BAYESIAN = "bayesian"
    RUPTURES = "ruptures"
    ENSEMBLE = "ensemble"

def get_detector(detector_type, **kwargs):
    """
    Factory function to create a regime detector
    
    Args:
        detector_type: RegimeDetectorType enum
        **kwargs: Additional parameters for the specific detector
        
    Returns:
        Instance of a regime detector
    """
    if detector_type == RegimeDetectorType.KMEANS:
        from .kmeans_regime_detector import KMeansRegimeDetector
        return KMeansRegimeDetector(**kwargs)
    elif detector_type == RegimeDetectorType.HMM:
        from .hmm_regime_detector import HMMRegimeDetector
        return HMMRegimeDetector(**kwargs)
    elif detector_type == RegimeDetectorType.BAYESIAN:
        from .bayesian_regime_detector import BayesianRegimeDetector
        return BayesianRegimeDetector(**kwargs)
    elif detector_type == RegimeDetectorType.RUPTURES:
        from .ruptures_regime_detector import RupturesRegimeDetector
        return RupturesRegimeDetector(**kwargs)
    elif detector_type == RegimeDetectorType.ENSEMBLE:
        from .ensemble_regime_detector import EnsembleRegimeDetector
        return EnsembleRegimeDetector(**kwargs)
    else:
        logger.error(f"Unknown detector type: {detector_type}")
        raise ValueError(f"Unknown detector type: {detector_type}")

# Import all classes for direct import
from .kmeans_regime_detector import KMeansRegimeDetector
from .hmm_regime_detector import HMMRegimeDetector
from .bayesian_regime_detector import BayesianRegimeDetector
from .ruptures_regime_detector import RupturesRegimeDetector
from .ensemble_regime_detector import EnsembleRegimeDetector 