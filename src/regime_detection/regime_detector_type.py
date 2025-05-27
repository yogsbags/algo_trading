from enum import Enum

class RegimeDetectorType(Enum):
    """Enum for different types of regime detectors"""
    KMEANS = "kmeans"
    HMM = "hmm"
    BAYESIAN = "bayesian"
    RUPTURES = "ruptures"

def get_detector(detector_type: RegimeDetectorType):
    """Get a detector instance based on type"""
    from .kmeans_regime_detector import KMeansRegimeDetector
    from .hmm_regime_detector import HMMRegimeDetector
    from .bayesian_regime_detector import BayesianRegimeDetector
    from .ruptures_regime_detector import RupturesRegimeDetector

    if detector_type == RegimeDetectorType.KMEANS:
        return KMeansRegimeDetector(n_regimes=4)
    elif detector_type == RegimeDetectorType.HMM:
        return HMMRegimeDetector(n_regimes=4)
    elif detector_type == RegimeDetectorType.BAYESIAN:
        return BayesianRegimeDetector(n_regimes=4)
    elif detector_type == RegimeDetectorType.RUPTURES:
        return RupturesRegimeDetector(penalty=10, min_size=20, method='dynp')
    else:
        raise ValueError(f"Unknown detector type: {detector_type}") 