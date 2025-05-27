from setuptools import setup, find_packages

setup(
    name="algo_trading",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.2',
        'scikit-learn>=0.24.2',
        'plotly>=5.1.0',
        'joblib>=1.1.0',
        
        # Market regime detection
        'hmmlearn>=0.2.7',
        'ruptures>=1.1.5',
        'arviz[all]>=0.12.0',
        
        # API and utilities
        'smartapi-python>=1.3.1',
        'websocket-client>=1.2.1',
        'pyotp>=2.6.0',
        'python-dotenv>=0.19.0',
        'requests>=2.26.0',
        'aiohttp>=3.8.1',
        
        # Technical analysis
        'ta>=0.9.0',
        'scipy>=1.7.0',
        'statsmodels>=0.13.0',
    ],
) 