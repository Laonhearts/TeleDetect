from setuptools import setup, find_packages

setup(

    name='deepfake_detector',
    
    version='0.1',
    
    packages=find_packages(),
    
    install_requires=[
    
        'tensorflow>=2.4.0',
        'requests',
        'numpy',
        'Pillow',
        'scikit-learn',
        'mtcnn',
    
    ],
    
    entry_points={
    
        'console_scripts': [
            'deepfake_detector=deepfake_detector.main:main',
    
        ],
    
    },

)
