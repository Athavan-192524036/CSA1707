from setuptools import setup, find_packages

setup(
    name="lane-marking-detection-agent",
    version="1.0.0",
    description="AI Agent for Lane Marking Detection Under Adverse Weather Conditions",
    author="Product & Engineering Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
        "tensorboard>=2.8.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "tensorrt": ["tensorrt>=8.0.0", "pycuda>=2021.1"],
        "onnx": ["onnx>=1.10.0", "onnxruntime-gpu>=1.10.0", "onnx-simplifier>=0.3.0"],
        "ros2": ["rclpy>=3.0.0", "sensor-msgs", "geometry-msgs", "nav-msgs", "cv-bridge"],
        "dev": ["pytest>=6.0", "black>=21.0", "flake8>=3.9", "mypy>=0.9"],
    },
    entry_points={
        "console_scripts": [
            "lane-detect=scripts.inference:main",
            "lane-train=scripts.train:main",
            "lane-eval=scripts.evaluate:main",
        ],
    },
)