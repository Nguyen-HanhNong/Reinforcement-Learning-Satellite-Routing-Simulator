# Autonomous Satellite Routing Simulator

## What is this project about?

This project is a simulator that uses machine learning and deep learning algorithms to predict the best route for a satellite to take in order to avoid space debris. The simulator uses a 3D model of the Earth made in PyQT and the space debris to calculate the best route for the satellite to take. The simulator also generates plots and statistics to show the performance of the machine learning and deep learning algorithms.

### Dependencies
- Python 3.9+ 
- **[PyQt5](https://pypi.org/project/PyQt5/)** for the GUI
- **[numpy](https://pypi.org/project/numpy/)** for math calculations  
- **[matplotlib](https://pypi.org/project/matplotlib/)** for 3D plotting 
- **[pandas](https://pypi.org/project/pandas/)** for data manipulation
- **[scikit-learn](https://pypi.org/project/scikit-learn/)** for machine learning algorithms
- **[TensorFlow](https://pypi.org/project/tensorflow/)** for deep learning algorithms

### Setup

Install the dependencies using the latest version of Pip:

```bash
pip install PyQt5 numpy matplotlib pandas scikit-learn tensorflow
```

### How to run the program

```bash
python model.py
```

### How to run the program that generates the plots and statistics

```bash
python getPerformanceMetrics.py
```

