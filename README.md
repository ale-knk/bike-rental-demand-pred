# Bike Rental Demand Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Under%20Development-yellow.svg)
![Conda](https://img.shields.io/badge/conda-4.10.3-blue.svg?logo=conda&logoColor=white)
![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-1.4.2-brightgreen.svg?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-1.21.0-blue.svg?logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-1.9.0-brightgreen.svg?logo=pytorch&logoColor=white)
![Seaborn](https://img.shields.io/badge/seaborn-0.11.1-blue.svg?logo=seaborn&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-orange.svg?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)
![Git](https://img.shields.io/badge/git-F05032.svg?logo=git&logoColor=white)


## Introduction

Welcome to my professional portfolio project in the fields of Artificial Intelligence, Data Science and Software Engineering. This repository focuses on predicting bicycle rental demand in an urban setting. The goal is to establish a solid foundation for training predictive models that can anticipate bike rental needs.

The primary aim isn't to create the most advanced model possible—which would entail long training times and high computational costs—but to develop a versatile software interface. This interface is packaged as a Python package, featuring a suite of classes, functions, and command-line tools that simplify the model training process.

Considering the sequential nature of the bike rental data and the substantial advancements brought by Transformer-based architectures in AI, this project employs Transformer models to effectively predict rental demand.


## Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/bike-rental-demand-pred.git
    cd bike-rental-demand-pred
    ```

2. **Create a virtual environment**
    For example, using conda:

    ```bash
    conda create -n your_env_name python==3.12
    conda activate your_env_name
    ```

3. **Install the package**
    In order to install the package (and all the deppendencies):

    ```bash
    pip install .
    ```

## Usage

Overall, this package provides two primary tools: one for training models and another for making predictions with them.

Both tools require a configuration file to execute. Examples of these configuration files can be found in the `config_examples/` directory.

```bash
train --config_path path_to_your_config_file
predict --config_path path_to_your_config_file
```

For both tools, you must specify an output directory where various files generated during execution will be saved. Examples of these output files are available in the `output_examples/` directory.



## Implementation

Internally, the project is organized into several modules, each responsible for specific tasks:

-   **`pybike.preprocessing`**: Handles data preprocessing, including cleaning and transforming raw data to prepare it for model training.
-   **`pybike.trips`** & **`pybike.stations`**: Define interfaces using `pylance` to work efficiently with trip and station data, ensuring optimal data handling and manipulation.
-   **`pybike.dataloader`**: Responsible for preparing and initializing data loaders that feed data into the models during training and evaluation.
-   **`pybike.model`**: Defines the model architecture using `PyTorch`, implementing Transformer-based structures tailored for demand prediction.
-   **`pybike.train`**: Facilitates the training of models, managing the training loop, optimization, and model checkpointing.
-   **`pybike.predict`**: Enables inference with trained models, allowing users to generate predictions based on new input data.
-   **`pybike.tools`**: Contains the two command-line tools, `train` and `predict`, which provide a user-friendly interface for model training and prediction tasks.

Each module is designed with modularity and scalability in mind, ensuring that the codebase remains maintainable and extensible as the project evolves.



## Acquired Skills

This project encompasses a wide range of skills in both Data Science and Artificial Intelligence, as well as Software Engineering:

#### Data Science and Artificial Intelligence

-   **Data Preprocessing**: Cleaning and transforming data to prepare datasets suitable for model training.
-   **Predictive Modeling**: Designing and training Transformer-based models for predicting bike rental demand.
-   **Model Evaluation**: Implementing metrics and techniques to assess the performance of trained models.
-   **Time Series Handling**: Working with sequential data, considering the temporal nature of bike rental demand.

#### Software Engineering

-   **Object-Oriented Programming (OOP)**: Designing classes and modular structures that facilitate code extensibility and maintainability.
-   **Good Programming Practices**:
    -   **Linting and Formatting**: Utilizing tools like `ruff` to ensure code quality and maintain a consistent style.
    -   **Static Typing**: Implementing type annotations to enhance readability and ease maintenance, supported by `pylance`.
-   **Python Package Development**: Creating structured packages that simplify code distribution, installation, and reuse.
-   **Dependency Management**: Using virtual environments and package management tools to efficiently handle project dependencies.
-   **Task Automation**: Developing command-line tools to automate common processes such as model training and prediction generation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

-   **Your Name** - [rb.jandro@gmail.com](mailto:rb.jandro@gmail.com)
-   **GitHub**: [yourusername](https://github.com/yourusername)

