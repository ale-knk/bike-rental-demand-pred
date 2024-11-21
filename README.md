# Bike Rental Demand Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Status](https://img.shields.io/badge/status-Under%20Development-yellow.svg)

En este repositorio se presenta un proyecto para mi portfolio profesional en el ámbito de la IA y Ciencia de Datos. Dado un sistema de alquiler de bicicletas en una ciudad, se pretende brindar unas bases para entrenar modelos de predicción de demanda.

Lejos de pretender entrenar el mejor modelo posible (lo que requeriría un gran tiempo de entrenamiento, recursos computacionales, etc), el objetivo principal es el de desarrollar una interfaz de software con la que poder entrenar a estos modelos. Esta interfaz se presenta en la forma de un paquete de python, en donde se definen distintas clases y funciones así como command line tools para entrenar los modelos.

Debido a la naturaleza secuencial de los datos (así como del gran impacto que ha tenido este tipo de arquitectura en el ámbito de la IA), se ha apostado por la creación de modelos Transformer-Based para la predicción de demanda.

 

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
Desde un punto de vista general, este paquete define dos herramientas principales: una para entrenar modelos y otra para hacer inferencia con ellos.

Ambas herramientas necesitan de un archivo de configuración para ser ejecutadas. Ejemplos de estos archivos de configuración se encuentran en `config_examples/`.


```
train --config_path path_to_your_config_file
predict -- config_path path_to_your_config_file
```

En ambas herramientas, se debe especificar un directorio output en el que serán guardados distintos archivos como resultado de su ejecutación. Ejemplos de estos archivos de salida se encuentran en `output_examples`.

## Implementation

Internamente, hay definidos distintos módulos para llevar a cabo distintas tareas:

- `pybike.preprocessing`: Encargado del preprocesamiento de los datos.
- `pybike.trips`& `pybike.stations`: Definen interfaces utilizando `pylance`para trabajar óptimamente con los datos.
- `pybike.dataloader`: Encargado de preparar e inicializar los dataloaders para el entrenamiento de modelos.
- `pybike.model`: Define la arquitectura del modelo utilizando `pytorch`.
- `pybike.train`: Para el entrenamiento de modelos.
- `pybike.predict`: Para la inferencia con modelos ya entrenados.
- `pybike.tools`: Se definen las dos command line tools `train`y `predict`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

-   **Your Name** - [rb.jandro@gmail.com](mailto:rb.jandro@gmail.com)
-   **GitHub**: [yourusername](https://github.com/yourusername)

