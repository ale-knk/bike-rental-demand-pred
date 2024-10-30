___

## **Modelo para la predicción de trayectos**
___ 

En esta página, se presenta un modelo para la predicción de trayectos de bicicletas compartidas. Primero, se definen las representaciones matemáticas de estaciones y bicicletas, seguidas por la modelación de trayectos entre estaciones utilizando un grafo dirigido ponderado. Luego, se formula el problema de predicción del próximo trayecto y se describe la arquitectura del modelo que incluye el uso de embeddings, capas totalmente conectadas, y un Transformer encoder. Finalmente, se detalla la función de pérdida utilizada para entrenar el modelo y se resume la arquitectura completa.

#### **1. Representación de Estaciones y Bicicletas**

- **Estaciones**: \( S = \{s_i\}_{i=1,...,n_s} \), donde cada estación \( s_i \) está representada por sus coordenadas de longitud y latitud \( (x^i, y^i) \).
- **Bicicletas**: \( B = \{b_i\}_{i=1,...,n_b} \), donde cada bicicleta \( b_i \) puede estar asignada a una estación específica en un momento dado.

#### **2. Trayectos entre Estaciones**

- **Grafo Dirigido Ponderado**: Utilizamos un grafo dirigido ponderado \( G = (V, E) \) para modelar los trayectos entre estaciones.
  - **Vértices (V)**: Conjunto de estaciones \( S \).
  - **Aristas (E)**: Cada arista \( e_{ij} \in E \) representa un trayecto desde la estación \( s_i \) a la estación \( s_j \). Formalmente, cada arista \( e_{ij} \) se denota como: \[ e_{ij} = (s_i, s_j, t_{ij}, t_{ij\_salida}, t_{ij\_llegada}) \] 
  Donde
    - \( s_i \) y \( s_j \): Estaciones de inicio y destino.
    - \( t_{ij} \): Tiempo de trayecto.
    - \( t_{ij\_salida} \): Tiempo de salida.
    - \( t_{ij\_llegada} \): Tiempo de llegada.    
  
  - **Pesos (W)**: Los pesos de las aristas están definidos por $w_{ij} = t_{ij\_llegada} - t_{ij\_salida}$ 


### **3. Definición del Problema de Predicción**

El objetivo es predecir el próximo trayecto \(e_{ij}\) en una secuencia de trayectos dados los trayectos anteriores. Formalmente, dado una secuencia de trayectos \(\tau = \{e_{i_1j_1}, e_{i_2j_2}, ..., e_{i_nj_n}\}\), queremos predecir el trayecto \(e_{i_{n+1}j_{n+1}}\).

Para predecir el próximo trayecto \(e_{i_{n+1}j_{n+1}}\), utilizamos un modelo de aprendizaje supervisado. La variable objetivo es el trayecto \(e_{i_{n+1}j_{n+1}}\) y las características del modelo están basadas en la secuencia de trayectos \(\tau\) hasta el momento \(n\).

Denotamos el trayecto \(e_{ij}\) como una variable aleatoria \(E\). La predicción del próximo trayecto puede ser formulada como:

\[ P(E_{n+1} = e_{i_{n+1}j_{n+1}} \mid \tau) = P(E_{n+1} = e_{i_{n+1}j_{n+1}} \mid e_{i_1j_1}, e_{i_2j_2}, ..., e_{i_nj_n}) \]

A su vez, como $e_{i_{n+1}j_{n+1}} = (s_{i_{n+1}},s_{j_{n+1}},t_{i_{n+1}},t_{j_{n+1}})$ podemos poner $E = S_{salida} \times S_{llegada} \times T_{salida} \times T_{llegada}$ y así descomponer la predicción del siguiente trayecto en cuatro distribuciones de probabilidad:

- $P( S_{salida} = s_{i_{n+1}} \mid \tau_{n})$: Predicción de la estación de salida \( s_{i_{n+1}} \) del siguiente trayecto.
- $P(S_{llegada} = s_{j_{n+1}} \mid s_{i_{n+1}}, \tau_{n})$: Predicción de la estación de llegada \( s_{i_{n+1}} \) del siguiente trayecto.
- $P(T_{salida} = t_{i_{n+1}j_{n+1}\_salida} \mid s_{i_{n+1}}, s_{j_{n+1}}, \tau_{n})$: Predicción del tiempo de salida \( t_{i_{n+1}j_{n+1}\_salida} \) del siguiente trayecto.
- $P( T_{llegada} = t_{i_{n+1}j_{n+1}\_llegada} \mid t_{i_{n+1}j_{n+1}\_salida}, s_{i_{n+1}}, s_{j_{n+1}}, \tau_{n})$: Predicción del tiempo de llegada \( t_{i_{n+1}j_{n+1}\_llegada} \) del siguiente trayecto.

Observamos que las predicciones se van añadiendo como variables que afectan a la distribución condicional de la siguiente distribución de probabilidad. Así, la estación de llegada predicha, además de depender de la secuencia de trayectos, también depende de la estación de salida predicha.

## **4. Arquitectura del Modelo**

Dado que el input al modelo será una secuencia de trayectos \(\tau = \{e_{i_1j_1}, e_{i_2j_2}, ..., e_{i_nj_n}\}\), cada trayecto \(e_{ij}\) se representa como una tupla \((s_i, s_j, t_{ij\_salida}, t_{ij\_llegada})\).

#### **Embeddings de Variables Categóricas**

- **Estación de Salida**: Representada por \(s_i\), se pasa por un embedding \(E_s\).
- **Estación de Llegada**: Representada por \(s_j\), se pasa por un embedding \(E_l\).

Las estaciones se transforman en vectores densos mediante las siguientes funciones:
\[ \mathbf{e}_s = E_s(s_i) 
 , \mathbf{e}_l = E_l(s_j) \]

#### **Embeddings de Variables Continuas**

- **Tiempo de Salida**: Representado por \(t_{ij\_salida}\), se pasa por un embedding \(E_{t_s}\).
- **Tiempo de Llegada**: Representado por \(t_{ij\_llegada}\), se pasa por un embedding \(E_{t_l}\)

Los tiempos se transforman en vectores densos mediante las siguientes funciones:
\[ \mathbf{e}_{t_s} = E_{t_s}(t_{ij\_salida}), \mathbf{e}_{t_l} = E_{t_l}(t_{ij\_llegada}) \]

#### **Concatenación de Embeddings**

Una vez obtenidos los embeddings de cada componente del trayecto, se concatenan:
\[ \mathbf{e}_{ij} = [\mathbf{e}_s, \mathbf{e}_l, \mathbf{e}_{t_s}, \mathbf{e}_{t_l}] \]

#### **Fully Connected Layer para Reducir Dimensionalidad**

La concatenación de los embeddings se pasa por una capa totalmente conectada para reducir su dimensionalidad:
\[ \mathbf{h}_{ij} = \text{ReLU}(\mathbf{W}_h \mathbf{e}_{ij} + \mathbf{b}_h) \]

donde \(\mathbf{W}_h\) y \(\mathbf{b}_h\) son los pesos y bias de la capa totalmente conectada, respectivamente.

#### **Positional Encoding**

Al vector de salida reducido \(\mathbf{h}_{ij}\) se le añade un encoding posicional \(\mathbf{p}_{ij}\) para incorporar información de la posición en la secuencia:
\[ \mathbf{z}_{ij} = \mathbf{h}_{ij} + \mathbf{p}_{ij} \]

#### **Transformer Encoder**

La secuencia de trayectos embedded \(\{\mathbf{z}_{i_1j_1}, \mathbf{z}_{i_2j_2}, ..., \mathbf{z}_{i_nj_n}\}\) se pasa por un Transformer encoder:

\[ \mathbf{H} = \text{TransformerEncoder}(\{\mathbf{z}_{i_1j_1}, \mathbf{z}_{i_2j_2}, ..., \mathbf{z}_{i_nj_n}\}) \]

#### **Predicción de la Estación de Salida**
El output del Transformer encoder \(\mathbf{H}\) se pasa por una capa totalmente conectada para predecir la estación de salida \(s_{i_{n+1}}\):

\[ \mathbf{p}_{s_{i_{n+1}}} = \text{Softmax}(\mathbf{W}_{s_i} \mathbf{H} + \mathbf{b}_{s_i}) \]

#### **Predicción de la Estación de Llegada**
El output del Transformer encoder \(\mathbf{H}\) junto con la predicción de la estación de salida \(s_{i_{n+1}}\) se pasa por una capa totalmente conectada para predecir la estación de llegada \(s_{j_{n+1}}\):

\[ \mathbf{p}_{s_{j_{n+1}}} = \text{Softmax}(\mathbf{W}_{s_j} [\mathbf{H}, \mathbf{p}_{s_{i_{n+1}}}] + \mathbf{b}_{s_j}) \]

#### **Predicción del Tiempo de Salida**
El output del Transformer encoder \(\mathbf{H}\) junto con las predicciones de las estaciones de salida y llegada \(s_{i_{n+1}}\) y \(s_{j_{n+1}}\) se pasa por una capa totalmente conectada para predecir el tiempo de salida \(t_{i_{n+1}j_{n+1}\_salida}\):

\[ \mathbf{p}_{t_{i_{n+1}j_{n+1}\_salida}} = \text{ReLU}(\mathbf{W}_{t_s} [\mathbf{H}, \mathbf{p}_{s_{i_{n+1}}}, \mathbf{p}_{s_{j_{n+1}}}] + \mathbf{b}_{t_s}) \]

#### **Predicción del Tiempo de Llegada**
El output del Transformer encoder \(\mathbf{H}\) junto con las predicciones de las estaciones de salida y llegada \(s_{i_{n+1}}\) y \(s_{j_{n+1}}\), y el tiempo de salida \(t_{i_{n+1}j_{n+1}\_salida}\) se pasa por una capa totalmente conectada para predecir el tiempo de llegada \(t_{i_{n+1}j_{n+1}\_llegada}\):

\[ \mathbf{p}_{t_{i_{n+1}j_{n+1}\_llegada}} = \text{ReLU}(\mathbf{W}_{t_l} [\mathbf{H}, \mathbf{p}_{s_{i_{n+1}}}, \mathbf{p}_{s_{j_{n+1}}}, \mathbf{p}_{t_{i_{n+1}j_{n+1}\_salida}}] + \mathbf{b}_{t_l}) \]

### **5. Función de Pérdida**

Para entrenar el modelo, utilizamos una combinación de pérdidas:

- **Categorical Cross-Entropy** para las estaciones de salida y llegada.
- **Mean Squared Error (MSE)** para los tiempos de salida y llegada.

La función de pérdida total es:
\[ \mathcal{L} = \mathcal{L}_{\text{CE}}(\mathbf{p}_{s_{i_{n+1}}}, s_{i_{n+1}}) + \mathcal{L}_{\text{CE}}(\mathbf{p}_{s_{j_{n+1}}}, s_{j_{n+1}}) + \mathcal{L}_{\text{MSE}}(\mathbf{p}_{t_{i_{n+1}j_{n+1}\_salida}}, t_{i_{n+1}j_{n+1}\_salida}) + \mathcal{L}_{\text{MSE}}(\mathbf{p}_{t_{i_{n+1}j_{n+1}\_llegada}}, t_{i_{n+1}j_{n+1}\_llegada}) \]

### **6. Resumen del Modelo**

1. **Inputs**: Secuencia de trayectos \(\tau\).
2. **Embeddings**: Convertir estaciones y tiempos a vectores densos.
3. **Concatenación**: Concatenar embeddings.
4. **Reducción de Dimensionalidad**: Capa totalmente conectada.
5. **Positional Encoding**: Añadir encoding posicional.
6. **Transformer Encoder**: Procesar secuencia.
7. **Predicciones**: Estación de salida, estación de llegada, tiempo de salida, tiempo de llegada.
8. **Pérdida**: Combinación de Categorical Cross-Entropy y MSE.


