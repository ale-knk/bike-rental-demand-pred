___

## **Transformer-Based Model for Bike Demand Prediction**
___ 

This page introduces a model designed to predict bike-sharing routes. It begins by defining the mathematical representations of stations and bikes, followed by modeling the routes between stations using a weighted directed graph. Next, the problem of predicting the next route is formulated, and the model's architecture is described, incorporating embeddings, fully connected layers, and a Transformer encoder. Finally, the loss function used to train the model is detailed, and the complete architecture is summarized.

### **1. Representation of Stations and Bikes**

-   **Stations**: \( S = \{s_i\}_{i=1,...,n_s} \), where each station \( s_i \) is represented by its longitude and latitude coordinates \( (x^i, y^i) \).
-   **Bikes**: \( B = \{b_i\}_{i=1,...,n_b} \), where each bike \( b_i \) can be assigned to a specific station at any given time.

### **2. Routes Between Stations**

-   **Weighted Directed Graph**: A weighted directed graph \( G = (V, E) \) is used to model the routes between stations.
    -   **Vertices (V)**: Set of stations \( S \).
    -   **Edges (E)**: Each edge \( e_{ij} \in E \) represents a route from station \( s_i \) to station \( s_j \). Formally, each edge \( e_{ij} \) is denoted as:  
    \[ e_{ij} = (s_i, s_j, t_{ij}, t_{ij\_departure}, t_{ij\_arrival}) \]  
    Where:
        -   \( s_i \) and \( s_j \): Starting and destination stations.
        -   \( t_{ij} \): Travel time.
        -   \( t_{ij\_departure} \): Departure time.
        -   \( t_{ij\_arrival} \): Arrival time.
  
    -   **Weights (W)**: The edge weights are defined as \( w_{ij} = t_{ij\_arrival} - t_{ij\_departure} \).

### **3. Problem Definition for Prediction**

The goal is to predict the next route \( e_{ij} \) in a sequence of routes, given the previous routes. Formally, given a sequence of routes \(\tau = \{e_{i_1j_1}, e_{i_2j_2}, ..., e_{i_nj_n}\}\), the objective is to predict the next route \( e_{i_{n+1}j_{n+1}} \).

To predict the next route \( e_{i_{n+1}j_{n+1}} \), a supervised learning model is used. The target variable is the route \( e_{i_{n+1}j_{n+1}} \), and the model features are based on the sequence of routes \(\tau\) up to time \( n \).

The route \( e_{ij} \) is denoted as a random variable \( E \). Predicting the next route can be formulated as:

\[ P(E_{n+1} = e_{i_{n+1}j_{n+1}} \mid \tau) = P(E_{n+1} = e_{i_{n+1}j_{n+1}} \mid e_{i_1j_1}, e_{i_2j_2}, ..., e_{i_nj_n}) \]

Since \( e_{i_{n+1}j_{n+1}} = (s_{i_{n+1}}, s_{j_{n+1}}, t_{i_{n+1}}, t_{j_{n+1}}) \), we can express \( E \) as \( S_{departure} \times S_{arrival} \times T_{departure} \times T_{arrival} \), decomposing the prediction of the next route into four probability distributions:

-   \( P(S_{departure} = s_{i_{n+1}} \mid \tau_{n}) \): Prediction of the departure station \( s_{i_{n+1}} \) for the next route.
-   \( P(S_{arrival} = s_{j_{n+1}} \mid s_{i_{n+1}}, \tau_{n}) \): Prediction of the arrival station \( s_{j_{n+1}} \) for the next route.
-   \( P(T_{departure} = t_{i_{n+1}j_{n+1}\_departure} \mid s_{i_{n+1}}, s_{j_{n+1}}, \tau_{n}) \): Prediction of the departure time \( t_{i_{n+1}j_{n+1}\_departure} \) for the next route.
-   \( P(T_{arrival} = t_{i_{n+1}j_{n+1}\_arrival} \mid t_{i_{n+1}j_{n+1}\_departure}, s_{i_{n+1}}, s_{j_{n+1}}, \tau_{n}) \): Prediction of the arrival time \( t_{i_{n+1}j_{n+1}\_arrival} \) for the next route.

Notably, the predictions are sequentially added as variables influencing the conditional distribution of the subsequent probability distribution. Thus, the predicted arrival station, in addition to depending on the sequence of routes, also depends on the predicted departure station.

## **4. Model Architecture**

The model input is a sequence of routes \(\tau = \{e_{i_1j_1}, e_{i_2j_2}, ..., e_{i_nj_n}\}\), where each route \(e_{ij}\) is represented as a tuple \((s_i, s_j, t_{ij\_departure}, t_{ij\_arrival})\).


#### **Summary of the Model Components**

| **Component**                  | **Details**                                                                                                                                                       | **Formula**                                                                 |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Categorical Embeddings**     | Transform station identifiers into dense vectors.                                                                                                                | \( \mathbf{e}_s = E_s(s_i), \mathbf{e}_l = E_l(s_j) \)                      |
| **Continuous Embeddings**      | Transform times into dense vectors.                                                                                                                              | \( \mathbf{e}_{t_s} = E_{t_s}(t_{ij\_departure}), \mathbf{e}_{t_l} = E_{t_l}(t_{ij\_arrival}) \) |
| **Embedding Concatenation**    | Concatenate all embeddings into a single vector.                                                                                                                 | \( \mathbf{e}_{ij} = [\mathbf{e}_s, \mathbf{e}_l, \mathbf{e}_{t_s}, \mathbf{e}_{t_l}] \) |
| **Fully Connected Layer**      | Reduce dimensionality of the concatenated embeddings.                                                                                                            | \( \mathbf{h}_{ij} = \text{ReLU}(\mathbf{W}_h \mathbf{e}_{ij} + \mathbf{b}_h) \) |
| **Positional Encoding**        | Add positional encoding to capture sequence order.                                                                                                               | \( \mathbf{z}_{ij} = \mathbf{h}_{ij} + \mathbf{p}_{ij} \)                   |
| **Transformer Encoder**        | Process the embedded sequence to capture contextual relationships.                                                                                               | \( \mathbf{H} = \text{TransformerEncoder}(\{\mathbf{z}_{i_1j_1}, ..., \mathbf{z}_{i_nj_n}\}) \) |
| **Prediction (Departure)**     | Predict the departure station using the Transformer output.                                                                                                      | \( \mathbf{p}_{s_{i_{n+1}}} = \text{Softmax}(\mathbf{W}_{s_i} \mathbf{H} + \mathbf{b}_{s_i}) \) |
| **Prediction (Arrival)**       | Predict the arrival station using the Transformer output and the predicted departure station.                                                                    | \( \mathbf{p}_{s_{j_{n+1}}} = \text{Softmax}(\mathbf{W}_{s_j} [\mathbf{H}, \mathbf{p}_{s_{i_{n+1}}}] + \mathbf{b}_{s_j}) \) |
| **Prediction (Departure Time)**| Predict departure time using the Transformer output and the predicted departure and arrival stations.                                                            | \( \mathbf{p}_{t_{i_{n+1}j_{n+1}\_departure}} = \text{ReLU}(\mathbf{W}_{t_s} [\mathbf{H}, \mathbf{p}_{s_{i_{n+1}}}, \mathbf{p}_{s_{j_{n+1}}}] + \mathbf{b}_{t_s}) \) |
| **Prediction (Arrival Time)**  | Predict arrival time using the Transformer output and the predicted departure/arrival stations and departure time.                                                | \( \mathbf{p}_{t_{i_{n+1}j_{n+1}\_arrival}} = \text{ReLU}(\mathbf{W}_{t_l} [\mathbf{H}, \mathbf{p}_{s_{i_{n+1}}}, \mathbf{p}_{s_{j_{n+1}}}, \mathbf{p}_{t_{i_{n+1}j_{n+1}\_departure}}] + \mathbf{b}_{t_l}) \) |

#### **Flow Overview**

1. Input a sequence of routes \(\tau = \{e_{i_1j_1}, ..., e_{i_nj_n}\}\).
2. Transform categorical variables (stations) into embeddings \( \mathbf{e}_s, \mathbf{e}_l \).
3. Transform continuous variables (times) into embeddings \( \mathbf{e}_{t_s}, \mathbf{e}_{t_l} \).
4. Concatenate embeddings into a single vector \( \mathbf{e}_{ij} \).
5. Reduce dimensionality using a fully connected layer: \( \mathbf{h}_{ij} \).
6. Add positional encoding to \( \mathbf{h}_{ij} \) to form \( \mathbf{z}_{ij} \).
7. Process sequence \(\{\mathbf{z}_{i_1j_1}, ..., \mathbf{z}_{i_nj_n}\}\) using a Transformer encoder to output \( \mathbf{H} \).
8. Sequentially predict:
   -   Departure station \(s_{i_{n+1}}\).
   -   Arrival station \(s_{j_{n+1}}\) using the predicted \(s_{i_{n+1}}\).
   -   Departure time \(t_{i_{n+1}j_{n+1}\_departure}\).
   -   Arrival time \(t_{i_{n+1}j_{n+1}\_arrival}\).

This structured approach ensures clarity and readability, focusing on each component of the model and its role in prediction.


### **5. Loss Function**

To train the model, we use a combination of loss functions:

-   **Categorical Cross-Entropy (CE)** for departure and arrival stations.
-   **Mean Squared Error (MSE)** for departure and arrival times.

The total loss function is defined as:
\[ \mathcal{L} = \mathcal{L}_{\text{CE}}(\mathbf{p}_{s_{i_{n+1}}}, s_{i_{n+1}}) + \mathcal{L}_{\text{CE}}(\mathbf{p}_{s_{j_{n+1}}}, s_{j_{n+1}}) + \mathcal{L}_{\text{MSE}}(\mathbf{p}_{t_{i_{n+1}j_{n+1}\_salida}}, t_{i_{n+1}j_{n+1}\_salida}) + \mathcal{L}_{\text{MSE}}(\mathbf{p}_{t_{i_{n+1}j_{n+1}\_llegada}}, t_{i_{n+1}j_{n+1}\_llegada}) \]


### **6. Model Summary**

| **Step**                       | **Description**                                                                 |
|--------------------------------|---------------------------------------------------------------------------------|
| **1. Inputs**                  | Input sequence of routes \(\tau\).                                              |
| **2. Embeddings**              | Convert stations and times into dense vectors.                                  |
| **3. Concatenation**           | Concatenate embeddings into a single vector.                                    |
| **4. Dimensionality Reduction**| Apply a fully connected layer to reduce dimensionality.                         |
| **5. Positional Encoding**     | Add positional encoding to capture sequence order.                              |
| **6. Transformer Encoder**     | Process the sequence to capture contextual relationships.                       |
| **7. Predictions**             | Sequentially predict departure station, arrival station, departure time, and arrival time. |
| **8. Loss Function**           | Combine Categorical Cross-Entropy for stations and MSE for times.               |
