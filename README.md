# Stock-Price-Prediction Using LSTM on ICICI Bank Data

  This project demonstrates the use of Long Short-Term Memory (LSTM) networks to predict the stock prices of ICICI Bank using historical stock data. The model is built using Keras with TensorFlow backend and makes use of scikit-learn for preprocessing and evaluation.

* **Project Overview**
 
In this project, we used historical stock data from ICICI Bank to predict future closing prices based on features such as OPEN, HIGH, LOW, CLOSE, and PREVCLOSE. The model uses LSTM, a type of recurrent neural network (RNN), to learn temporal dependencies in the data.
 

**Requirements
Software Dependencies**
To run this project, the following Python libraries are required:

  * Pandas: Data manipulation and analysis.
  * NumPy: Scientific computing with arrays and matrices.
  * Matplotlib: Visualization of data and predictions.
  * scikit-learn: For preprocessing and evaluation metrics.
  * Keras: For building the LSTM model.
  * TensorFlow: Backend for Keras.

* Steps:
**Data Preprocessing**:

  * Dataset Loading: Read the dataset and filter it based on the stock symbol 
    (HDFCBANK) and series (EQ).
  * Datetime Parsing: Convert the TIMESTAMP column to datetime and set it as 
    the index.
  * Drop Unnecessary Columns: Drop columns like SYMBOL, SERIES, and ISIN 
    which are not needed for the analysis.
    
**Feature Selection**:

  * Correlation Analysis: Use a correlation heatmap to visualize the 
    relationships between numerical features like OPEN, CLOSE, HIGH, LOW, 
    PRICE_CHANGE etc.
  * Feature Selection: Select relevant features for model input.
**Data Scaling**:

  * MinMaxScaler: Apply MinMax scaling to scale the features between 0 and 1 
    This is crucial for LSTM models to handle numerical data properly.
  * Train-Test Split: Split the data into training and testing sets, using 
    the scaled data.
**Sequence Creation**:

  * Window-based Input: Create time-sequenced data (X_train, y_train, 
    X_test, y_test) based on a window_size (60 in this case). This helps the 
    LSTM model to capture temporal dependencies.
**Model Construction**:

  * LSTM Layers: Construct an LSTM model with two LSTM layers. The first LSTM 
    layer is return_sequences=True to output sequences to the next layer. The 
    second LSTM layer is return_sequences=False as it's the last LSTM layer.
  * Dropout Layer: Add a Dropout layer with a rate of 0.4 to reduce 
     overfitting.
  * Dense Layer: The output layer is a Dense layer with 1 unit to predict a       single value (price).
  * Kernel Initialization: Use HeNormal() for kernel initialization to 
    improve the model's convergence.
    
**Model Compilation**:

  * Optimizer: Compile the model using the Adam optimizer.
  * Loss Function: Use mean_squared_error as the loss function, which is 
    typical for regression tasks.
    
**Model Training**:

  * Epochs & Batch Size: Train the model for a defined number of epochs(100) 
    and use a batch size of 16.
  * Validation Data: Validate the model on the test data during training to 
    monitor its performance.
    
**Model Evaluation**:

  * Predictions: Predict stock prices on both the training and testing data.
  * Inverse Scaling: Use the inverse_transform method of the scaler to 
     convert the predicted values back to the original scale.
  * Error Metrics: Compute evaluation metrics like RMSE (Root Mean Squared 
    Error), MAE (Mean Absolute Error), and R² score for both training and 
    test data.
    
**Future Price Prediction:**

**Sliding Window:** Use the last window_size days of data as input to 
    predict the future stock prices for the next 30 days.
**Future Predictions:** For each day, the model predicts the next price, and 
    the input is updated with the predicted price for the next iteration.
**Generate Future Dates:** Generate future dates based on the last date in 
    the test data and combine the past and predicted prices.
    
**Visualization:**

  **Combining Past and Future Data:** Combine the actual past prices with 
     the predicted future prices.
    
   **Plotting:** Optionally visualize the combined data on a plot to see 
      the trend of predicted prices.
  

**Conclusion:**

  * The LSTM model is used to predict future stock prices based on 
     historical data.
  * The model's performance is evaluated using metrics such as RMSE, MAE, 
     and R² score.
  * The future prices for the next 30 days are predicted, and the model can 
     be improved by fine-tuning the architecture and hyperparameters.




 
     
      
