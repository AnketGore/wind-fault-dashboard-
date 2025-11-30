import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Flatten, LayerNormalization, Layer

# --- Self-Attention Layer ---
class TimeSeriesSelfAttention(Layer):
    def __init__(self, units, **kwargs):
        super(TimeSeriesSelfAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.query = Dense(self.units)
        self.key = Dense(self.units)
        self.value = Dense(self.units)
        self.combine_heads = Dense(self.units)
        self.layernorm = LayerNormalization()

    def call(self, inputs):
        Q = self.query(inputs)
        K = self.key(inputs)
        V = self.value(inputs)
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        scaled_scores = attention_scores / tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
        attention_weights = tf.nn.softmax(scaled_scores, axis=-1)
        context = tf.matmul(attention_weights, V)
        attention_output = self.combine_heads(context)
        return self.layernorm(attention_output + inputs)

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("T1.csv")
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d %m %Y %H:%M')
    df.set_index('Date/Time', inplace=True)
    df = df[['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']]
    return df.dropna()

# --- Feature Engineering ---
def add_features(df):
    df['Power_Residual'] = df['LV ActivePower (kW)'] - df['Theoretical_Power_Curve (KWh)']
    df['Power_Coefficient'] = df['LV ActivePower (kW)'] / (df['Theoretical_Power_Curve (KWh)'] + 1e-6)
    df['WindSpeed_Mean'] = df['Wind Speed (m/s)'].rolling(6).mean().bfill()
    df['WindSpeed_Std'] = df['Wind Speed (m/s)'].rolling(6).std().bfill()
    df['Delta_WindSpeed'] = df['Wind Speed (m/s)'].diff().fillna(0)
    df['Delta_ActivePower'] = df['LV ActivePower (kW)'].diff().fillna(0)
    df['Turbulence_Intensity'] = df['WindSpeed_Std'] / (df['WindSpeed_Mean'] + 1e-6)
    df['WindDir_sin'] = np.sin(np.deg2rad(df['Wind Direction (°)']))
    df['WindDir_cos'] = np.cos(np.deg2rad(df['Wind Direction (°)']))
    return df

# --- Sequence Creation ---
def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])
    return np.array(X), np.array(y)

# --- Model Builder ---
def build_model(seq_len, num_features):
    inputs = Input(shape=(seq_len, num_features))
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = LSTM(64, return_sequences=True)(x)
    x = TimeSeriesSelfAttention(64)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# --- Fault Explanation ---
def fault_explanation(residual, cp, turbulence):
    if residual < -500 and cp < 0.5:
        return "Low efficiency → possible gearbox or blade pitch fault."
    elif turbulence > 0.2:
        return "High turbulence → increased stress on blades."
    else:
        return "Operating conditions appear normal."

# --- Streamlit App ---
st.set_page_config(page_title="Wind Turbine Fault Dashboard", layout="wide")
st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Model Training", "Fault Detection"])

df = load_data()
df = add_features(df)

features = ['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Power_Residual', 'Power_Coefficient',
            'WindSpeed_Mean', 'WindSpeed_Std', 'Delta_WindSpeed', 'Delta_ActivePower',
            'Turbulence_Intensity', 'WindDir_sin', 'WindDir_cos']

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
X, y = create_sequences(scaled, seq_len=24)

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

if page == "Overview":
    st.title(" Wind Turbine Fault Detection")
    st.markdown("""
    This dashboard uses a CNN–LSTM–Self-Attention model to predict power output and detect faults in wind turbines.
    """)

elif page == "Model Training":
    st.title(" Model Training")
    epochs = st.sidebar.slider("Training Epochs", 1, 50, 5)
    with st.spinner("Training model..."):
        model = build_model(seq_len=24, num_features=X.shape[2])
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=0)
        y_pred = model.predict(X_test).flatten()

    y_pred_rescaled = scaler.inverse_transform(np.hstack([y_pred.reshape(-1,1), np.zeros((len(y_pred), X.shape[2]-1))]))[:, 0]
    y_true_rescaled = scaler.inverse_transform(np.hstack([y_test.reshape(-1,1), np.zeros((len(y_test), X.shape[2]-1))]))[:, 0]

    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))
    r2 = r2_score(y_true_rescaled, y_pred_rescaled)

    st.metric("MAE", f"{mae:.2f} kW")
    st.metric("RMSE", f"{rmse:.2f} kW")
    st.metric("R² Score", f"{r2:.4f}")

elif page == "Fault Detection":
    st.title("Fault Detection")
    threshold_factor = st.sidebar.slider("Fault Threshold (×MAE)", 1.0, 5.0, 2.0)

    with st.spinner("Training model..."):
        model = build_model(seq_len=24, num_features=X.shape[2])
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=0)
        y_pred = model.predict(X).flatten()

    y_pred_rescaled = scaler.inverse_transform(np.hstack([y_pred.reshape(-1,1), np.zeros((len(y_pred), X.shape[2]-1))]))[:, 0]
    y_true_rescaled = scaler.inverse_transform(np.hstack([y.reshape(-1,1), np.zeros((len(y), X.shape[2]-1))]))[:, 0]
    residuals = y_true_rescaled - y_pred_rescaled
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
    threshold = threshold_factor * mae
    fault_flags = np.abs(residuals) > threshold

    timestamps = df.index[24:]
    st.subheader(" Power Prediction vs Actual")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timestamps[-len(y_pred_rescaled):], y_true_rescaled[-len(y_pred_rescaled):], label="Actual Power", color='blue')
    ax.plot(timestamps[-len(y_pred_rescaled):], y_pred_rescaled, label="Predicted Power", color='orange')
    ax.fill_between(timestamps[-len(y_pred_rescaled):], y_true_rescaled[-len(y_pred_rescaled):], y_pred_rescaled,
                    where=fault_flags, color='red', alpha=0.3, label="Fault Zone")
    ax.legend()
    ax.set_ylabel("Power (kW)")
    st.pyplot(fig)

    st.subheader(" Detected Faults")
    fault_df = pd.DataFrame({
        "Timestamp": timestamps[-len(y_pred_rescaled):],
        "Actual Power": y_true_rescaled[-len(y
