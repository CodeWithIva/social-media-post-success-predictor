import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys  # For flushing output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Keras Tuner Import ---
from keras_tuner import RandomSearch
# --------------------------

# Keras Components
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model

# --- STATIC HYPERPARAMETERS ---
VOCAB_SIZE = 1000
MAX_LENGTH = 20


# Note: EMBEDDING_DIM is now tuned by the Keras Tuner
# ----------------------------

# ==========================================
# 1. DATA GENERATION (Definition and Call)
# ==========================================
def generate_dummy_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'post_text': [
            "Amazing sunset! Best time of day. 🌅" if i % 2 == 0 else "Check out this quick link for details."
            for i in range(n_samples)
        ],
        'followers_count': np.random.randint(100, 10000, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'contains_video': np.random.choice([0, 1], n_samples),
    }

    # Logic to simulate engagement and conversion
    likes_list = []
    comments_list = []
    shares_list = []
    conversion_list = []

    for i in range(n_samples):
        score = data['followers_count'][i] / 100
        if data['contains_video'][i] == 1: score += 50
        if 18 <= data['hour_of_day'][i] <= 21: score += 30

        likes = int(score + np.random.normal(0, 10))
        likes_list.append(likes)

        comments_list.append(int(likes / np.random.randint(5, 15) + np.random.randint(1, 5)))
        shares_list.append(int(likes / np.random.randint(10, 25) + np.random.randint(0, 3)))

        conversion = np.clip(np.random.normal(loc=likes / 3000, scale=0.01), 0.001, 0.1)
        conversion_list.append(conversion)

    data['likes'] = likes_list
    data['comments'] = comments_list
    data['shares'] = shares_list
    data['conversion_rate'] = conversion_list

    return pd.DataFrame(data)


# Creates the DataFrame 'df'
df = generate_dummy_data()

# ==========================================
# 2. FEATURE ENGINEERING & TARGET DEFINITION
# ==========================================
df['text_length'] = df['post_text'].apply(len)
df['has_exclamation'] = df['post_text'].apply(lambda x: 1 if '!' in x else 0)
df['is_prime_time'] = df['hour_of_day'].apply(lambda x: 1 if 18 <= x <= 21 else 0)

# Define Thresholds
median_likes = df['likes'].median()
median_conversion = df['conversion_rate'].median()

# Define Success: Post must be above average in BOTH engagement AND conversion rate.
df['is_successful'] = (
        (df['likes'] > median_likes) &
        (df['conversion_rate'] > median_conversion)
).astype(int)

# Define the final target variable
y = df['is_successful']

# Define the comprehensive list of numerical input features
numerical_features = [
    'followers_count',
    'contains_video',
    'text_length',
    'has_exclamation',
    'is_prime_time',
    'likes',
    'comments',
    'shares',
    'conversion_rate'
]

# ==========================================
# 3. TEXT PREPROCESSING (Tokenization and Padding)
# ==========================================
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<unk>")
tokenizer.fit_on_texts(df['post_text'])
sequences = tokenizer.texts_to_sequences(df['post_text'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

# ==========================================
# 4. PREPARING DUAL INPUTS FOR TRAINING (Scaling and Splitting)
# ==========================================
X_numerical = df[numerical_features]

# Scale Numerical Data
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

# Splits all data into training and testing sets
X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    padded_sequences, X_numerical_scaled, y, test_size=0.2, random_state=42
)


# ==========================================
# 5. BUILDING THE DYNAMIC KERAS MODEL FUNCTION
# ==========================================
def build_model(hp):
    # Hyperparameter choices for dense layers
    hp_units = hp.Int('units', min_value=16, max_value=64, step=16)
    hp_embed_dim = hp.Int('embedding_dim', min_value=16, max_value=64, step=16)
    hp_dropout = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)

    # --- 1. TEXT INPUT PATH ---
    text_input = Input(shape=(MAX_LENGTH,), name='text_input')

    # Use the tunable embedding dimension
    x = Embedding(input_dim=VOCAB_SIZE, output_dim=hp_embed_dim, input_length=MAX_LENGTH)(text_input)
    x = Flatten()(x)
    text_output = Dense(hp_units, activation='relu')(x)

    # --- 2. NUMERICAL INPUT PATH ---
    num_input = Input(shape=(len(numerical_features),), name='numerical_input')
    num_output = Dense(hp_units, activation='relu')(num_input)

    # --- 3. COMBINE & CLASSIFY ---
    combined = Concatenate()([text_output, num_output])

    # Use the tunable units for the final layer
    z = Dense(hp_units, activation='relu')(combined)

    # Use the tunable dropout rate
    z = Dropout(hp_dropout)(z)
    z = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[text_input, num_input], outputs=z)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ==========================================
# 6. HYPERPARAMETER TUNING & FINAL TRAINING
# ==========================================
print("\n--- Starting Hyperparameter Search (Max 10 Trials) ---")

# 1. Initialize the Tuner (RandomSearch)
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='keras_tuner_dir',
    project_name='social_media_predictor'
)

tuner.search_space_summary()

# 2. Start the Search (Runs 10 trials, 10 epochs each)
tuner.search(
    [X_text_train, X_num_train],
    y_train,
    epochs=10,
    validation_data=([X_text_test, X_num_test], y_test)
)

# 3. Get the Best Model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# --- CORRECTED LINE for retrieving the model ---
model = tuner.hypermodel.build(best_hps)

print("\n--- Hyperparameters of the Best Model Found ---")
print(best_hps.values)

# 4. Final Training of the Best Model (Creates 'history')
print("\n--- Final Training of the Best Model ---")
history = model.fit(
    [X_text_train, X_num_train],
    y_train,
    epochs=50, # Train the final model for a longer period
    batch_size=32,
    verbose=1,
    validation_data=([X_text_test, X_num_test], y_test)
)

# 5. Save the Final Best Model (Added the save_format='h5' for robustness)
print("\n--- Saving the Best Model to Disk (best_social_media_model.h5) ---")
model.save('best_social_media_model.h5', save_format='h5')

# The script now proceeds to Section 7 (Generating Graphs) and Section 8 (Evaluation)

# ==========================================
# 7. GENERATING GRAPHS
# ==========================================
print("\n--- Generating Training Curves ---")
# Plot 1: Training and Validation Accuracy/Loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: Confusion Matrix
print("\n--- Generating Confusion Matrix ---")
y_pred_probs = model.predict([X_text_test, X_num_test], verbose=0)
y_pred_classes = np.round(y_pred_probs)

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.show()

# ==========================================
# 8. FINAL EVALUATION
# ==========================================
print("\n" + "=" * 50)
print("  --- FINAL MODEL EVALUATION ---")
print("=" * 50)

loss, accuracy = model.evaluate(
    [X_text_test, X_num_test],
    y_test,
    verbose=0
)

# Print the final result clearly as a percentage
print("=" * 50)
print(f"| FINAL MODEL LOSS: {loss:.4f}")
print(f"| FINAL MODEL ACCURACY: {accuracy * 100:.2f}%")
print("=" * 50)

# Ensure the output is immediately displayed
sys.stdout.flush()