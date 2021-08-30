from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Embedding, Dense, Dropout, LSTM, Conv1D, Conv2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, Masking

def import_architecture(architecture):
    if architecture == 'fc':
        return fc
    elif architecture == 'cnn1d':
        return cnn1d
    elif architecture == 'lstm':
        return lstm
    elif architecture == 'attention':
        return attention
    else:
        raise Exception(f"{architecture} is not a valid architecture value.")

def fc(input_shape, hidden_units=32):
    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(*input_shape[1:],), activation='relu'))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])
    return model

def cnn1d(input_shape, channels=44, hidden_units=32, kernel_size=3):
    model = Sequential()
    model.add(Conv1D(44, kernel_size, input_shape=(*input_shape[1:],), activation='relu'))
    model.add(Flatten())
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])
    return model

def lstm(input_shape, recurrent_units=55, hidden_units=32):
    model = Sequential()
    model.add(LSTM(recurrent_units, input_shape=(*input_shape[1:],)))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])
    return model

def attention(input_shape, attention_units=32, hidden_units=30):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(*input_shape[1:],)))
    model.add(SeqSelfAttention(units=attention_units, attention_activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])
    return model


