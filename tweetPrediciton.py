import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

#57% Accurate, I originally had issues with my model giving each tweet the same predicted emotions, but using the Bidirectional LSTM, it improved
#My model significantly.

def main():

    trainingFrame = pd.read_csv('train.csv')
    X_train = trainingFrame['Tweet'].values
    Y_train = trainingFrame.iloc[:, 2:].values

    #We will now load our validation data into a data fram

    validationFrame= pd.read_csv('validation.csv')
    X_val= validationFrame['Tweet']
    Y_val = validationFrame.iloc[:, 2:].values

    testFrame = pd.read_csv('test.csv')
    X_test = testFrame['Tweet'].values
    Y_test = testFrame.iloc[:, 2:].values

    #Let's now tokenize our text.

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    #now to convert the text into sequences.

    xTrainSequences = tokenizer.texts_to_sequences(X_train)
    xValidationSequences = tokenizer.texts_to_sequences(X_val)
    xTestSequences = tokenizer.texts_to_sequences(X_test)
    #We should pad our sequence to ensure they have a consistent length.

    xTrainPadded = pad_sequences(xTrainSequences, maxlen=180, padding ='post',truncating='post')
    xValidationPadded= pad_sequences(xValidationSequences, maxlen=180, padding= 'post', truncating='post')
    xTestPadded = pad_sequences(xTestSequences, maxlen=180, padding ='post', truncating='post')

    vocab_size = len(set(xTrainPadded.flatten()))+1
    input_length = xTrainPadded.shape[1]
    embedding_dim = 50
    lstm_units=250
    output_dim = 11

    model= Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length = input_length))
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    model.add(Bidirectional(LSTM(lstm_units))) # added the bidirectional LSTM increased my accuracy by 20%
    #model.add(SpatialDropout1D(0.2))
    #model.add(LSTM(lstm_units))
    model.add(Dense(output_dim, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics =['accuracy'])

    model.fit(xTrainPadded, Y_train, epochs = 5, batch_size= 32, validation_data=(xValidationPadded,Y_val))

    test_loss, test_accuracy = model.evaluate(xTestPadded, Y_test)

    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    predictions = model.predict(xTestPadded)

    predictedEmotions = (predictions>0.3).astype(int)

    for i, tweet in enumerate(X_test):
        print(f"Sample {i + 1} - Tweet: {tweet}")

        emotion_label =testFrame.columns[2:]
        predictedIndicies = np.where(predictedEmotions[i]==1)[0]
        predictedLabels = emotion_label[predictedIndicies]



        print("Predicted Emotions:", ", ".join(predictedLabels))
        print()


if __name__ == "__main__":
    main()