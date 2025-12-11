# Train and test a CNN on the MNIST dataset using Keras (TensorFlow)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (
    Input, Reshape, Conv2D, MaxPooling2D,
    BatchNormalization, Dropout, Flatten, Dense
)
from keras.models import Model

#---------------------------------------------------------
# 関数       : build_model
# 概要       : MNIST手書き数字認識のためのCNNモデルを構築し、
#              学習に使用するKerasモデルを生成して返す
# 入力値     : なし
# 戻り値     : 構築されたKerasモデルオブジェクト
# 処理内容   :
#              1. 入力画像(28×28)を受け取り、形状を(28×28×1)に変換する
#              2. 畳み込み層とプーリング層を組み合わせて特徴量を抽出する
#              3. Flattenで1次元に変換する
#              4. 全結合層(Dense)により分類処理を行う
#              5. 出力層でsoftmaxを用いて0〜9の10クラス分類を行う
#---------------------------------------------------------
def build_model():
    # Input: 28x28 grayscale image
    inputs = Input(shape=(28, 28), name="input_1")

    # (28, 28) → (28, 28, 1) を変換して、Conv2D が期待する 1 チャンネルのグレースケール画像を表する
    x = Reshape((28, 28, 1), name="reshape_1")(inputs)

    # Block 1: 16 filters
    x = Conv2D(16, (3, 3), padding="same", activation="relu", name="conv2d_1")(x)
    x = Conv2D(16, (3, 3), padding="same", activation="relu", name="conv2d_2")(x)
    x = MaxPooling2D((2, 2), name="max_pooling2d_1")(x)        # Spatial size: 28×28 → 14×14;
    x = BatchNormalization(name="batch_normalization_1")(x)

     # Block 2: 32 filters
    x = Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_3")(x)
    x = Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_4")(x)
    x = MaxPooling2D((2, 2), name="max_pooling2d_2")(x)        # Spatial size: 14×14 → 7×7.
    x = BatchNormalization(name="batch_normalization_2")(x)

    # Block 3 : 64 filters
    x = Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2d_5")(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2d_6")(x)
    x = MaxPooling2D((2, 2), name="max_pooling2d_3")(x)        # Spatial size: 7×7 → 3×3 (because 7/2 ≈ 3 with floor).
    x = BatchNormalization(name="batch_normalization_3")(x)

    # Block 4 : 128 filters
    x = Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2d_7")(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2d_8")(x)
    x = MaxPooling2D((3, 3), name="max_pooling2d_4")(x)        # -> (1, 1, 128)
    x = BatchNormalization(name="batch_normalization_4")(x)

    # Classifier head 
    x = Dropout(0.5, name="dropout_1")(x)  # drops 50% of units during training to reduce overfitting.
    x = Flatten(name="flatten_1")(x)  # converts (1,1,128) → (128,)                  
    x = Dense(9, activation="relu", name="dense_1")(x) # fully connected layer with 9 units 
    outputs = Dense(10, activation="softmax", name="dense_2")(x)  # 10 classes
    #outputs = Dense(1, activation="sigmoid", name="dense_2")(x)  
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

#---------------------------------------------------------
# 関数       : show_sample_predictions
# 概要       : テストデータからランダムに画像を選択し、
#              モデルの予測結果と正解ラベルを描画して表示する
# 入力値     : model      - 学習済みモデル
#              x_test     - テスト画像データ
#              y_test     - テストラベル
#              num_samples- 表示する画像数（デフォルト10）
# 戻り値     : なし（matplotlibで結果を表示）
# 処理内容   :
#              1. テストデータからnum_samples個をランダム抽出する
#              2. 抽出画像をモデルに入力し、予測確率を取得する
#              3. softmax出力からargmaxで予測ラベルを求める
#              4. 画像を描画し、正解(T:)と予測(P:)ラベルを表示する
#---------------------------------------------------------
def show_sample_predictions(model, x_test, y_test, num_samples=10):
    # Randomly choose num_samples test indices
    idx = np.random.choice(len(x_test), num_samples, replace=False)

    images = x_test[idx]
    labels = y_test[idx]

    # Predict probabilities
    preds = model.predict(images)
    pred_labels = np.argmax(preds, axis=1)

    # Plot results
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        plt.title(f"T:{labels[i]}\nP:{pred_labels[i]}")
    plt.tight_layout()
    plt.show()

#---------------------------------------------------------
# メイン処理部分
#---------------------------------------------------------
if __name__ == "__main__":
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess data
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # Build model
    model = build_model()
    model.summary()

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=2
    )

    # Evaluate on test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    # Save model
    model.save("mnist_cnn_model.h5")
    print("Model saved as mnist_cnn_model.h5")

    # Test on a few images and show predictions
    show_sample_predictions(model, x_test, y_test)
