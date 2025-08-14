import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

print("Versão do Tensorflow: ", tf.__version__)
print("GPU disponível: ", tf.config.list_physical_devices('GPU'))

(ds_train, ds_test), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

IMG_SIZE = 160
BATCH_SIZE = 32

def preprocess(image, label):
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  image = tf.cast(image, tf.float32) / 255.00
  return image, label

train_ds = ds_train.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1)
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(train_ds, validation_data=test_ds, epochs=25)

loss, acc = model.evaluate(test_ds)
print(f"Acurácia final: {acc*100:.2f}%")

def show_one_prediction(dataset, model):
    for batch in dataset.take(1):
        # batch can be (images, labels) either batched or not
        images, labels = batch
        if len(images.shape) == 4:   # batched -> (B,H,W,3)
            img = images[3]
            lab = labels[3]
        else:                        # não batched -> (H,W,3)
            img = images
            lab = labels

        img_np = img.numpy()
        # se a imagem estiver pré-processada para [-1,1] (MobileNet preprocess), reescale para exibir:
        if img_np.min() < 0:
            disp = (img_np + 1.0) / 2.0
        else:
            disp = img_np

        plt.imshow(np.clip(disp, 0, 1))
        plt.axis("off")
        pred = model.predict(np.expand_dims(img_np, axis=0))[0,0]
        print("Prob.(Cachorro):", float(pred))
        print("Previsão:", "Cachorro" if pred >= 0.5 else "Gato", "| Valor real:", "Cachorro" if int(lab.numpy())==1 else "Gato")
        break

show_one_prediction(test_ds, model)
