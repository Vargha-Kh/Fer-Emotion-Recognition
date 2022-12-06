from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import load_model
from hp import load_hps
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from datasets.dataset import Dataset
from plotting import plot
import tensorflow_addons as tfa
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
tf.keras.backend.clear_session()


def model_evaluation(model, test_gen):
    pred = (model.predict(test_gen) > 0.5).astype("int32")
    y_test = test_gen.labels
    print("Model Prediction:")
    print('Classification report:\n', classification_report(y_test, pred))
    print('Accuracy score:\n', accuracy_score(y_test, pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, pred))
    print("\nModel Evaluation on Test Set:")
    model.evaluate(test_gen, batch_size=200)


def train():
    hps = load_hps(dataset_dir="./fer2013/", model_name='custom_model', n_epochs=300, batch_size=256,
                   learning_rate=0.001,
                   lr_reducer_factor=0.1,
                   lr_reducer_patience=12, img_size=48, split_size=0.25, framework='keras')
    model = load_model(model_name=hps['model_name'])
    # model = models.load_model('./best_model.h5')
    
    
    METRICS = [
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'),
        metrics.CategoricalAccuracy(name='categorical_accuracy'),
        metrics.Accuracy(name='accuracy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc')
    ]

    optimizer = tfa.optimizers.AdamW(
        learning_rate=hps['learning_rate'], weight_decay=0.01
    )

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=["accuracy"]
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=hps['lr_reducer_factor'],
                                  patience=hps['lr_reducer_patience'],
                                  verbose=1,
                                  min_delta=0.00001)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=50,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    tensorboard_callback = TensorBoard(log_dir="./logs")

    model_checkpoint_acc = ModelCheckpoint("./best_model.h5", monitor='val_accuracy', save_best_only=True,
                                           verbose=1)
    if hps['framework'] == 'keras':
        train_generator, validation_generator = Dataset.keras_preprocess(
            dataset_dir=hps['dataset_dir'],
            img_size=hps['img_size'],
            batch_size=hps['batch_size'],
            augment=True, split_size=hps['split_size'])

        callbacks = [reduce_lr, model_checkpoint_acc, early_stopping, tensorboard_callback]
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // hps['batch_size'],
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // hps['batch_size'],
            epochs=hps['n_epochs'],
            callbacks=callbacks,
            verbose=1)
        plot(history)
        # model_evaluation(model, validation_generator)


if __name__ == '__main__':
    train()
