# import numpy as np
# import time
# import matplotlib.pyplot as plt

# from src.nn.layer import ReLu, Softmax
# from src.nn.model import Sequential
# from src.nn.loss import CrossEntropy
# from src.nn.optimizer import SGD, AdaGrad, RMSProp, Adam
# from src.nn.early_stopping import EarlyStopping, MonitorEarlyStopping
# from nn.datasets import datasets


# def plot_history(history):
#     plt.figure(figsize=(12, 5))

#     # Loss
#     plt.subplot(1, 2, 1)
#     plt.plot(history["loss"], label="Training Loss")
#     plt.plot(history["val_loss"], label="Validation Loss")
#     plt.title("Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()

#     # Accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(history["accuracy"], label="Training Accuracy")
#     plt.plot(history["val_accuracy"], label="Validation Accuracy")
#     plt.title("Accuracy")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     # X = R(N, d)
#     # Y = R(N, classes)
#     # N = number of samples
#     N, d = 10000, 2
#     classes = 4
#     new_training = True
#     X, Y = datasets(N=N, classes=classes, d=d)
#     T = N * 0.2
#     X_test, Y_test = X[: int(T)], Y[: int(T)]
#     X_train, Y_train = X[int(T) :], Y[int(T) :]

#     # fig, axs = plt.subplots(1, 2, figsize=(8, 8))
#     # axs[0].scatter(
#     #     X_train[:, 0], X_train[:, 1], c=np.argmax(Y_train, axis=1), cmap="viridis", s=10
#     # )
#     # axs[0].set_title("Training Data")
#     # axs[1].scatter(
#     #     X_test[:, 0], X_test[:, 1], c=np.argmax(Y_test, axis=1), cmap="viridis", s=10
#     # )
#     # axs[1].set_title("Test Data")
#     # plt.show()

#     # Layer dimensions
#     d1 = 5

#     # Hyperparameters
#     eta = 1e-1
#     momentum = 0.9
#     nesterov = True
#     weight_decay = 0.0001
#     regularization_lamda = 0.01

#     layers = [ReLu(d, d1), Softmax(d1, classes)]
#     model = Sequential(
#         layers=layers,
#         loss=CrossEntropy(),
#         optimizer=Adam(eta=eta),
#         regularization=None,
#         regularization_lambda=regularization_lamda,
#     )

#     model.summary(d)

#     if not new_training:
#         # Load the model from saved state
#         print("Loading model from saved state...")
#         model.load("./saved")

#     early_stopping = EarlyStopping(
#         patience=20,
#         min_delta=0.1,
#         monitor=MonitorEarlyStopping.VAL_ACCURACY,
#         is_store=True,
#     )

#     hist = model.fit(
#         X_train,
#         Y_train,
#         epochs=10,
#         verbose=True,
#         frequency=1,
#         batch_size=256,
#         early_stopping=early_stopping,
#     )

#     # plot_history(hist)
#     model.save("./saved")

#     # Evaluate the model on the test set
#     test_loss, test_accuracy = model.calculate_loss_accuracy(X_test, Y_test)
#     print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
