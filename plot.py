import pickle
import matplotlib.pyplot as plt

history = None
with open('./histosy.pickle', mode='rb') as f:
    history = pickle.load(f)

# plt.plot(history['loss'], "o-", label="loss",)
# plt.plot(history['val_loss'], "o-", label="val_loss")
plt.plot(history['acc'], "o-", label="acc",)
plt.plot(history['val_acc'], "o-", label="val_acc")
plt.title('train history')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='center right')
plt.show()
