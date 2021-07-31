from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import load_model
# from graphviz import Graph
# from pydot_ng import InvocationException
from tensorflow.keras.layers import Dense,Dropout,Flatten,Input,Conv2D
from tensorflow.keras.optimizers import Adam
from process import maybeExtract
import tensorflow as tf
import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_confusion_matrix
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn import metrics
import matplotlib
font = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 6
}
matplotlib.rc('font', **font)
# Define overall_accuracy, avarage_accuracy, kappa coefficient, confusion matrix
def accuracy_eval(label_tr, label_pred):
    overall_accuracy = metrics.accuracy_score(label_tr, label_pred)
    avarage_accuracy = np.mean(metrics.precision_score(label_tr, label_pred, average=None))
    kappa = metrics.cohen_kappa_score(label_tr, label_pred)
    cm = metrics.confusion_matrix(label_tr, label_pred)
    return overall_accuracy, avarage_accuracy, kappa, cm


adam=Adam(learning_rate=0.001)
PATCH=15
testlist=["paviaU",'Indian_pines','Salinas']
TRAIN, TEST = maybeExtract(testlist[0], PATCH)
TRAIN_PATCH_7=TRAIN["train_patch"]
TRAIN_LABELS=TRAIN["train_labels"]
Test_PATCH_7=TEST["test_patch"]
Test_LABLE=TEST["test_labels"]
num=9
x1=TRAIN_PATCH_7.shape[1]
y1=TRAIN_PATCH_7.shape[2]
z1=TRAIN_PATCH_7.shape[3]
TRAIN_PATCH_5,TRAIN_PATCH_3,TRAIN_PATCH_1=TRAIN_PATCH_7[:,1:14,1:14,:],TRAIN_PATCH_7[:,2:13,2:13,:],TRAIN_PATCH_7[:,3:12,3:12,:]
Test_PATCH_5,Test_PATCH_3,Test_PATCH_1=Test_PATCH_7[:,1:14,1:14,:],Test_PATCH_7[:,2:13,2:13,:],Test_PATCH_7[:,3:12,3:12,:]
print(TRAIN_PATCH_7.shape,TRAIN_PATCH_5.shape,TRAIN_PATCH_3.shape,TRAIN_PATCH_1.shape,TRAIN_LABELS.shape)
print(Test_PATCH_7.shape,Test_PATCH_5.shape,Test_PATCH_3.shape,Test_PATCH_1.shape,Test_LABLE.shape)
x2=TRAIN_PATCH_5.shape[1]
y2=TRAIN_PATCH_5.shape[2]
z2=TRAIN_PATCH_5.shape[3]
x3=TRAIN_PATCH_3.shape[1]
y3=TRAIN_PATCH_3.shape[2]
z3=TRAIN_PATCH_3.shape[3]
x4,y4,z4=TRAIN_PATCH_1.shape[1],TRAIN_PATCH_1.shape[2],TRAIN_PATCH_1.shape[3]

# input1
input1=Input(shape=(x1,y1,z1),name="TRAIN_PATCH_7")
Tlayer0=Conv2D(8,(7,7),padding="same",activation="relu")(input1)
Tlayer1=Conv2D(8,(5,5),padding="same",activation="relu")(input1)
Tlayer2=Conv2D(8,(3,3),padding="same",activation="relu")(input1)
Tlayer3=Conv2D(8,(1,1),padding="same",activation="relu")(input1)
Tinput1=tf.concat([Tlayer0,Tlayer1,Tlayer2,Tlayer3],axis=3)

# input2
input2=Input(shape=(x2,y2,z2),name="TRAIN_PATCH_5")
Tlayer0=Conv2D(8,(7,7),padding="same",activation="relu")(input2)
Tlayer1=Conv2D(8,(5,5),padding="same",activation="relu")(input2)
Tlayer2=Conv2D(8,(3,3),padding="same",activation="relu")(input2)
Tlayer3=Conv2D(8,(1,1),padding="same",activation="relu")(input2)
Tinput2=tf.concat([Tlayer0,Tlayer1,Tlayer2,Tlayer3],axis=3)

# input3
input3=Input(shape=(x3,y3,z3),name="TRAIN_PATCH_3")
Tlayer0=Conv2D(8,(7,7),padding="same",activation="relu")(input3)
Tlayer1=Conv2D(8,(5,5),padding="same",activation="relu")(input3)
Tlayer2=Conv2D(8,(3,3),padding="same",activation="relu")(input3)
Tlayer3=Conv2D(8,(1,1),padding="same",activation="relu")(input3)
Tinput3=tf.concat([Tlayer0,Tlayer1,Tlayer2,Tlayer3],axis=3)

# input4
input4=Input(shape=(x4,y4,z4),name="TRAIN_PATCH_1")
Tlayer0=Conv2D(8,(7,7),padding="same",activation="relu")(input4)
Tlayer1=Conv2D(8,(5,5),padding="same",activation="relu")(input4)
Tlayer2=Conv2D(8,(3,3),padding="same",activation="relu")(input4)
Tlayer3=Conv2D(8,(1,1),padding="same",activation="relu")(input4)
Tinput4=tf.concat([Tlayer0,Tlayer1,Tlayer2,Tlayer3],axis=3)



# three convolution of input1
layer1=Conv2D(32,(5,5),padding="same",activation="relu")(Tinput1)
layer2=Conv2D(32,(5,5),padding="same",activation="relu")(layer1+Tinput1)
layer3=Conv2D(32,(5,5),padding="same",activation="relu")(layer2+layer1+Tinput1)
layer4=Conv2D(32,(7,7),padding="valid",activation="relu")(layer3+layer2+layer1+Tinput1)

# three convolution of input2
layer_1=Conv2D(32,(5,5),padding="same",activation="relu")(Tinput2)
layer_2=Conv2D(32,(5,5),padding="same",activation="relu")(layer_1+Tinput2)
layer_3=Conv2D(32,(5,5),padding="same",activation="relu")(layer_2+layer_1+Tinput2)
layer_4=Conv2D(32,(5,5),padding="valid",activation="relu")(layer_3+layer_2+layer_1+Tinput2)

# three convolution of input3
layer__1=Conv2D(32,(5,5),padding="same",activation="relu")(Tinput3)
layer__2=Conv2D(32,(5,5),padding="same",activation="relu")(layer__1+Tinput3)
layer__3=Conv2D(32,(5,5),padding="same",activation="relu")(layer__2+layer__1+Tinput3)
layer__4=Conv2D(32,(3,3),padding="valid",activation="relu")(layer__3+layer__2+layer__1+Tinput3)

# three convolution of input4
layer___1=Conv2D(32,(5,5),padding="same",activation="relu")(Tinput4)
layer___2=Conv2D(32,(5,5),padding="same",activation="relu")(layer___1+Tinput4)
layer___3=Conv2D(32,(5,5),padding="same",activation="relu")(layer___2+layer___1+Tinput4)
layer___4=Conv2D(32,(1,1),padding="valid",activation="relu")(layer___3+layer___2+layer___1+Tinput4)

result1=tf.concat([layer4,layer_4,layer__4,layer___4],axis=3);

layer11=Flatten()(result1)
layer12=Dense(512,activation="relu")(layer11)
layer13=Dropout(0.5)(layer12)
layer14=Dense(256,"relu")(layer13)
layer15=Dropout(0.5)(layer14)
predict=Dense(9,activation="softmax")(layer15)

model=Model(inputs=[input1,input2,input3,input4],outputs=predict)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
plot_model(model,to_file="modelHIS(2).png")
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath,  verbose=1, save_best_only=True,mode='auto')
callbacks_list = [checkpoint]
history= model.fit({"TRAIN_PATCH_7":TRAIN_PATCH_7,"TRAIN_PATCH_5":TRAIN_PATCH_5,"TRAIN_PATCH_3":TRAIN_PATCH_3,"TRAIN_PATCH_1":TRAIN_PATCH_1},TRAIN_LABELS,batch_size=128,epochs=50,callbacks=callbacks_list,validation_split=0.5)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model_best=load_model("weights.best.hdf5")
print("Load the best model successfully, start testing:")
y_pred=model_best.predict([Test_PATCH_7,Test_PATCH_5,Test_PATCH_3,Test_PATCH_1])
# mat=confusion_matrix(Test_LABLE.argmax(axis=1),y_pred.argmax(axis=1))
# loss,acc=model_best.evaluate([Test_PATCH_7,Test_PATCH_5,Test_PATCH_3,Test_PATCH_1],Test_LABLE)

print(classification_report(Test_LABLE.argmax(axis=1),y_pred.argmax(axis=1),digits=4))
OA, AA, Kappa, cm = accuracy_eval(Test_LABLE.argmax(axis=1), y_pred.argmax(axis=1))
print (" OA: %g, AA: %g, Kappa:%g " % ( OA, AA, Kappa))
print()
# plot_confusion_matrix(conf_mat=mat,figsize=(15,15),show_normed=True)
print("Data:{}".format(testlist[2]))
# print("Test_Loss:",loss)
# print("Test_Accuracy:",acc)


