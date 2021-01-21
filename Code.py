#MERT DUMANLI 160315002 NORMAL
#(index numbers) - Data 
#1: Bed (1-200)
#2: Bird (201-400)
#3: Cat (401-600)
#4: Dog (601-800)
#5: Down (801-100)
#6: Eight (1001-1200)
#7: Five (1201-1400)
#8: Four (1401-1600)
#9: Go (1601-1800)
#10: Happy (1801-2000)
#11: House (2001-2200)
#12: Left (2201-2400)
#13: Marvin (2401-2600)
#14: Nine (2601-2800)
#15: No (2801-3000)
#16: Off (3001-3200)
#17: On (3201-3400)
#18: One (3401-3600)
#19: Right (3601-3800)
#20: Seven (3801-4000)
#21: Sheila (4001-4200)
#22: Six (4201-4400)
#23: Stop (4401-4600)
#24: Three (4601-4800)
#25: Tree (4801-5000)
#26: Two (5001-5200)
#27: Up (5201-5400)
#28: Wow (5401-5600)
#29: Yes (5601-5800)
#30: Zero (5801-6000)
#______________________________________________________________________________________________________
#adding required libraries
from librosa.feature import mfcc
from sklearn.preprocessing import scale
from glob import glob
import numpy as np
import scipy
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D
from sklearn.model_selection import train_test_split
#______________________________________________________________________________________________________
#importing music
file_list = glob("Data/*.wav")
for filename in file_list:
    base = filename.split('(')[1].lower()#read
    rate, aud = scipy.io.wavfile.read(filename)
    print (aud.shape, rate)
    coeff = mfcc(aud.astype(np.float32), sr=rate, n_mfcc=100)
    coeff = scale(coeff)
    np.save("Features/"+base, coeff)#saving
#______________________________________________________________________________________________________
#Model Design (2D)
def get_model(no_inputs, no_outputs):
    m = Sequential()
    m.add(Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='relu',input_shape=no_inputs))
    m.add(BatchNormalization())
    m.add(MaxPooling2D())
    m.add(Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='sigmoid'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D())
    m.add(Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='tanh'))
    m.add(BatchNormalization())
    m.add(Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='relu'))
    m.add(BatchNormalization())
    m.add(Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='relu'))
    m.add(BatchNormalization())
    m.add(Flatten())
    m.add(Dense(256,activation='relu'))
    m.add(Dropout(0.2))
    m.add(Dense(no_outputs, activation="softmax"))
    
    return m
#______________________________________________________________________________________________________
file_list = glob("Features/*.npy")
trainX = []
trainY = []
idx = 0
category_dict = {}
for filename in file_list:
    base = filename.split('.')[0].lower()
    data = np.load(filename)#loading
    surplus = data.shape[0] % 100
    data = data[:-surplus,:]
    data = np.reshape(data,[-1,100,1])
    datax = []
    for i in np.arange(0,data.shape[0]-100,20):
        datax.append(data[i:i+100])
    datax = np.array(datax)
    
    trainX.append(datax)
    trainY.extend([idx]*datax.shape[0])
    category_dict[idx] = filename
    idx+=1
#______________________________________________________________________________________________________
xtrain, xtest, ytrain, ytest = train_test_split(trainX,trainY,test_size=0.3,random_state=42,shuffle=True)
print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
m = get_model((100,100, 1), 50)
print(m.summary())
m.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
m.fit(xtrain, ytrain, batch_size=256, epochs=300, validation_data=(xtest, ytest))