"""
Created on Mon Jan  7 13:46:32 2019
@author: jtreguer
Validates LSTM with attention prediction of a sinus series
using a cosinus series.
Data are passed on to the model via a generator function
"""
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import initializers
from keras import regularizers
from keras import optimizers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


print("LSTM model with sinus")
# Load data
SAMPLES = 5000
PERIOD = 50
t = np.linspace(-PERIOD * np.pi, PERIOD * np.pi, SAMPLES)
X = np.cos(t)
Y = np.sin(t)

print(len(t))

##############################
# Create input vector for LSTM
##############################
  
# Version with 2 distinct arrays

def gen_data(data_in,data_out, batch_size=100, timesteps=50,dim = 1, exit_layer = 10):
    total_length = len(data_in) - timesteps
    batch_nbr = total_length // batch_size
    last_batch_size = total_length % batch_size
    while True:
        batch_count = 1
        while batch_count <= batch_nbr:
            s0 = np.zeros((batch_size, layer2))
            c0 = np.zeros((batch_size, layer2))            
            outputs = data_out[timesteps+(batch_count-1)*batch_size:timesteps+batch_count*batch_size].reshape((batch_size,1))
            selected_inputs = data_in[(batch_count-1)*batch_size:batch_count*batch_size+timesteps]
            batch_count = batch_count + 1
            inputs = []
            for j in range(batch_size):
                inputs.append(selected_inputs[j:j+timesteps].reshape((timesteps,dim)))
            yield [np.array(inputs),s0,c0], outputs    
        outputs = data_out[timesteps+(batch_count-1)*batch_size:timesteps+(batch_count-1)*batch_size+last_batch_size].reshape((last_batch_size,1))     
        inputs = []
        s0 = np.zeros((last_batch_size, exit_layer))
        c0 = np.zeros((last_batch_size, exit_layer))
        selected_inputs = data_in[(batch_count-1)*batch_size:(batch_count-1)*batch_size+timesteps+last_batch_size]
        for j in range(last_batch_size):
            inputs.append(selected_inputs[j:j+timesteps].reshape((timesteps,dim)))
        yield [np.array(inputs),s0,c0], outputs

        

# Split
wf_train = 0.8
wf_test = 0.1
L = len(Y)
o_train = int(round(L*wf_train))
o_test = int(np.ceil(L*wf_test))
            
X_training = X[:o_train]
Y_training = Y[:o_train]
X_val = X[o_train:o_train+o_test]
Y_val = Y[o_train:o_train+o_test]


# Model builder

def one_step_attention(a, s_prev,timesteps):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = layers.RepeatVector(timesteps)(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = layers.Concatenate(axis=2)([s_prev,a]) # (Tx,n_s)
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = layers.Dense(10, activation = "tanh")(concat) # 10
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = layers.Dense(1, activation = "relu")(e) # 1
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = layers.Dense(1, activation = 'softmax')(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = layers.Dot(axes = 1)([alphas,a])

    
    return context

def attention_model(idim, timesteps, n_1,n_2):
    
    X = layers.Input(shape=(timesteps,idim))
    s0 = layers.Input(shape=(n_2,), name='s0')
    c0 = layers.Input(shape=(n_2,), name='c0')
    s = s0
    c = c0

#    outputs = []
    
    a = layers.LSTM(n_1,return_sequences=True)(X)
    
    context = one_step_attention(a,s,timesteps)
    
    s, _, c = layers.LSTM(n_2, return_state = True)(context,initial_state=[s,c]) # return_state = True => last state or last sequene of states / last state / cell state

    out = layers.Dense(1, activation='tanh')(s)

#    outputs.append(out)
    
    model = models.Model(inputs=[X,s0,c0],outputs=out)
    
   
    return model 

# Instantiate and compile
ndim = 1
BS = 20
look_back = 50
layer1 = 32
layer2 = 64
epochs = 100
sampling = 10
model = attention_model(ndim, look_back,layer1,layer2)
opt = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.01)
model.compile(loss = 'mse', metrics = ['mse'], optimizer=opt)
    
# Train neural network with fit
if 0:
    def make_data(X,Y,timesteps= 5):
        inputs = np.empty((len(X)-timesteps,timesteps))
        for i in range(timesteps,len(X)):
            inputs[i-timesteps] = X[i-timesteps:i]
        return inputs.reshape((inputs.shape[0],inputs.shape[1],1)), Y[timesteps:]
    
    X_tr, Y_tr = make_data(X_training,Y_training,timesteps=look_back)
    X_vl, Y_vl = make_data(X_val,Y_val, timesteps=look_back)
    
    print(X_tr.shape,Y_tr.shape)
    
    s0 = np.zeros((o_train-look_back, layer2))
    c0 = np.zeros((o_train-look_back, layer2))
    s0_vl = np.zeros((o_test-look_back, layer2))
    c0_vl = np.zeros((o_test-look_back, layer2))
    
    history = model.fit([X_tr, s0, c0], Y_tr,
                        verbose = 1,
                        epochs=epochs,
                        batch_size=BS,
                        validation_data= ([X_vl,s0_vl,c0_vl],Y_vl))

# train with generator fit

train_gen = gen_data(X_training,Y_training,batch_size=BS,timesteps=look_back,dim=1, exit_layer = layer2)
val_gen = gen_data(X_val,Y_val,batch_size=BS,timesteps=look_back,dim=1, exit_layer = layer2)

# Callback to track progress
from keras.callbacks import Callback

spe = ((o_train - look_back) // BS)+1
val_steps = ((o_test - look_back) // BS)+1



class pred_perf(Callback):
    def __init__(self):
        self.preds = []
        self.targets = []        
    
    def on_train_begin(self, logs={}):
        self.y_pred = []
        self.y_train = []
        
    def on_epoch_begin(self, epoch, logs={}):
        self.data = gen_data(X_training,Y_training,batch_size=BS,timesteps=look_back,dim=1,exit_layer = layer2)   
    
    def on_epoch_end(self, epoch, logs={}):
        if epoch % sampling:
            return
        else:
            y_pred_arr = np.zeros((o_train - look_back,1))
            y_true_arr = np.zeros((o_train - look_back,1))
            
            for i in range(spe-1):
                x_tr, y_tr = next(self.data)
                y_pred_arr[i*BS:(i+1)*BS] = self.model.predict(x_tr)
                y_true_arr[i*BS:(i+1)*BS] = y_tr
            y_pred_arr = np.squeeze(y_pred_arr)    
            self.preds.append(y_pred_arr)
            self.targets.append(y_true_arr)
            return

predict_epoch_callback = pred_perf()


history = model.fit_generator(train_gen,
                              epochs = epochs,
                              verbose = 1,
                              steps_per_epoch = spe,
                              validation_steps = val_steps,
                              validation_data = val_gen,
                              use_multiprocessing = False,
                              callbacks = [predict_epoch_callback])




plt.figure(figsize=(9,9))
plt.title("Loss vs epochs")
plt.xlabel("epochs")
plt.ylabel("Losses")
plt.plot(range(1,epochs+1),history.history['loss'])
plt.plot(range(1,epochs+1),history.history['val_loss'])
plt.legend(['Loss training','Loss test'])
plt.savefig("LSTM_online_attention_loss_epochs_"+str(layer1)+"_"+str(layer2)+".png")

plt.figure(figsize=(18,9))
plt.title('Predictions vs truth')
lab = []
plt.plot(predict_epoch_callback.targets[0][:300])
for i in range(epochs):
    if (i % sampling == 0):
        ci = (1 - i / epochs)
        plt.plot(predict_epoch_callback.preds[i // sampling][:300], c =(ci,ci,ci))
        lab.append('predicted_'+str(i))
lab.append('Truth')
plt.legend(lab)

plt.figure(figsize=(9,9))
plt.title("RMSE vs epochs")
plt.xlabel("epochs")
plt.ylabel("RMSE")
plt.plot(range(1,epochs+1),np.sqrt(history.history['mean_squared_error']))
plt.plot(range(1,epochs+1),np.sqrt(history.history['val_mean_squared_error']))
plt.legend(['RMSE training','RMSE test'])
#plt.savefig("LSTM_online_rmse_epochs_"+str(layer1)+"_"+str(layer2)+".png")

#print("Min test MAE ",min(history.history['val_mean_absolute_error']))
print("Min test RMSE ",np.sqrt(min(history.history['val_mean_squared_error'])))
#print("Max test R^2 ",max(history.history['val_r2_keras']))
rmse = np.sqrt(history.history['val_mean_squared_error'])
print("Epoch of min validation RMSE %i " % np.argmin(rmse))

#lstm_nn.save("lstm_online_"+str(layer1)+"_"+str(layer2)+".h5")

# Evaluate prediction with generator

s = t[o_test+o_train:SAMPLES]
X_test = np.cos(s)
Y_test = np.sin(s)
#X_ts, Y_ts= make_data(X_test,Y_test,timesteps=look_back)
test_gen = gen_data(X_test,Y_test,batch_size=BS,timesteps=look_back,dim=1)

#s0_ts = np.zeros((len(s)-look_back, layer2))
#c0_ts = np.zeros((len(s)-look_back, layer2))
#preds = model.predict([X_ts,s0_ts,c0_ts])

#model.evaluate_generator(test_gen,
#                         steps = int(np.ceil((SAMPLES-o_train-o_test) // BS)))
preds = model.predict_generator(test_gen,
                         steps = ((SAMPLES-o_train-o_test - look_back) // BS) + 1)
Y_ts = Y_test[look_back:]

plt.figure(figsize=(10,5))
plt.plot(preds)
plt.plot(Y_ts)
plt.plot(X_test)
plt.legend(('preds','Y','X'))

print(mean_squared_error(Y_ts,preds))