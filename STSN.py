import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.utils import np_utils#one_hot编码
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D, MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from load_data import Get_All_Data
import tensorflow as tf
from keras_flops import get_flops
import keras.backend
tf.random.set_seed(2)
import numpy as np
np.set_printoptions(threshold=np.inf)
import time, os
import keras
from metrics import evaluate_performance
keras.backend.set_image_data_format('channels_last')
from keras.layers import *
from keras.models import *
from keras.utils.vis_utils import plot_model #visualize model
from keras.models import load_model
from keras.optimizers import Adam
from load_data import Get_All_Data
# os.chdir('D:/论文2/upload to GitHub/')
os.environ["PATH"] += os.pathsep + 'E:/Program Files (x86)/Graphviz2.38/bin' #used for visualizing the model

global_start_time = time.time()
#1.导入数据集
# X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b,X_train_2,X_test_2=\
# 	Get_All_Data(TG=35, time_lag=7, TG_in_one_day=72, forecast_day_number=5, TG_in_one_week=360)
# print('训练集和测试集的大小分别是：', X_train_2.shape,Y_train.shape, X_test_2.shape,Y_test.shape)#训练集和测试集
#
# X_train_1 = np.array(X_train_1)
# X_test_1 = np.array(X_test_1)
# print(type(X_test_1))
# X_train_1 = X_train_1.reshape(X_train_1.shape[0],  53, 6, 1)
# X_test_1 = X_test_1.reshape(X_test_1.shape[0],  53, 6, 1)
# print(X_train_1.shape)
# #
# # #2.模型构建
# # #2.1 普通时序模型建立
# # model = Sequential()
# # model.add(Convolution2D(
# # 	filters=32,
# # 	kernel_size=[3,3],
# # 	input_shape=(53,6,1),
# # 	padding="same",
# # 	activation='relu'
# # ))
# # model.add(MaxPooling2D(
# #     pool_size=(2,2),
# #     strides=(2,2),
# #     padding="same",#padding method
# # ))
# # model.add(Convolution2D(
# # 	filters=64,
# # 	kernel_size=[3,3],
# # 	input_shape=(53,6,1),
# # 	padding="same",
# # 	activation='relu'
# # ))
# # model.add(MaxPooling2D(
# #     pool_size=(2,2),
# #     strides=(2,2),
# #     padding="same",#padding method
# # ))
# # model.add(Flatten())
# # model.add(Dense(1024,activation='relu'))
# # model.add(Dense(53,activation='relu'))
# #
# # #3.模型训练
# #
# # # We add metrics to get more results you want to see
# # model.compile(loss='mae', optimizer='sgd')
# #
# # #4.训练
# # print("Tranining-----------------")
# # for step in range(301):
# #     cost = model.train_on_batch(X_train, Y_train)#训练的时候是按照批次来训练
# #     #训练的返回值是LOSS损失函数的值
# #     if step %100 == 0:
# #         print('train_cost:', cost)
# #
# # #5.训练
# # print("Tranining-----------------")
# # cost = model.evaluate(X_test, Y_test, batch_size=40)#测试的时候用的是evaluate去评估，用测试集评估，batch_siza是需要的参数，这里只有40个就全部传入
# # print('trset_cost:', cost)
# #
# #
#
# X_train_2 = np.array(X_train_2)
# X_test_2 = np.array(X_test_2)
# print(type(X_test_2))
# X_train = X_train_2.reshape(X_train_2.shape[0],  53, 6, 1)
# X_test = X_test_2.reshape(X_test_2.shape[0],  53, 6, 1)
# print(X_train_2.shape)
# #
# # #2.模型构建
# # #2.1 普通时序模型建立
# # model = Sequential()
# # model.add(Convolution2D(
# # 	filters=32,
# # 	kernel_size=[3,3],
# # 	input_shape=(53,6,1),
# # 	padding="same",
# # 	activation='relu'
# # ))
# # model.add(MaxPooling2D(
# #     pool_size=(2,2),
# #     strides=(2,2),
# #     padding="same",#padding method
# # ))
# # model.add(Convolution2D(
# # 	filters=64,
# # 	kernel_size=[3,3],
# # 	input_shape=(53,6,1),
# # 	padding="same",
# # 	activation='relu'
# # ))
# # model.add(MaxPooling2D(
# #     pool_size=(2,2),
# #     strides=(2,2),
# #     padding="same",#padding method
# # ))
# # model.add(Flatten())
# # model.add(Dense(1024,activation='relu'))
# # model.add(Dense(53,activation='relu'))
# #
# # #3.模型训练
# #
# # # We add metrics to get more results you want to see
# # model.compile(loss='mae', optimizer='sgd')
# #
# # #4.训练
# # print("Tranining-----------------")
# # for step in range(301):
# #     cost = model.train_on_batch(X_train, Y_train)#训练的时候是按照批次来训练
# #     #训练的返回值是LOSS损失函数的值
# #     if step %100 == 0:
# #         print('train_cost:', cost)
# #
# # #5.训练
# # print("Tranining-----------------")
# # cost = model.evaluate(X_test, Y_test, batch_size=40)#测试的时候用的是evaluate去评估，用测试集评估，batch_siza是需要的参数，这里只有40个就全部传入
# # print('trset_cost:', cost)
# #
# #
#
# def Unit(x, filters, pool=False):
# 	res = x
# 	if pool:
# 		x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
# 		res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same")(res)
# 	out = BatchNormalization()(x)
# 	out = Activation("relu")(out)
# 	out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)
#
# 	out = BatchNormalization()(out)
# 	out = Activation("relu")(out)
# 	out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)
#
# 	out = keras.layers.add([res, out])
#
# 	return out
#
# def attention_3d_block(inputs,timesteps):
#     #input_dim = int(inputs.shape[2])
#     a = Permute((2, 1))(inputs)
#     a = Dense(timesteps, activation='linear')(a)
#     a_probs = Permute((2, 1))(a)
#     #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
#     output_attention_mul = multiply([inputs, a_probs])
#     return output_attention_mul
# #1.定义模型
# def multi_input_model(time_lag):
#     """build multi input model构建多输入模型"""
#     input1_ = Input(shape=(53, time_lag-1, 3), name='input1')#模块一，时序提取模块，原始数据X_train_1
#     input2_ = Input(shape=(53, time_lag-1, 3), name='input2')#模块二，空间异质提取模块，GCN数据X_train_3
#     input3_ = Input(shape=(53, time_lag-1, 1), name='input3')#模块三，网络性能提升模块，原始数据X_train_1
#
#
#     #first model
#     x1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input1_)
#     x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(x1)
#     x1 = Conv2D(filters=32, kernel_size=[1, 1], strides=(2, 2), padding="same")(x1)
#     x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x1)
#     x1 = Flatten()(x1)
#     x1 = Dense(53)(x1)
#
#     # second model，数据需要修改
#     shared_layer_one = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")
#     x21 = shared_layer_one(input2_)
#     x21 = Flatten()(x21)
#     x21 = Dense(53)(x21)
#
#     x22 = shared_layer_one(input2_)
#     x22 = Flatten()(x22)
#     x22 = Dense(53)(x22)
#
#     x2 = keras.layers.add([x21, x22])
#
#
#     # third model
#     x3 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input3_)
#     x3 = Unit(x3, 32)
#     x3 = Unit(x3, 64, pool=True)
#     x3 = Flatten()(x3)
#     x3 = Dense(53)(x3)
#
#     out = keras.layers.add([x1, x2, x3])
#
#     out = Reshape(target_shape=(53, 1))(out)
#
#     out = LSTM(128, return_sequences=True,input_shape=(53, 1))(out)
#     out = attention_3d_block(out, 53)#shape of the output is（53，128）
#
#     out = Flatten()(out)
#     out = Dense(53)(out)
#
#     model = Model(inputs=[input1_, input2_, input3_], outputs=[out]) #[input1_, input2_, input3_]
#     return model
#
# def build_model(X_train_1,X_train_2,X_train_3,X_train_4,Y_train,X_test_1,X_test_2,X_test_3,X_test_4,Y_test,\
# 	Y_test_original,batch_size,epochs,a,time_lag):
#
# 	X_train_1 = X_train_1.reshape(X_train_1.shape[0],  53, time_lag-1, 3)
# 	X_train_2 = X_train_2.reshape(X_train_2.shape[0],  53, time_lag-1, 3)
# 	X_train_3 = X_train_3.reshape(X_train_3.shape[0],  53, time_lag-1, 1)
# 	X_train_4 = X_train_4.reshape(X_train_4.shape[0],  53, time_lag-1, 1)
# 	Y_train = Y_train.reshape(Y_train.shape[0], 53)
#
# 	print('X_train_1.shape is', X_train_1.shape)
# 	print('X_train_1.type is', type(X_train_1))
# 	print('X_train_2.shape is', X_train_2.shape)
# 	print('X_train_3.shape is', X_train_3.shape)
# 	print('Y_train.shape is', Y_train.shape)
#
# 	X_test_1 = X_test_1.reshape(X_test_1.shape[0],  53, time_lag-1, 3)#(1075,276,5,3),1075个时间段的276个站点的5个前后时间段的三个不同的维度
# 	X_test_2 = X_test_2.reshape(X_test_2.shape[0],  53, time_lag-1, 3)
# 	X_test_3 = X_test_3.reshape(X_test_3.shape[0],  53, time_lag-1, 1)
# 	X_test_4 = X_test_4.reshape(X_test_4.shape[0],  53, time_lag-1, 1)
# 	Y_test = Y_test.reshape(Y_test.shape[0], 53)
#
# 	if epochs == 50:
# 		model = multi_input_model(time_lag)
# 		model.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
# 		model.summary()
# 		model.fit([X_train_1, X_train_2, X_train_3, X_train_4], Y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=False)#, validation_split=0.05
# 		output = model.predict([X_test_1, X_test_2, X_test_3, X_test_4], batch_size=batch_size)
# 	else:
# 		# train models every 10 epoches
# 		model = load_model('testresult/'+str(epochs-10)+'-model-with-graph.h5')
# 		model.fit([X_train_1, X_train_2, X_train_3, X_train_4], Y_train, batch_size=batch_size, epochs=10, verbose=2, shuffle=False)# , validation_split=0.05
# 		output = model.predict([X_test_1, X_test_2, X_test_3, X_test_4], batch_size=batch_size)
#
# 	#rescale the output of this model将输出进行反归一化
# 	predictions = np.zeros((output.shape[0], output.shape[1]))
# 	for i in range(len(predictions)):
# 		for j in range(len(predictions[0])):
# 			predictions[i, j] = round(output[i, j]*a, 0)
# 			if predictions[i, j] < 0:
# 				predictions[i, j] = 0
#
# 	RMSE,R2,MAE,WMAPE=evaluate_performance(Y_test_original,predictions)
# 	#visualize the model structure
# 	plot_model(model, to_file='model.png', show_shapes=True)
# 	#print(model.summary())
#
# 	return model,Y_test_original,predictions,RMSE,R2,MAE,WMAPE

# os.chdir('D:/论文2/upload to GitHub/')
os.environ["PATH"] += os.pathsep + 'E:/Program Files (x86)/Graphviz2.38/bin' #used for visualizing the model

global_start_time = time.time()

def Unit(x, filters, pool=False):
	res = x
	if pool:
		x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
		res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same")(res)
	out = BatchNormalization()(x)
	out = Activation("relu")(out)
	out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

	out = BatchNormalization()(out)
	out = Activation("relu")(out)
	out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

	out = keras.layers.add([res, out])

	return out

def attention_3d_block(inputs,timesteps):
	# inputs.shape = (batch_size, time_steps, input_dim)
    #input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)#将输入维度进行转换(batch_size, input_dim, time_steps)
    a = Dense(timesteps, activation='softmax')(a)#计算attention权重，该层有timesteps个神经元
    a_probs = Permute((2, 1))(a)
    print(a_probs)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

# Define the model
def multi_input_model(time_lag):
    """build multi input model构建多输入模型"""
    input1_ = Input(shape=(53, time_lag-1, 3), name='input1')
    input2_ = Input(shape=(53, time_lag-1, 3), name='input2')
    input3_ = Input(shape=(53, time_lag-1, 1), name='input3')
    input4_ = Input(shape=(53, time_lag-1, 1), name='input4')

    #first model时序特征维度提取模块
    x1 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input1_)
    x1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(x1)
    x1 = Conv2D(filters=32, kernel_size=[1, 1], strides=(2, 2), padding="same")(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x1)
    x1 = Flatten()(x1)
    x1 = Dense(53)(x1)

    # second model网络性能调整模块
    x2 = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(input2_)
    x2 = Unit(x2, 32, pool=True)
    x2 = Unit(x2, 64, pool=True)
    x2 = Flatten()(x2)
    x2 = Dense(53)(x2)


    # third model异质空间特征提取模块
    shared_layer_one = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")
    x31 = shared_layer_one(input3_)
    x31 = Flatten()(x31)
    x31 = Dense(53)(x31)

    x32 = shared_layer_one(input4_)
    x32 = Flatten()(x32)
    x32 = Dense(53)(x32)

    out = keras.layers.add([x1, x2, x31, x32])


    out = Reshape(target_shape=(53, 1))(out)

    out = LSTM(128, return_sequences=True,input_shape=(53, 1))(out)
    out = Conv1D(filters=128, kernel_size=3, strides=1, padding="same", input_shape=(53, 128), data_format=None)(out)

    out = attention_3d_block(out, 53)#shape of the output is（53，128）
    out = Flatten()(out)
    out = Dense(53)(out)

    model = Model(inputs=[input1_, input2_, input3_, input4_], outputs=[out]) #[input1_, input2_, input3_]
    return model

def build_model(X_train_1,X_train_2,X_train_3,X_train_4,Y_train,X_test_1,X_test_2,X_test_3,X_test_4,Y_test,\
	Y_test_original,batch_size,epochs,a,time_lag):

	X_train_1 = X_train_1.reshape(X_train_1.shape[0],  53, time_lag-1, 3)
	X_train_2 = X_train_2.reshape(X_train_2.shape[0],  53, time_lag-1, 3)
	X_train_3 = X_train_3.reshape(X_train_3.shape[0],  53, time_lag-1, 1)
	X_train_4 = X_train_4.reshape(X_train_4.shape[0],  53, time_lag-1, 1)
	Y_train = Y_train.reshape(Y_train.shape[0], 53)

	print('X_train_1.shape is', X_train_1.shape)
	print('X_train_1.type is', type(X_train_1))
	print('X_train_2.shape is', X_train_2.shape)
	print('X_train_3.shape is', X_train_3.shape)
	print('Y_train.shape is', Y_train.shape)

	X_test_1 = X_test_1.reshape(X_test_1.shape[0],  53, time_lag-1, 3)#(1075,276,5,3),1075个时间段的276个站点的5个前后时间段的三个不同的维度
	X_test_2 = X_test_2.reshape(X_test_2.shape[0],  53, time_lag-1, 3)
	X_test_3 = X_test_3.reshape(X_test_3.shape[0],  53, time_lag-1, 1)
	X_test_4 = X_test_4.reshape(X_test_4.shape[0],  53, time_lag-1, 1)
	Y_test = Y_test.reshape(Y_test.shape[0], 53)

	if epochs == 50:
		model = multi_input_model(time_lag)
		model.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
		model.summary()
		model.fit([X_train_1, X_train_2, X_train_3, X_train_4], Y_train, batch_size=batch_size, epochs=epochs, verbose=2, shuffle=False)#, validation_split=0.05
		output = model.predict([X_test_1, X_test_2, X_test_3, X_test_4], batch_size=batch_size)
	else:
		# train models every 10 epoches
		model = load_model('testresult/'+str(epochs-10)+'-model-with-graph.h5')
		model.fit([X_train_1, X_train_2, X_train_3, X_train_4], Y_train, batch_size=batch_size, epochs=10, verbose=2, shuffle=False)# , validation_split=0.05
		output = model.predict([X_test_1, X_test_2, X_test_3, X_test_4], batch_size=batch_size)

	#rescale the output of this model将输出进行反归一化
	predictions = np.zeros((output.shape[0], output.shape[1]))
	for i in range(len(predictions)):
		for j in range(len(predictions[0])):
			predictions[i, j] = round(output[i, j]*a, 0)
			if predictions[i, j] < 0:
				predictions[i, j] = 0

	RMSE,R2,MAE,WMAPE=evaluate_performance(Y_test_original,predictions)
	#visualize the model structure
	plot_model(model, to_file='model.png', show_shapes=True)
	#print(model.summary())

	return model,Y_test_original,predictions,RMSE,R2,MAE,WMAPE

'''
def get_flops(model):
	run_meta = tf.compat.v1.RunMetadata()
	opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

	# We use the Keras session graph in the call to the profiler.
	flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph,
								run_meta=run_meta, cmd='op', options=opts)


	return flops.total_float_ops  # Prints the "flops" of the model.
'''

def Save_Data(path,model,Y_test_original,predictions,RMSE,R2,MAE,WMAPE,Run_epoch):
	print(Run_epoch)
	RMSE_ALL=[]
	R2_ALL=[]
	MAE_ALL=[]
	WMAPE_ALL=[]
	Average_train_time=[]
	RMSE_ALL.append(RMSE)
	R2_ALL.append(R2)
	MAE_ALL.append(MAE)
	WMAPE_ALL.append(WMAPE)
	model.save(path+str(Run_epoch)+'-model-with-graph.h5')
	np.savetxt(path+str(Run_epoch)+'-RMSE_ALL.txt', RMSE_ALL)
	np.savetxt(path+str(Run_epoch)+'-R2_ALL.txt', R2_ALL)
	np.savetxt(path+str(Run_epoch)+'-MAE_ALL.txt', MAE_ALL)
	np.savetxt(path+str(Run_epoch)+'-WMAPE_ALL.txt', WMAPE_ALL)
	with open(path+str(Run_epoch)+'-predictions.csv', 'w') as file:
		predictions = predictions.tolist()
		for i in range(len(predictions)):
			file.write(str(predictions[i]).replace("'", "").replace("[", "").replace("]", "")+"\n")
	with open(path+str(Run_epoch)+'-Y_test_original.csv', 'w') as file:
		Y_test_original = Y_test_original.tolist()
		for i in range(len(Y_test_original)):
			file.write(str(Y_test_original[i]).replace("'", "").replace("[", "").replace("]", "")+"\n")
	duration_time = time.time() - global_start_time
	Average_train_time.append(duration_time)
	np.savetxt(path+str(Run_epoch)+'-Average_train_time.txt', Average_train_time)
	print('total training time(s):', duration_time)




X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b,X_train_2,X_test_2,X_train_3,X_test_3,X_train_4,X_test_4=\
	Get_All_Data(TG=35, time_lag=7, TG_in_one_day=72, forecast_day_number=5, TG_in_one_week=360)
Run_epoch = 50  # first training 50 epoch, and then add 10 epoch every time 初始训练epoch，以后每次加10，运行15次
for i in range(1):
	model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE = build_model(X_train_1,X_train_2,X_train_3,X_train_4,Y_train,X_test_1,X_test_2,X_test_3,X_test_4,Y_test,\
		Y_test_original,batch_size=64,epochs=Run_epoch,a=a,time_lag=7)
	Save_Data("testresult/", model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, Run_epoch)
	Run_epoch += 10
	flops = get_flops(model, batch_size=1)
	print(f"FLOPS: {flops / 10 ** 9:.03} G")
