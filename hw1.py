import numpy as np
import gzip
import NN_layers
import matplotlib.pyplot as plt
import pdb
import os
#fashion mnist dataset path      
url_train_image = 'Fashion_MNIST_data/train-images-idx3-ubyte.gz'
url_train_labels = 'Fashion_MNIST_data/train-labels-idx1-ubyte.gz'
url_test_image = 'Fashion_MNIST_data/t10k-images-idx3-ubyte.gz'
url_test_labels = 'Fashion_MNIST_data/t10k-labels-idx1-ubyte.gz'

#use gzip open .gz to get ubyte
train_image_ubyte = gzip.open(url_train_image,'r')
test_image_ubyte = gzip.open(url_test_image,'r')
train_label_ubyte = gzip.open(url_train_labels,'r')
test_label_ubyte = gzip.open(url_test_labels,'r')

direct_path = r"D:\碩一上修課資料\深度學習_林嘉文\Fashion_HM_1"

##START YOUR CODE
class Get_mnist_data:
    def load_data(self , type : str):
        if type == 'train':
            data = np.frombuffer(train_image_ubyte.read(), dtype=np.uint8 , offset = 16)
            return data.reshape(-1, 28, 28)
        elif type == 'train_label':
            data = np.frombuffer(train_label_ubyte.read(), dtype=np.uint8 , offset = 8)
            return data   
        elif type == 'test':
            data = np.frombuffer(test_image_ubyte.read(), dtype=np.uint8 , offset = 16)
            return data.reshape(-1, 28, 28)
        elif type == 'test_label':
            data = np.frombuffer(test_label_ubyte.read(), dtype=np.uint8 , offset = 8)
            return data  
obj = Get_mnist_data()
dataset = {
    'training':{
        "data" : obj.load_data('train')
        ,"label" : obj.load_data('train_label')
                }
    ,"testing":{
        "data" : obj.load_data('test')
        ,"label" : obj.load_data('test_label')
                }
            }
# print(dataset["training"]["data"].shape)
# print(dataset["training"]["label"].shape)
# print(dataset["testing"]["data"].shape)
# print(dataset["testing"]["label"].shape)

# for i in range(100):
#     first_image = dataset["training"]["data"][i]
#     first_image = np.array(first_image, dtype='float')
#     pixels = first_image.reshape((28, 28))
#     plt.imshow(pixels, cmap='gray')
#     plt.show()

#normalization
dataset['training']['data'] = dataset['training']['data'] / 255
dataset['testing']['data'] = dataset['testing']['data'] / 255

#validation split
valid_split = 54000         #(9:1)
valid = {'data' : dataset['training']['data'][valid_split : ] , 'label' : dataset['training']['label'][valid_split : ]}
dataset.update({'valid' : valid})
dataset['training']['data'] = dataset['training']['data'][ : valid_split]
dataset['training']['label'] = dataset['training']['label'][ : valid_split]
print("training data shape :",dataset['training']['data'].shape , dataset['training']['label'].shape)
print("validation data shape :" , dataset['valid']['data'].shape , dataset['valid']['label'].shape)
print("testing data shape :" , dataset['testing']['data'].shape , dataset['testing']['label'].shape)

batch_size = 32
epoch = 50
LR = 0.003
LR_min = LR/10
lr_distance = (LR - LR_min) / (epoch - 20)

#construct layers
flatten = NN_layers.flatten()
fc_layer1 = NN_layers.full_connected(28*28 , 256 , 0.9)
relu1 = NN_layers.relu()
fc_layer2 = NN_layers.full_connected(256 , 128 , 0.9)
relu2 = NN_layers.relu()
fc_layer3 = NN_layers.full_connected(128 , 10 , 0.9)
softmax = NN_layers.softmax()
crs_etp = NN_layers.cross_entropy()

#deep NN layers
fc_layer4 = NN_layers.full_connected(256, 256, 0.9)
relu3 = NN_layers.relu()
fc_layer7 = NN_layers.full_connected(128, 128, 0.9)
relu6 = NN_layers.relu()

#another activate function:
sigmoid_1 = NN_layers.sigmoid()
sigmoid_2 = NN_layers.sigmoid()

#another activate function:
tanh_1 = NN_layers.tanh()
tanh_2 = NN_layers.tanh()

def train(data_input , label_input): # [32, 28, 28]
    accuracy , loss , predict = pass_forward(data_input , label_input)
    grad_out = softmax.backward(label_input)
    grad_out = fc_layer3.backward(grad_out, LR)
    
    # grad_out = relu6.backward(grad_out)
    # grad_out = fc_layer7.backward(grad_out, LR)

    grad_out = relu2.backward(grad_out)
    # grad_out = sigmoid_2.backward(grad_out)
    # grad_out = tanh_2.backward(grad_out)
    grad_out = fc_layer2.backward(grad_out, LR)


    # grad_out = relu3.backward(grad_out)
    # grad_out = fc_layer4.backward(grad_out, LR)

    grad_out = relu1.backward(grad_out)
    # grad_out = sigmoid_1.backward(grad_out)
    # grad_out = tanh_1.backward(grad_out)
    grad_out = fc_layer1.backward(grad_out , LR)
    return accuracy , loss , predict
    
def pass_forward(data_input , label_input):
    output = flatten.forward(data_input)
    output = fc_layer1.forward(output)
    output = relu1.forward(output)
    # output = sigmoid_1.forward(output)
    # output = tanh_1.forward(output)

    # output = fc_layer4.forward(output)      #deep
    # output = relu3.forward(output)


    output = fc_layer2.forward(output)
    output = relu2.forward(output)
    # output = sigmoid_2.forward(output)
    # output = tanh_2.forward(output)

    # output = fc_layer7.forward(output)      #deep
    # output = relu6.forward(output)

    output = fc_layer3.forward(output)
    output = softmax.forward(output)
    accuracy , loss , predict = crs_etp.forward(output, label_input)
    return accuracy , loss , predict
    
def plot_comparison(train, valid, test , item):
    epoch_len = len(train)
    x_axis = np.array([i+1 for i in range(epoch_len)])
    plt.xlabel("Epoch")

    plt.plot(x_axis , train)
    plt.plot(x_axis , valid)
    plt.plot(x_axis , test)

    pic_name = ''
    if item == "accuracy":
        pic_name = "linear_LR_Accuracy_50_batch_32.png"
        plt.ylabel("Accuracy")
    else:
        pic_name = "linear_LR_Loss_50_batch_32.png"
        plt.ylabel("Loss")
    plt.title(item)
    plt.legend(['train', 'valid' , 'test'], loc="best")
    plt.savefig(pic_name)
    plt.close()

plot_train_loss , plot_train_acc = [] , []
plot_valid_loss , plot_valid_acc = [] , []
plot_test_loss , plot_test_acc = [] , []

for i in range(1, epoch + 1):
    print("current EPOCH :" , i)
    print("linear_LR : " , LR)
    #/////////////////////////////////training//////////////////////////////////////
    train_loss_per_batch , train_acc_per_batch = [] , []
    for batch_idx in range(int((dataset['training']['data'].shape[0]) / batch_size)):
        batch_start_idx = batch_idx * batch_size
        if batch_idx * batch_size > dataset['training']['data'].shape[0]:
            batch_end_idx = dataset['training']['data'].shape[0]
        else:
            batch_end_idx = (batch_idx + 1) * batch_size
        train_data_input = dataset['training']['data'][batch_start_idx : batch_end_idx]
        train_label_input = dataset['training']['label'][batch_start_idx : batch_end_idx]

        train_data_input = np.expand_dims(train_data_input , axis=1)
        train_label_input = np.eye(10)[train_label_input]       #one hot encoding

        accuracy , loss , predict = train(train_data_input, train_label_input)
        train_acc_per_batch.append(accuracy)
        train_loss_per_batch.append(loss)

        if len(train_acc_per_batch)%100 == 0:
            print("Training : " , (batch_idx + 1)*batch_size , np.sum(train_loss_per_batch)/(len(train_loss_per_batch) * batch_size),
                                                                np.sum(train_acc_per_batch)/(len(train_acc_per_batch) * batch_size)) 
    train_loss_per_epoch = np.sum(train_loss_per_batch)/(len(train_loss_per_batch) * batch_size)
    train_acc_per_epoch = np.sum(train_acc_per_batch)/(len(train_acc_per_batch) * batch_size)
    print("training Loss : " , train_loss_per_epoch)
    print("training accuracy : " , train_acc_per_epoch)
    plot_train_loss.append(train_loss_per_epoch)
    plot_train_acc.append(train_acc_per_epoch)

    #linear LR changing
    if i > 20:
        LR -= lr_distance
    #/////////////////////////////////validation//////////////////////////////////////
    valid_loss_per_batch , valid_acc_per_batch = [] , []
    for batch_idx in range(int((dataset['valid']['data'].shape[0]) / batch_size)):
        batch_start_idx = batch_idx * batch_size
        if batch_idx * batch_size > dataset['valid']['data'].shape[0]:
            batch_end_idx = dataset['valid']['data'].shape[0]
        else:
            batch_end_idx = (batch_idx + 1) * batch_size
        valid_data_input = dataset['valid']['data'][batch_start_idx : batch_end_idx]
        valid_label_input = dataset['valid']['label'][batch_start_idx : batch_end_idx]

        valid_data_input = np.expand_dims(valid_data_input , axis=1)
        valid_label_input = np.eye(10)[valid_label_input]

        accuracy , loss , predict = pass_forward(valid_data_input, valid_label_input)
        valid_acc_per_batch.append(accuracy)
        valid_loss_per_batch.append(loss)

        if len(valid_acc_per_batch)%50 == 0:
            print("validation : " , (batch_idx + 1)*batch_size , np.sum(valid_loss_per_batch)/(len(valid_loss_per_batch) * batch_size),
                                                                np.sum(valid_acc_per_batch)/(len(valid_acc_per_batch) * batch_size))
    valid_loss_per_epoch = np.sum(valid_loss_per_batch)/(len(valid_loss_per_batch) * batch_size)
    valid_acc_per_epoch = np.sum(valid_acc_per_batch)/(len(valid_acc_per_batch) * batch_size)
    print("validation Loss : " , valid_loss_per_epoch)
    print("validation accuracy : " , valid_acc_per_epoch)
    plot_valid_loss.append(valid_loss_per_epoch)
    plot_valid_acc.append(valid_acc_per_epoch)

    #/////////////////////////////////testing//////////////////////////////////////
    test_loss_per_batch , test_acc_per_batch = [] , []
    for batch_idx in range(int((dataset['testing']['data'].shape[0]) / batch_size)):
        batch_start_idx = batch_idx * batch_size
        if batch_idx * batch_size > dataset['testing']['data'].shape[0]:
            batch_end_idx = dataset['testing']['data'].shape[0]
        else:
            batch_end_idx = (batch_idx + 1) * batch_size
        test_data_input = dataset['testing']['data'][batch_start_idx : batch_end_idx]
        test_label_input = dataset['testing']['label'][batch_start_idx : batch_end_idx]

        test_data_input = np.expand_dims(test_data_input , axis=1)
        test_label_input = np.eye(10)[test_label_input]

        accuracy , loss , predict = pass_forward(test_data_input, test_label_input)
        test_acc_per_batch.append(accuracy)
        test_loss_per_batch.append(loss)

        if len(test_acc_per_batch)%100 == 0:
            print("testing : " , (batch_idx + 1)*batch_size , np.sum(test_loss_per_batch)/(len(test_loss_per_batch) * batch_size),
                                                                np.sum(test_acc_per_batch)/(len(test_acc_per_batch) * batch_size))
    test_loss_per_epoch = np.sum(test_loss_per_batch)/(len(test_loss_per_batch) * batch_size)
    test_acc_per_epoch = np.sum(test_acc_per_batch)/(len(test_acc_per_batch) * batch_size)
    print("testing Loss : " , test_loss_per_epoch)
    print("testing accuracy : " , test_acc_per_epoch)
    plot_test_loss.append(test_loss_per_epoch)
    plot_test_acc.append(test_acc_per_epoch)

    with open(os.path.join(direct_path, "ReLU.txt"), 'a') as fp:
        fp.write(" Epoch {:2d} | Train acc. {:.4f} | Train loss {:.4f} | Val acc. {:.4f} | Val loss {:.4f} | test_acc {:.4f} | test loss {:.4f}\n".format(
            i,
            train_acc_per_epoch,
            train_loss_per_epoch,
            valid_acc_per_epoch,
            valid_loss_per_epoch,
            test_acc_per_epoch,
            test_loss_per_epoch
        ))
    fp.close()

with open('weights_result.npy', 'wb') as f:
    np.save(f, fc_layer1.weights)
    np.save(f, fc_layer1.bias)
    np.save(f, fc_layer2.weights)
    np.save(f, fc_layer2.bias)
    np.save(f, fc_layer3.weights)
    np.save(f, fc_layer3.bias)

plot_comparison(plot_train_acc , plot_valid_acc , plot_test_acc , "accuracy")
plot_comparison(plot_train_loss , plot_valid_loss , plot_test_loss , 'loss')





