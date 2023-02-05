import numpy as np

with open("weights_result.npy" , "rb") as f:
    fc_layer1_weights = np.load(f)
    fc_layer1_bias = np.load(f)
    fc_layer2_weights = np.load(f)
    fc_layer2_bias = np.load(f)
    fc_layer3_weights = np.load(f)
    fc_layer3_bias = np.load(f)

    print("fc_layer1_weights shape : ", fc_layer1_weights.shape)
    print("fc_layer1_bias shape : ", fc_layer1_bias.shape)
    print("fc_layer2_weights shape : ", fc_layer2_weights.shape)
    print("fc_layer2_bias shape : ", fc_layer2_bias.shape)
    print("fc_layer3_weights shape : ", fc_layer3_weights.shape)
    print("fc_layer3_bias shape : ", fc_layer3_bias.shape)    

    print("fc_layer1_weights : ", fc_layer1_weights)
    print("fc_layer1_bias : ", fc_layer1_bias)
    print("fc_layer2_weights : ", fc_layer2_weights)
    print("fc_layer2_bias : ", fc_layer2_bias)
    print("fc_layer3_weights : ", fc_layer3_weights)
    print("fc_layer3_bias : ", fc_layer3_bias)    

