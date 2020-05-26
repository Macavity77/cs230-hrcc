#!/usr/bin/env python
# coding: utf-8

# In[2]:


#data source: HWDB1.1 - 3,755 GB2312-80 level-1 Chinese characters and 171 alphanumeric and symbols.
#300 writers, total -- 1172907, symbol -- 51158, Chinese -- 1121749, class -- 3755

import math
#from pycasia import CASIA
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)


# In[3]:



print(tf.__version__)


# In[170]:





# In[14]:


import os
import numpy as np
import struct
from PIL import Image

    
data_dir = '/Users/zewang/CASIA_data'
# train_data_dir = "./CASIA_data/HWDB1.1trn_gnt"
train_data_dir = os.path.join(data_dir, 'HWDB1.1trn1')
dev_data_dir = os.path.join(data_dir, 'HWDB1.1trn2')
test_data_dir = os.path.join(data_dir, 'HWDB1.1tst')


def read_from_gnt_dir(gnt_dir=train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size: break
            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)
            tagcode = header[5] + (header[4]<<8)
            width = header[6] + (header[7]<<8)
            height = header[8] + (header[9]<<8)
            if header_size + width*height != sample_size:
                break
            image = np.fromfile(f, dtype='uint8', count=width*height).reshape((height, width))
            yield image, tagcode
    for file_name in os.listdir(gnt_dir):
        print(file_name)
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image, tagcode in one_file(f):
                    yield image, tagcode
char_set = set()
for _, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
    char_set.add(tagcode_unicode)
char_list = list(char_set)
char_dict = dict(zip(sorted(char_list), range(len(char_list))))
print(len(char_dict)) #0-3925 共3926
print(char_dict)
import pickle
f = open('char_dict', 'wb')
pickle.dump(char_dict, f)
f.close()
train_counter = 0
test_counter = 0
for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
    print(train_counter)
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
    im = Image.fromarray(image)
    dir_name = '/Users/zewang/CASIA_data/train/' + '%0.5d'%char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    im.convert('RGB').save(dir_name+'/' + str(train_counter) + '.png')
    train_counter += 1
for image, tagcode in read_from_gnt_dir(gnt_dir=test_data_dir):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
    im = Image.fromarray(image)
    dir_name = '/Users/zewang/CASIA_data/test/' + '%0.5d'%char_dict[tagcode_unicode]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    im.convert('RGB').save(dir_name+'/' + str(test_counter) + '.png')
    test_counter += 1


# In[15]:





# In[17]:



print(train_counter) #62566
print(test_counter) #11670


# In[18]:


import scipy
import imageio
from PIL import Image

def process_png(path):
    image = np.array(imageio.imread(path))
    my_image = np.array(Image.fromarray(image).resize(size=(64,64))).reshape((1, 64*64*3)).T
    return my_image


# In[20]:


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name = "C")
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, C, 1.0)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix).T
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    
    return one_hot


# In[85]:


import glob, os
def load_data_set(char_dict):
    X_train = np.zeros((64 * 64 *3, 1))
    #print(X_train.shape)
    Y_train = [0]
    X_test = np.zeros((64 * 64 *3, 1))
    Y_test = [0]
    for i in range(0, len(char_dict)):
        dir_name = '/Users/zewang/CASIA_data/train/' + '%0.5d'%i
        os.chdir(dir_name)
        print(dir_name)
        for file in glob.glob("*.png"):
            print(file)
            X_train = np.append(X_train, process_png(file), 1)
            Y_train = np.append(Y_train, i)
            break #单纯为了加速
        dir_name = '/Users/zewang/CASIA_data/test/' + '%0.5d'%i
        os.chdir(dir_name)
        print(dir_name)
        for file in glob.glob("*.png"):
            print(file)
            X_test = np.append(X_test, process_png(file), 1)
            Y_test = np.append(Y_test, i)
            break #单纯为了加速
    print(type(Y_train[0]))
    X_train = X_train[:, 1:]
    Y_train_hot = one_hot_matrix(Y_train, len(char_dict))
    Y_train_hot = Y_train_hot[:,1:]
    print(X_train.shape)
    print(Y_train_hot.shape)
        
    X_test = X_test[:, 1:]
    Y_test_hot = one_hot_matrix(Y_test, len(char_dict))
    Y_test_hot = Y_test_hot[:,1:]
    print(X_test.shape)
    print(Y_test_hot.shape)
    return X_train, Y_train_hot, X_test, Y_test_hot, len(char_dict)


# In[22]:


X_train, Y_train, X_test, Y_test, classes = load_data_set(char_dict) #classes这个好像还不是很确定


# In[ ]:


# Example of a picture


# In[38]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[39]:


print(X_train[:,0]) #first case
print(X_train[2000:2020,0])


# In[40]:


# Flatten the training and test images
# X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
# X_test_flatten = X_test.reshape(X_test.shape[0], -1).T
# Normalize image vectors
X_train_nor = X_train / 255.
X_test_nor = X_test / 255.

# Y_train = convert_to_one_hot(Y_train_orig, classes)
# Y_test = convert_to_one_hot(Y_test_orig, classes)

print ("number of training examples = " + str(X_train_nor.shape[1]))
print ("number of test examples = " + str(X_test_nor.shape[1]))
print ("X_train shape: " + str(X_train_nor.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test_nor.shape))
print ("Y_test shape: " + str(Y_test.shape))


# In[41]:


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name = "Placeholder_1")
    Y = tf.placeholder(tf.float32, [n_y, None], name = "Placeholder_2")
    return X, Y


# In[50]:


def initialize_parameters():
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    #这个地方的parameter是不对的
    W1 = tf.get_variable("W1", [4000,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [4000,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [3000,4000], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [3000,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [3926,3000], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [3926,1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


# In[ ]:





# In[51]:


def forward_propagation(X, parameters):
    #Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, tf.cast(X, tf.float32)), b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, A1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                                              # Z3 = np.dot(W3, A2) + b3
    ### END CODE HERE ###
    
    return Z3


# In[52]:


#tf.compat.v1.reset_default_graph()


# In[53]:


# GRADED FUNCTION: compute_cost 

def compute_cost(Z3, Y):
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    ### END CODE HERE ###
    
    return cost


# In[54]:


tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))


# In[55]:


from tensorflow.python.framework import ops


# In[58]:


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# In[69]:


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.01,
          num_epochs = 100, minibatch_size = 128, print_cost = True):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    print(n_x)
    print(n_y)
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X_train, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y_train)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    
    #ada_optimizer = tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(cost)
    #rmsprop_optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):
            
            print("running on epoch:" + str(epoch))

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters


# In[70]:


parameters = model(X_train_nor, Y_train, X_test_nor, Y_test)


# In[71]:


def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction


# In[79]:


def forward_propagation_for_predict(X, parameters):
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3


# In[93]:


import scipy
import imageio
from PIL import Image

image = np.array(imageio.imread("/Users/zewang/CASIA_data/test/01414/10306.png"))
print(image.shape)
my_image = np.array(Image.fromarray(image).resize(size=(64,64))).reshape((1, 64*64*3)).T
print(my_image.shape)
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




