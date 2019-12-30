
import numpy as np
import sys

train = np.genfromtxt(sys.argv[1], delimiter=',', dtype=int)
test = np.genfromtxt(sys.argv[2], delimiter=',', dtype=int)
num_epochs = int(sys.argv[6])
lr = float(sys.argv[9])
h_units = int(sys.argv[7])
init_flag = int(sys.argv[8])

#initializing parameters
if init_flag == 1:
    alpha1 = np.random.uniform(low=-0.1, high=0.1, size=(h_units, train.shape[1]-1))
    alpha0 = np.zeros((h_units,1))
    alpha = np.hstack((alpha0, alpha1))
    beta1 = np.random.uniform(low=-0.1, high=0.1, size=(10, h_units))
    beta0 = np.zeros((10,1))
    beta = np.hstack((beta0, beta1))
if init_flag == 2:
    alpha = np.zeros((h_units, train.shape[1]))
    beta = np.zeros((10,h_units+1))

    
def softmax(B):
    return np.divide(np.exp(B),np.sum(np.exp(B)))
        
def cross_entropy(y, y_hat):
    return -np.dot(y.T,np.log(y_hat))


def NNforward(input, alpha, beta):
    X = np.copy(input)
    y = X[0]
    Y = np.zeros((10,1), dtype = int)
    Y[y] = 1
    X[0] = 1
    X = X.reshape(-1,1)
    A = np.dot(alpha, X).reshape(-1,1)
    Z = 1/(1+np.exp(-A)).reshape(-1,1)
    Z = np.append(1, Z).reshape(-1,1)
    B = np.dot(beta, Z).reshape(-1,1)
    Y_hat = softmax(B)
    J = cross_entropy(Y, Y_hat)[0][0]
    return A, Z, B, Y_hat, J


def NNBackward(input, alpha, beta, A, Z, B, Y_hat, J):
    X = np.copy(input)
    y = X[0]
    Y = np.zeros((10,1), dtype = int)
    Y[y] = 1
    X[0] = 1
    X = X.reshape(-1,1)
    G_b = Y_hat - Y
    G_beta = np.dot(G_b, Z.T)
    beta_star = beta[:, 1:]
    Z_star = Z[1:]
    G_z = np.dot(beta_star.T, G_b).reshape(-1,1)
    G_a = np.array([Z_star[i]*(1-Z_star[i])*G_z[i] for i in range(len(Z_star))]).reshape(-1,1)
    G_alpha = np.dot(G_a, X.T)
    return G_alpha, G_beta

def learn(train, test, alpha, beta):
    ce_train_list = []
    ce_test_list = []
    for i in range(num_epochs):
        for j in range(train.shape[0]):
            A, Z, B, Y_hat, J = NNforward(train[j], alpha, beta)
            G_alpha, G_beta = NNBackward(train[j], alpha, beta, A, Z, B, Y_hat, J)
            alpha = alpha - lr*G_alpha
            beta = beta - lr*G_beta
        
        ce1 = 0   
        for k in range(train.shape[0]):
            ce1 += NNforward(train[k], alpha, beta)[4]
        ce_train_list.append(ce1/train.shape[0])
            
        ce2 = 0   
        for l in range(test.shape[0]):
            ce2 += NNforward(test[l], alpha, beta)[4]
        ce_test_list.append(ce2/test.shape[0])
    return alpha, beta, ce_train_list, ce_test_list

def predict(input, alpha, beta):
    Label = []
    for i in range(input.shape[0]):
        Y_hat = NNforward(input[i], alpha, beta)[3]
        label = np.where(Y_hat == np.max(Y_hat))[0][0]
        Label.append(label)
    Label1 = np.array(Label)
    true_label = input[:,0]
    count = 0
    for j in range(len(Label)):
        if (Label1[j] != true_label[j]):
            count+=1
    error = count/len(Label)
    return Label, error


alpha, beta, ce_train_list, ce_test_list = learn(train, test, alpha, beta)


train_label, train_error = predict(train, alpha, beta)
test_label, test_error = predict(test, alpha, beta)

file4 = open(sys.argv[5], "w")
for i in range(num_epochs):
    file4.writelines("epoch=" + str(i+1) + " crossentropy(train): " + str(ce_train_list[i]) + "\n")
    file4.writelines("epoch=" + str(i+1) + " crossentropy(test): " + str(ce_test_list[i]) + "\n")
file4.writelines("error(train): " + str(train_error) + "\n")
file4.writelines("error(test): " + str(test_error))
file4.close()

file5 = open(sys.argv[3], "w")
for i in range(len(train_label)):
    file5.writelines(str(train_label[i]) + "\n")
file5.close()

file6 = open(sys.argv[4], "w")
for i in range(len(test_label)):
    file6.writelines(str(test_label[i]) + "\n")
file6.close()





