from csv import reader
import numpy as np
import math

# Load file CSV
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Transfer string column into float column
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Find maxmin in dataset
def maxmin(data):
    Max,Min = list(),list()
    for i in range(13):
        Max.append(np.max(data[:, i]))
        Min.append(np.min(data[:, i]))
    return Max, Min

# Split 80% for training and remaining for testing
def splitData(dataset):
    datatrain = dataset[:473]
    datatest = dataset[473:]
    return datatrain, datatest

# Minmax nomalization
def normalization(dataset,Max,Min):
    for i in range(13):
        for row in dataset:
            row[i] = (row[i] - Min[i]) / (Max[i]- Min[i])
def count(data):
    c1,c2 = 0, 0
    for i in data:
        if i[13] == 1: c1 += 1
        else: c2 += 1
    return c1, c2

#Theta Function
def Theta(data, array_w, b):
    theta = 0
    for i in range(13):
        theta += array_w[i] * data[i]
    return theta + b

#Sigmoid Function
def Sigmoid(Theta):
    return 1 / (1 + math.exp(-Theta))

#Loss Function
def cost(array_w, b, data):
    loss = 0
    c1,c2=count(data)
    w0=(c1+c2)/(2*c2)
    w1=(c1+c2)/(2*c1)
    for i in data:
        loss += (w1*-i[13] * math.log(Sigmoid(Theta(i, array_w, b))) - (w0*(1 - i[13]) * math.log(1 - Sigmoid(Theta(i, array_w, b)))))
    return loss / len(data)

#Dao ham Loss
def derive(data, array_w, b):
    db = 0
    dw = [0]*13
    c1,c2=count(data)
    for i in data:
        if i[13] == 1: wr = (c1+c2)/(2*c1)
        else: wr = (c1+c2)/(2*c2)
        sigmoid = Sigmoid(Theta(i, array_w, b))
        for j in range(len(i) - 1):
            dw[j] += ((sigmoid - i[13]) * i[j] * wr)
        db += (sigmoid - i[13])* wr
    return dw,db

#Divide Function
def divide(p):
    if p >= 0.5:
        return 1
    else:
        return 0

#Accuracy
def accuracy0(data, array_w, b):
    count = 0
    for i in data:
        predict = divide(Sigmoid(Theta(i, array_w, b)))
        if i[13] == predict:
            count += 1
    return count / len(data) * 100

def accuracy1(data, array_w, b):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in data:
        predict = divide(Sigmoid(Theta(i, array_w, b)))
        if i[13] == 1 and predict == 1: tp+= 1
        elif i[13] == 1 and predict == 0: fn += 1
        elif i[13] == 0 and predict == 1: fp += 1
        elif i[13] == 0 and predict == 0: tn += 1
    accuracy = (tp+tn) / (tp + tn + fp + fn)
    return tp, fn, fp, tn, accuracy

def Sum(tp,fp,fn ):
    Precision= tp / (tp + fp)
    Recall= tp / (tp + fn)
    F1 = (2 * Recall * Precision) / (Recall + Precision)
    print("Precision: ",Precision)
    print("Recall: ", Recall)
    print("F1: ", F1)

def LogisticRegression(datatrain, datatest):
    wm = 0.01
    b = 1
    array_w = [0]*13
    while True:
        loss = cost(array_w, b, datatrain)
        print(loss)
        if loss < 0.437:
            break
        dw, db = derive(datatrain, array_w, b)
        for i in range(13):
            array_w[i] -=  wm * dw[i]/len(datatrain)
        b = b - wm * db/len(datatrain)
        print(round(accuracy0(datatest,array_w,b),2))
    tp,fn,fp,tn,Accuracy = accuracy1(datatest, array_w, b)
    print( "Accuracy: ",Accuracy)
    Sum(tp,fn,fp)


filename = 'cleveland-heart.csv'
dataset = load_csv(filename)
dataset = np.array(dataset, dtype = 'float')
datatrain,datatest =splitData(dataset) # Split data
Max , Min =maxmin(datatrain)    # Find max,min
normalization(datatest,Max,Min)  # Minmax nomalization
normalization(datatrain,Max,Min)
LogisticRegression(datatrain, datatest)