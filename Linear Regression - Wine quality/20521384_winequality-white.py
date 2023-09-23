from csv import reader
# Read csv file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
#  Convert string to float type
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
# Find min max each column
def min_max(dataset):
    a = list()       # Min-max array
    for i in range(len(dataset[0])):
        tempRow = [row[i] for row in dataset]
        Mini = min(tempRow)
        Maxi = max(tempRow)
        a.append([Mini, Maxi])
    return a
#  Minmax normalization
def normalization(dataset,a):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - a[i][0]) / (a[i][1] - a[i][0])

# train Linear Regression on 80% , Test on the remaining 20%
def d(sub_dataset, dataset_w, b):
    # sub_dataset=[a,b,c,..]
    # dataset_w=[a,b,c,...]
    # b
    s = 0
    for i in range(len(dataset_w)):
        s += sub_dataset[i] * dataset_w[i]
    return s + b


def loss(dataset, dataset_w, b):
    # dataset=[[a,b,c,...],[a,b,c,...],...]
    cols = len(dataset[0])
    L = 0
    for i in dataset:
        L += ((i[cols - 1] - d(i[0:cols - 1], dataset_w, b)) ** 2)
    return L / (2 * len(dataset))


def dw(dataset, dataset_w, b):
    cols = len(dataset[0])
    rows = len(dataset)
    data_dw = [0] * cols
    for i in range(rows):
        y = dataset[i][cols - 1]
        dt = d(dataset[i], dataset_w, b)
        data_dw[cols - 1] += float((y - dt) / rows)
        for j in range(cols - 1):
            data_dw[j] += float((y - dt) * dataset[i][j] / rows)
    return data_dw


def train(dataset):
    cols = len(dataset[0])
    # dataset_w=[0.1]*(cols-1)
    dataset_w = [0.07671997744611578, -0.29660623222809274, -0.010597330915987664, 0.8605186780845981,
                 -0.003016755422958975, 0.22722841907129784, -0.01553594902248619, -1.2728792274634806,
                 0.13617969125706925, 0.10761689133182632, 0.23186189403783738]
    # b=0
    b = 0.40826780683715586
    before = 0
    Loss = loss(dataset, dataset_w, b)
    distance = 0.000000000000001
    u = 0.1
    while abs(before - Loss) > distance:
        before = Loss
        data_dw = dw(dataset, dataset_w, b)
        for i in range(cols - 1):
            dataset_w[i] = dataset_w[i] + u * data_dw[i]
        b = b + u * data_dw[len(data_dw) - 1]
        Loss = loss(dataset, dataset_w, b)
        print(Loss)
    return (dataset_w, b)

filename = "winequality-white.csv"
dataset = load_csv(filename)
#print('Loaded data file {0} with {1} rows and {2} column'.format(filename, len(dataset), len(dataset[0])))
#print(dataset[0])
a = list()
datatrain=list()
datatest=list()

for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
a = min_max(dataset)
#print(a)
normalization(dataset,a)
#print(dataset)
#  split data into 80% and 20%
for i in range(len(dataset)):
    if i <= ((len(dataset)*0.8)):
        datatrain.append(dataset[i])
    else:
        datatest.append(dataset[i])
#print(datatest,datatrain)
x=train(datatrain)
#print(x)