x =[2, 1, 5, 3, 4]
y =[9, 5, 21, 13, 17]
w, b = 10, 20
m = 5
while True:
    Loss, wNew, bNew = 0, 0, 0
    for i in range(5):
        Loss += ((w*x[i] + b - y[i])**2)/(2*m)
    if Loss <= 0.00000001:
        break
    for i in range(5):
        wNew += (w * x[i] + b - y[i]) * x[i]
        bNew += (w * x[i] + b - y[i]) / m
    w -= 0.0001*wNew
    b -= 0.0001*bNew
print("y = " + str(w) + "x + "+ str(b))
#import matplotlib.pyplot as plt
#plt.plot([1,2,3],[2,3,4])
#plt.show()