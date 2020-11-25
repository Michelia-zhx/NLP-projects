import matplotlib.pyplot as plt
train_loss = []
train_loss_file = open("outputs/output1.txt", encoding='utf-8')
for line in train_loss_file.readlines():
    if line.split(' ')[0] == "Training":
        train_loss.append(eval(line.split(' ')[-1]))

x = range(40)
plt.figure()
plt.xlabel("epoch")
plt.ylabel("training loss")
plt.plot(x, train_loss, alpha = 0.7)
plt.show()