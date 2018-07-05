import input_data
import directory

train_dir = directory.train_dir
train, train_label = input_data.get_files(train_dir)

#print(train_label)
#print(train) #Disable as it prints too much stuff.

for i in range(0, 20):
    print("Label:" + str(train_label[i]))
    print("Image:" + str(train[i]) + "\n")



