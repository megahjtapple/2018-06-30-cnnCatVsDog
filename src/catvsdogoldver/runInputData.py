import input_data

train_dir = '/userDocs/user000/workspaces/2018-06-30-tensorflowCNN/Data/catVsDog/train/'
train, train_label = input_data.get_files(train_dir)

#print(train_label)
#print(train) #Disable as it prints too much stuff.

for i in range(0, 4):
    print("Label:" + str(train_label[i]))
    print("Image:" + str(train[i]) + "\n")



