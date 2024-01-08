import pandas as pd

#Describe train csv data
def describe_train_csv():
    #Load data from train.csv
    train_data = pd.read_csv('./model/train.csv')
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    train_data.head()

    #Describe the csv file
    print(train_data.describe())

#Describe test csv data
def describe_test_csv():
    #Load data from test.csv
    test_data = pd.read_csv('./model/test.csv')
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    test_data.head()

    #Describe the csv file
    print(test_data.describe())

print('Train data:')
describe_train_csv()
print('Test data:')
describe_test_csv()