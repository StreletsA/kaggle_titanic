from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

INPUT_DATA_TRAIN_LEN = 5
OUTPUT_DATA_TRAIN_LEN = 1

INPUT_NEURONS_COUNT = 5
HIDDEN_NEURONS_COUNT = 10
OUTPUT_NEURONS_COUNT = 1

EPOCH_COUNT = 500

FILE_OUTPUT = 'answers.csv'

ds = SupervisedDataSet(INPUT_DATA_TRAIN_LEN, OUTPUT_DATA_TRAIN_LEN)
net = buildNetwork(INPUT_NEURONS_COUNT, HIDDEN_NEURONS_COUNT, OUTPUT_NEURONS_COUNT)


def save_answers(answers):
    with open(FILE_OUTPUT, 'w') as f:
        lines = []
        for i in answers.keys():
            if answers[i][0] >= 0.5:
                lines.append(str(i) + ',' + '1\n')
            else:
                lines.append(str(i) + ',' + '0\n')
        f.writelines(lines)
        f.close()


def get_test_data(path):
    data = []

    with open(path) as file:
        lines = [i.split(',') for i in file.read().splitlines()]

    lines = lines[1:]

    c = 1
    for line in lines:
        c += 1
        pclass = int(line[1])
        sex = line[4]

        if line[4] == '':
            status = line[3].split(' ')[1]

            if status in ['Mr.', 'Master.', 'Dr.']:
                sex = 0
            else:
                sex = 1
        elif sex == 'male':
            sex = 0
        elif sex == 'female':
            sex = 1

        if line[5] == '':
            age = 0
        else:
            age = float(line[5])
        siblings = int(line[6])
        parch = int(line[7])
        inputs = list([pclass, sex, age, siblings, parch])

        data.append(inputs)

    return data


def get_train_data(path):

    data = []

    with open(path) as file:
        lines = [i.split(',') for i in file.read().splitlines()]

    lines = lines[1:]

    c = 1
    for line in lines:
        c += 1

        ans = int(line[1])
        pclass = int(line[2])
        sex = line[5]

        if line[5] == '':
            status = line[4].split(' ')[1]

            if status in ['Mr.', 'Master.', 'Dr.']:
                sex = 0
            else:
                sex = 1

        elif sex == 'male':
            sex = 0
        elif sex == 'female':
            sex = 1

        if line[6] == '':
            age = 0
        else:
            age = float(line[6])
        siblings = int(line[7])
        parch = int(line[8])

        inputs = list([pclass, sex, age, siblings, parch])
        output = list([ans])

        data.append([inputs, output])

    return data


if __name__ == '__main__':

    train_data = get_train_data('train.csv')
    test_data = get_test_data('test.csv')

    for i in train_data:
        ds.addSample(i[0], i[1])

    trainer = BackpropTrainer(net, ds)
    trainer.trainEpochs(EPOCH_COUNT)

    answers = dict()

    c = 1
    for i in test_data:
        answers[c] = net.activate(i)
        c += 1

    save_answers(answers)