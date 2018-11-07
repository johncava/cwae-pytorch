

def create_dataset():
    dataset = []

    with open('mnist_train.csv', 'r') as f:
        for line in f:
            line = line.strip('\n').split(',')
            x = float(line[0])
            y = [float(l) for l in line[1:]]
            dataset.append([x,y])

    return dataset
