from typing import Dict, List
from matplotlib import pyplot as plt


def get_data_points(text_file: str) -> List[Dict]:
    data_points = []

    with open(text_file) as data_file:
        for _ in range(3):
            dice_scores = []
            name = data_file.readline().strip('\n')

            for i in range(10):
                dice_scores.append(float(data_file.readline()))

            data_points.append({'Name' : name, "Dice Scores" : dice_scores})


    return data_points


if __name__ == '__main__':
    results = get_data_points('data_points.txt')

    for result in results:
        plt.plot([epoch for epoch in range(1, 11)], result['Dice Scores'], label = result['Name'])

    plt.title('Dice Score - Validation Data')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.show()