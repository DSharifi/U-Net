from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


train_imgs = 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip'
train_masks = 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip'

test_imgs = 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip'
test_masks = 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip'

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_path", action='store', type=str, required=True)
    data_path = parser.parse_args().data_path

    print(data_path)


    for zipurl in (train_masks, train_imgs, test_imgs, test_masks):
        with urlopen(zipurl) as zip_response:
            with ZipFile(BytesIO(zip_response.read())) as zfile:
                zfile.extractall(data_path)