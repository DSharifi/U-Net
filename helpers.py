import matplotlib.pyplot as plt

def plot_imgs(img_left, img_right):
    _, ax = plt.subplots(1,2, constrained_layout=True, figsize= (10,10))

    ax[0].set_title('Input Image')
    ax[0].imshow(img_left)

    ax[1].set_title('Output Image')
    ax[1].imshow(img_right)

    plt.show()
