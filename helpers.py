import matplotlib.pyplot as plt

def crop_feature_map(tensor, target_height, target_width):

    curr_height = tensor.size()[2]
    curr_width = tensor.size()[3]

    if target_height > curr_height or target_width > curr_width:
        raise Exception("Target size must not be greater than feature map size of tensor.")

    x_margin = (curr_width - target_width) // 2
    y_margin = (curr_height - target_height) // 2

    from_y, to_y = y_margin, curr_height - y_margin
    from_x, to_x = x_margin, curr_width - x_margin

    if target_height < to_y - from_y:
        to_y -= 1
    if target_width < to_x - from_x:
        to_x -= 1

    cropped_tensor = tensor[:, :, from_y:to_y, from_x:to_x]

    return cropped_tensor


    

def plot_imgs(img_left, img_right):
    _, ax = plt.subplots(1,2, constrained_layout=True, figsize= (10,10))

    ax[0].set_title('Input Image')
    ax[0].imshow(img_left)

    ax[1].set_title('Output Image')
    ax[1].imshow(img_right)

    plt.show()
