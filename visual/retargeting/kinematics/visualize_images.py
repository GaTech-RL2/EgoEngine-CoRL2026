import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_data(src_dir):

    img = np.array(Image.open(f"{src_dir}/image_0.png"))
    dep = np.array(np.load(f"{src_dir}/depth_0.npy"))
    segm = np.array(np.load(f"{src_dir}/segm_0.npy"))

    print(dep.shape, segm.shape, segm.max(), segm.min())
    _, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[1].imshow(dep)
    axs[2].imshow(segm)
    plt.show()

load_data("/Users/shuocheng/Downloads/debug_env/1")
