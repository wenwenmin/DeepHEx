from torch.utils.data import Dataset
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from utils import get_disk_mask

def dimensionality_reduction(arr, target_size):
    x = np.arange(arr.shape[-1])
    f = interp1d(x, arr, axis=-1)
    arr_compressed = f(np.linspace(0, arr.shape[-1] - 1, target_size))
    return arr_compressed

class SpotDataset(Dataset):

    def __init__(self, x_all, y, enhance_y, locs, enhance_locs, radius, k=6):
        super().__init__()
        # x_all = x_all[:, :, 0:1000]
        #x_all = dimensionality_reduction(x_all, y.shape[-1])
        mask = get_disk_mask(radius)
        his = get_patches_flat(x_all, locs, mask)
        gene= get_patches_genes_test(locs, enhance_locs, enhance_y, k=k)
        x = dict(his=his, gene=gene)
        self.x = x
        self.y = y
        self.locs = locs
        self.size = x_all.shape[:2]
        self.radius = radius
        self.mask = mask

    def __len__(self):
        return len(self.x['his'])

    def __getitem__(self, idx):
        x_item = {key: value[idx] for key, value in self.x.items()}
        y_item = self.y[idx]
        return x_item, y_item


def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    mask = np.ones_like(mask, dtype=bool)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]

        x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list

def get_patches_genes(locs, y, k=10):

    tree = cKDTree(locs)

    _, indices = tree.query(locs, k=k)

    genes_list = [y[idx] for idx in indices]
    genes_list = np.stack(genes_list)

    pos_list = [locs[idx] for idx in indices]
    pos_list = np.stack(pos_list)

    return genes_list, pos_list

def get_patches_genes_test(locs1, locs2, y, k=6):

    tree = cKDTree(locs2)

    _, indices = tree.query(locs1, k=k)

    patches = [y[idx] for idx in indices]

    return patches

def get_center_coordinates_rounded(h, w, block_size):

    step = block_size

    center_y = np.arange(step / 2, h, step)
    center_x = np.arange(step / 2, w, step)

    grid_x, grid_y = np.meshgrid(center_x, center_y)

    grid_x = np.round(grid_x).astype(int)
    grid_y = np.round(grid_y).astype(int)

    locs1 = np.stack([grid_y.ravel(), grid_x.ravel()], axis=-1)

    return locs1