import numpy as np
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F
import pandas as pd
from utils import read_lines, load_tsv

def get_not_in_tissue_coords(coords, img_xy):
    img_x, img_y = img_xy
    coords = coords.astype(img_x.dtype)
    coords = [list(val) for val in np.array(coords)]
    not_in_tissue_coords = []
    not_in_tissue_index = []
    for i in range(img_x.shape[0]):
        for j in range(img_x.shape[1]):
            ij_coord = [img_x[i, j], img_y[i, j]]
            if ij_coord not in coords:
                not_in_tissue_coords.append(ij_coord)
                not_in_tissue_index.append([int(i), int(j)])
    return not_in_tissue_coords, np.array(not_in_tissue_index)

def get10Xtestset(test_counts, test_coords):
    test_counts = np.array(test_counts)
    test_coords = np.array(test_coords)
    delta_x = 1
    delta_y = 2
    x_min = min(test_coords[:, 0]) - min(test_coords[:, 0]) % 2
    y_min = min(test_coords[:, 1]) - min(test_coords[:, 1]) % 2
    test_input_x, test_input_y = np.mgrid[x_min:max(test_coords[:, 0]) + delta_x:delta_x,
                                 y_min:max(test_coords[:, 1]) + delta_y:delta_y]
    for i in range(1, test_input_y.shape[0], 2):
        test_input_y[i] = test_input_y[i] + delta_y / 2
    not_in_tissue_coords, not_in_tissue_xy = get_not_in_tissue_coords(test_coords, (test_input_x, test_input_y))
    not_in_tissue_x = not_in_tissue_xy.T[0]
    not_in_tissue_y = not_in_tissue_xy.T[1]
    test_set = [None] * test_counts.shape[1]
    for i in range(test_counts.shape[1]):
        test_data = griddata(test_coords, test_counts[:, i], (test_input_x, test_input_y), method="nearest")
        test_data[not_in_tissue_x, not_in_tissue_y] = 0
        test_set[i] = test_data
    test_set = np.array(test_set)
    return test_set

def getHRSGE(gene_set):
    _, h, w = gene_set.shape
    gene_set = torch.Tensor(gene_set)
    gene_set = gene_set.unsqueeze(1)
    HR_gene_set = F.interpolate(gene_set, size=(2 * h - 1, 2 * w - 1), mode='bilinear', align_corners=False)
    HR_gene_set = HR_gene_set.squeeze(1)
    return HR_gene_set

def get_10X_position_info(integral_coords):
    integral_coords = np.array(integral_coords)
    delta_x = 1
    delta_y = 2
    x_min = min(integral_coords[:, 0]) - min(integral_coords[:, 0]) % 2
    y_min = min(integral_coords[:, 1]) - min(integral_coords[:, 1]) % 2
    y = list(np.arange(y_min, max(integral_coords[:, 1]) + delta_y, delta_y))
    imputed_x, imputed_y = np.mgrid[x_min:max(integral_coords[:, 0]) + delta_x:delta_x / 2,
                           y_min:y[-1] + delta_y:delta_y / 2]
    for i in range(1, imputed_y.shape[0], 2):
        imputed_y[i] -= delta_y / 4
    for i in range(2, imputed_y.shape[0], 4):
        imputed_y[i:i + 2] += delta_y / 2
    
    def get_barcode(x, y):
        x_str = str(int(x)) if int(x) == x else str(x)
        y_str = str(int(y)) if int(y) == y else str(y)
        return x_str + "x" + y_str

    imputed_barcodes = [get_barcode(val[0], val[1]) for val in 
                        np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).T]
    
    imputed_coords = pd.DataFrame(np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).astype(np.float32).T,
                                  columns=['row', 'col'], index=imputed_barcodes)
    neighbor_matrix = pd.DataFrame(np.zeros((imputed_coords.shape[0], imputed_coords.shape[0]), dtype=np.int32),
                                   columns=imputed_barcodes, index=imputed_barcodes)
    
    row1 = imputed_coords[imputed_coords["row"] == min(imputed_coords["row"])].sort_values("col")
    for i in range(len(row1) - 1):
        if row1["col"].iloc[i + 1] - row1["col"].iloc[i] == delta_y / 2:
            neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
            neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1
    for row in list(np.array(imputed_x).T[0])[:-1]:
        row0 = imputed_coords[imputed_coords["row"] == row].sort_values("col")
        row1 = imputed_coords[imputed_coords["row"] == row + delta_x / 2].sort_values("col")
        for i in range(len(row1) - 1):
            if row1["col"].iloc[i + 1] - row1["col"].iloc[i] == delta_y / 2:
                neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
                neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1
        for i in range(len(row0)):
            for j in range(len(row1)):
                flag = 0
                if abs(imputed_coords.loc[row0.index[i], "col"] - imputed_coords.loc[
                    row1.index[j], "col"]) == delta_y / 4:
                    neighbor_matrix.loc[row0.index[i], row1.index[j]] = 1
                    neighbor_matrix.loc[row1.index[j], row0.index[i]] = 1
                    flag += 1
                if flag >= 2: continue

    target_columns = [f"{int(v[0])}x{int(v[1])}" for v in integral_coords]
    existing_cols = neighbor_matrix.columns.intersection(target_columns)
    neighbor_matrix = neighbor_matrix.loc[:, existing_cols]
    
    not_in_tissue_coords = []
    for i in range(len(imputed_coords)):
        if imputed_coords.index[i] in neighbor_matrix.columns: continue
        if sum(neighbor_matrix.iloc[i]) < 2:
            not_in_tissue_coords.append(list(imputed_coords.iloc[i]))
    position_info = [imputed_x, imputed_y, not_in_tissue_coords]
    return position_info

def img2expr(imputed_img, gene_ids, integral_coords, position_info):
    [imputed_x, imputed_y, not_in_tissue_coords] = position_info
    imputed_img = imputed_img.numpy()
    if type(not_in_tissue_coords) == np.ndarray:
        not_in_tissue_coords = [list(val) for val in not_in_tissue_coords]
    integral_barcodes = ['{}x{}'.format(int(row[0]), int(row[1])) for row in integral_coords]
    integral_barcodes = pd.Index(integral_barcodes)
    imputed_counts = pd.DataFrame(np.zeros((imputed_img.shape[1] * imputed_img.shape[2] - len(not_in_tissue_coords),
                                            imputed_img.shape[0])), columns=gene_ids)
    imputed_coords = pd.DataFrame(np.zeros((imputed_img.shape[1] * imputed_img.shape[2] - len(not_in_tissue_coords),
                                            2)), columns=['array_row', 'array_col'])
    imputed_barcodes = [None] * len(imputed_counts)
    integral_coords_list = [list(i.astype(int)) for i in np.array(integral_coords)]
    flag = 0
    for i in range(imputed_img.shape[1]):
        for j in range(imputed_img.shape[2]):
            sx, sy = imputed_x[i, j], imputed_y[i, j]
            spot_coords = [sx, sy]
            if spot_coords in not_in_tissue_coords: continue
            if [int(sx), int(sy)] in integral_coords_list and int(sx)==sx and int(sy)==sy:
                imputed_barcodes[flag] = integral_barcodes[integral_coords_list.index([int(sx), int(sy)])]
            else:
                x_id = str(int(sx)) if int(sx) == sx else str(sx)
                y_id = str(int(sy)) if int(sy) == sy else str(sy)
                imputed_barcodes[flag] = x_id + "x" + y_id
            imputed_counts.iloc[flag , :] = imputed_img[:, i, j]
            imputed_coords.iloc[flag , :] = spot_coords
            flag = flag + 1
    imputed_counts.index = imputed_barcodes
    imputed_coords.index = imputed_barcodes
    return imputed_counts, imputed_coords

def get_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts = load_tsv(f'{prefix}cnts.csv')
    cnts = cnts.iloc[:, cnts.var().to_numpy().argsort()[::-1]]
    cnts = cnts[gene_names]
    locs, meta = get_locs(prefix)
    return cnts, locs, meta

def get_locs(prefix):
    locs_raw = load_tsv(f'{prefix}locs.csv')
    x_phys = locs_raw['x'].values.astype(float)
    y_phys = locs_raw['y'].values.astype(float)

    def find_robust_step_and_offset(data):
        unique_vals = np.sort(np.unique(data.round(1)))
        diffs = np.diff(unique_vals)

        real_steps = diffs[diffs > 30]
        step = np.median(real_steps) if len(real_steps) > 0 else 100.0

        offset = np.min(data) % step
        return step, offset

    x_step, x_offset = find_robust_step_and_offset(x_phys)
    y_step, y_offset = find_robust_step_and_offset(y_phys)

    locs_v = np.stack([x_phys//x_step, y_phys//y_step], -1)

    meta = {
        'x_step': x_step, 'y_step': y_step, 
        'x_offset': x_offset, 'y_offset': y_offset
    }
    return locs_v, meta

def main(prefix):
    cnts, locs, meta = get_data(prefix)
    print(f"Meta Info: {meta}")
    
    testSet = get10Xtestset(cnts, locs)
    HR_testSet = getHRSGE(testSet)
    Hr_locs = get_10X_position_info(locs)
    
    HR_testSet = torch.nn.functional.pad(HR_testSet, (0, 1, 0, 1, 0, 0), mode='constant', value=0)
    imputed_counts, imputed_coords = img2expr(HR_testSet, cnts.columns, locs, Hr_locs)
    
    imputed_counts.to_csv(f'{prefix}HRcnts111.csv')
    
    imputed_coords.columns = ['x', 'y']
    imputed_coords['x'] = imputed_coords['x'] * meta['x_step'] + meta['x_offset']
    imputed_coords['y'] = imputed_coords['y'] * meta['y_step'] + meta['y_offset']
    
    imputed_coords.to_csv(f'{prefix}HRlocs111.csv')
    
    print(f"Total Spots: {len(imputed_coords)}")

def run_get_hr(config):
    dir = config['global']['directory']
    
    main(dir)