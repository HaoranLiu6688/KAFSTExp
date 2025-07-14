import time
import os
import numpy as np
import cupy as cp
import pandas as pd
import time
import json
import anndata
import matplotlib.pyplot as plt
from tqdm import tqdm
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

rand_seed = 2
cp.random.seed(rand_seed)


def get_r(data1, data2, dim=1, func=pearsonr):
    adata1 = data1.X
    adata2 = data2.X
    r1, p1 = [], []
    for g in range(data1.shape[dim]):
        if dim == 1:
            r, pv = func(adata1[:, g], adata2[:, g])
        elif dim == 0:
            r, pv = func(adata1[g, :], adata2[g, :])
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1


def nys_klms_train_multigene(train_input, train_label, type_kernel, param_kernel, step_size, m):
    input_dimension, train_size = np.shape(train_input)

    kmeans = KMeans(n_clusters=m, random_state=rand_seed).fit(train_input.T.get())

    nys_dict = cp.array(kmeans.cluster_centers_.T, dtype=cp.float32)
    kernel_matrix = cp.zeros((nys_dict.shape[1], nys_dict.shape[1]), dtype=cp.float32)
    for i in range(nys_dict.shape[1]):
        for j in range(nys_dict.shape[1]):
            kernel_matrix[i, j] = ker_eval(nys_dict[:, i:i+1], nys_dict[:, j:j+1], type_kernel, param_kernel)
            kernel_matrix[j, i] = kernel_matrix[i, j]

    eigvalue, eigvector = np.linalg.eig(kernel_matrix.get())
    eigvalue_inv_sqrt = np.diag(1.0 / np.sqrt(eigvalue))
    pconst = cp.array(eigvalue_inv_sqrt @ eigvector.T)
    del kernel_matrix, eigvalue, eigvector

    aprioriErr = train_label[0:1, :]
    z = pconst @ ker_eval(train_input[:, 0:1], nys_dict, type_kernel, param_kernel)
    weightVector = step_size * z @ aprioriErr

    for n in tqdm(range(1, train_size)):
        z = pconst @ ker_eval(train_input[:, n:n+1], nys_dict, type_kernel, param_kernel)
        net_output = weightVector.T @ z
        aprioriErr = train_label[n:n+1, :] - net_output.T
        weightVector += step_size * z @ aprioriErr

    return weightVector, pconst, nys_dict


def nys_klms_test_multigene(weight, test_input, test_label, type_kernel, param_kernel, p, d):
    k_test_dict = p @ ker_eval(test_input, d, type_kernel, param_kernel).T  # (test_size, n_train)
    y_te = cp.dot(k_test_dict.T, weight)

    adata_ture = anndata.AnnData(test_label.get())
    adata_pred = anndata.AnnData(y_te.get())
    pear, p = get_r(adata_pred, adata_ture)
    p = np.where(p == 0, 1e-323, p)

    mse = np.mean((y_te - test_label) ** 2, axis=0)
    return pear, mse, -np.log10(p)


def ker_eval(x1, x2, ker_type, ker_param):
    N1 = x1.shape[1]
    N2 = x2.shape[1]
    if ker_type == "Gauss":
        if N1 == N2:
            y = cp.exp(-cp.sum((x1 - x2) ** 2, axis=0) * ker_param).T
        elif N1 == 1:
            y = cp.exp(-cp.sum((x1 - x2) ** 2, axis=0, keepdims=True) * ker_param).T
        else:
            dist = cp.sum(x1 ** 2, axis=0, keepdims=True).T + cp.sum(x2 ** 2, axis=0, keepdims=True) - 2 * cp.dot(x1.T, x2)
            dist = cp.maximum(dist, 0)
            y = cp.exp(-dist * ker_param)
    else:
        raise ValueError("Unsupported kernel type")
    return y


def pearson_correlation(x, y):
    """
    Compute the Pearson correlation coefficient between two arrays x and y
    """
    # Compute the mean
    mean_x = cp.mean(x, axis=0)
    mean_y = cp.mean(y, axis=0)
    # Compute the standard deviation
    std_x = cp.std(x, axis=0)
    std_y = cp.std(y, axis=0)
    # Compute the covariance
    covariance = cp.mean((x - mean_x) * (y - mean_y), axis=0)
    # Compute the Pearson correlation coefficient
    correlation = covariance / (std_x * std_y)

    return correlation


if __name__ == "__main__":
    task = 'IDC'
    fm = 'UNI'
    fold_num = len(os.listdir("./"+task+"/cv_data/"+fm)) // 4
    pcc_summary = []
    pccs = []
    pv = []
    mse_summary = []
    mses = []
    heg_pcc = []
    hvg_pcc = []

    start_time = time.time()
    for fold in range(fold_num):
        # Load training and testing data
        fold = str(fold + 1)
        patch_embedding_train = pd.read_csv("./"+task+"/cv_data/"+fm+"/fold"+fold+"_train_embeddings.csv", header=None)
        train_label = pd.read_csv("./"+task+"/cv_data/"+fm+"/fold"+fold+"_train_gene_labels.csv")
        patch_embedding_test = pd.read_csv("./"+task+"/cv_data/"+fm+"/fold"+fold+"_test_embeddings.csv", header=None)
        test_label = pd.read_csv("./"+task+"/cv_data/"+fm+"/fold"+fold+"_test_gene_labels.csv")

        memory_pool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(memory_pool.malloc)

        # Convert pandas DataFrame to CuPy array
        patch_train_cupy = cp.array(patch_embedding_train.T.values, dtype=cp.float32)
        patch_test_cupy = cp.array(patch_embedding_test.T.values, dtype=cp.float32)
        label_train_cupy = cp.array(train_label.values, dtype=cp.float32)
        label_test_cupy = cp.array(test_label.values, dtype=cp.float32)

        del patch_embedding_train, train_label, patch_embedding_test, test_label

        sample_indices = cp.random.choice(label_train_cupy.shape[0], label_train_cupy.shape[0], replace=False)
        patch_train_cupy = patch_train_cupy[:, sample_indices]
        label_train_cupy = label_train_cupy[sample_indices, :]

        heg_ind = cp.argsort(cp.mean(label_test_cupy, axis=0))[::-1][:50]
        hvg_ind = cp.argsort(cp.var(label_test_cupy, axis=0))[::-1][:50]

        weight, p_con, nys = nys_klms_train_multigene(patch_train_cupy, label_train_cupy, "Gauss", 0.0005, 0.2, 150)
        pearson, mse, pp = nys_klms_test_multigene(weight, patch_test_cupy, label_test_cupy, "Gauss", 0.0005, p_con, nys)

        pcc_summary.append(np.nanmean(pearson))

        heg_pcc.append(np.nanmean(pearson[heg_ind.get()]))
        hvg_pcc.append(np.nanmean(pearson[hvg_ind.get()]))

        pccs.append(pearson)
        pv.append(pp)
        mse_summary.append(np.nanmean(mse).get())
        mses.append(mse.get())

        print("fold %s" % fold)
        print("The mean of PCC: %.4f" % (np.nanmean(pearson)))
        print("The std of PCC: %.4f" % (np.nanstd(pearson)))
        print("The mean of MSE: %.4f" % (np.nanmean(mse)))
        print("The std of MSE: %.4f" % (np.nanstd(mse)))
        print("===========================================================================")

    end_time = time.time()

    pv = np.mean(np.array(pv), axis=0)
    pccs = np.mean(np.array(pccs), axis=0)
    mses = np.mean(np.array(mses), axis=0)
    with open('E:/Spatial transcriptomics/st_kaf/' + task + '/var_50genes.json', 'r') as f:
        genes = json.load(f)['genes']
    df = pd.DataFrame([pccs, mses, pv], columns=genes)
    sorted_df = df.sort_values(by=0, axis=1, ascending=False)

    sorted_log_df = df.sort_values(by=2, axis=1, ascending=False)
    top_10_records = sorted_log_df.columns[:10].tolist()

    pcc_values = sorted_df.iloc[0]
    pcc_10th_value = pcc_values.iloc[9]
    mse_values = sorted_df.iloc[1]
    columns = sorted_df.columns

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = '#9AC4DB'
    ax1.bar(columns, pcc_values, color=color, label='PCC')
    ax1.set_xlabel('Genes', fontsize=12)
    ax1.set_ylabel('Pearson Correlation Coefficient (PCC)', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.tick_params(axis='x', rotation=90)

    ax1.axhline(y=pcc_10th_value, color='gray', linestyle='--', linewidth=1)

    ax1.annotate(
        f'Top 10 PCC Threshold={pcc_10th_value:.3f}',
        xy=(len(columns)-13 - 0.5, pcc_10th_value),
        xytext=(10, 5),
        textcoords='offset points',
        fontsize=12,
        color='gray',
        ha='left',
        va='bottom'
    )

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.plot(columns, mse_values, color=color, marker='o', linestyle='-', linewidth=2, label='MSE')
    ax2.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax2.tick_params(axis='y')

    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.9), ncol=1)

    plt.tight_layout()
    plt.show()

    print("Summary:")
    print("The mean of PCC: %.4f" % (np.mean(pcc_summary)))
    print("The std of PCC: %.4f" % (np.std(pcc_summary)))

    print("The mean of HEG PCC: %.4f" % (np.mean(heg_pcc)))
    print("The std of HEG PCC: %.4f" % (np.std(heg_pcc)))

    print("The mean of HVG PCC: %.4f" % (np.mean(hvg_pcc)))
    print("The std of HVG PCC: %.4f" % (np.std(hvg_pcc)))

    print("The mean of MSE: %.4f" % (np.mean(mse_summary)))
    print("The std of MSE: %.4f" % (np.std(mse_summary)))
    print(top_10_records)
    print(f"Code execution time: {end_time - start_time:.4f} seconds")

