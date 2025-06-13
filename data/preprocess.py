import numpy as np


def uniform_idxs(n_selection, start, end):
    res = []
    while len(res) < n_selection:
        rand_int = int(np.random.uniform(start, end))
        if rand_int not in res:
            res.append(rand_int)
    res = sorted(res)
    return res


def pdf(x):
    mean = np.mean(x)
    std = np.std(x)
    y_out = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(- (x - mean) ** 2 / (2 * std ** 2))
    return y_out


def middle_idxs(n_selection, len_arr):
    x = np.arange(0, len_arr, 1)
    selection_pdf = pdf(x)
    selection_prob = [float(i) / sum(selection_pdf) for i in selection_pdf]
    results = np.random.choice(len_arr, size=n_selection, p=selection_prob, replace=False)
    return sorted(results)


def random_idxs(n_selection, array_len):
    res = []
    while len(res) < n_selection:
        rand_int = int(np.random.uniform(0, array_len))
        if rand_int not in res:
            res.append(rand_int)
    res = sorted(res)
    return res


def random_idxs_from_arr(n_selection, array_len, arr):
    res = []
    main_arr = list(set(range(0, array_len)) - set(arr))
    while len(res) < n_selection:
        rand_int = int(np.random.uniform(0, len(main_arr)))
        if main_arr[rand_int] not in res:
            res.append(main_arr[rand_int])
    res = sorted(res)

    return res


def split_and_select_indices(lst, m):
    # Convert the list to a numpy array
    np_array = np.array(lst)

    # Split the array into m parts
    parts = np.array_split(np_array, m)

    # Randomly select an index from each part
    selected_indices = [np.random.choice(part) for part in parts]

    return selected_indices


def split_and_select_indices_fixed(lst, m):
    # Convert the list to a numpy array
    np_array = np.array(lst)

    # Split the array into m parts
    parts = np.array_split(np_array, m)

    # select the middle index form each part
    selected_indices = [part[int(len(part) // 2)] for part in parts]

    return selected_indices
