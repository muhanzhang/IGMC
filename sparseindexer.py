import numpy as np
import scipy.sparse as sp

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        # Iterating over the rows this way is significantly more efficient
        # than csr_matrix[row_index,:] and csr_matrix.getrow(row_index)
        for row_start, row_end in zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:]):
            data.append(csr_matrix.data[row_start:row_end])
            indices.append(csr_matrix.indices[row_start:row_end])
            indptr.append(row_end - row_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.n_columns = csr_matrix.shape[1]

    def __getitem__(self, row_selector, col_selector=None):
        indices = np.concatenate(self.indices[row_selector])
        data = np.concatenate(self.data[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))
        # if col_selector:
        #     mask = [c in col_selector for c in indices]
        #     indices = indices[mask]
        #     for i,c in enumerate(np.sort(col_selector)):
        #         indices[indices==c] = i
        #     data = data[mask]
        #     for i in range(indptr.size-1):
        #         indptr[i+1] = indptr[i] + np.sum(mask[indptr[i]:indptr[i+1]])
        #
        #     shape = [indptr.shape[0] - 1, len(col_selector)]
        # else:

        shape = [indptr.shape[0] - 1, self.n_columns]
        return sp.csr_matrix((data, indices, indptr), shape=shape)

class SparseColIndexer:
    def __init__(self, csc_matrix):
        data = []
        indices = []
        indptr = []

        # Iterating over the rows this way is significantly more efficient
        # than csr_matrix[row_index,:] and csr_matrix.getrow(row_index)
        for col_start, col_end in zip(csc_matrix.indptr[:-1], csc_matrix.indptr[1:]):
            data.append(csc_matrix.data[col_start:col_end])
            indices.append(csc_matrix.indices[col_start:col_end])
            indptr.append(col_end - col_start)  # nnz of the row

        self.data = np.array(data, dtype=object)
        self.indices = np.array(indices, dtype=object)
        self.indptr = np.array(indptr, dtype=object)
        self.n_rows = csc_matrix.shape[0]

    def __getitem__(self, col_selector):
        indices = np.concatenate(self.indices[col_selector])
        data = np.concatenate(self.data[col_selector])
        indptr = np.append(0, np.cumsum(self.indptr[col_selector]))

        shape = [self.n_rows ,indptr.shape[0] - 1]
        return sp.csc_matrix((data, indices, indptr), shape=shape)


