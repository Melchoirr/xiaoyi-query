"""
models/__init__.py - 统一模型注册表

支持7种检索模型：
- PatternSearch: 欧氏距离 KNN 检索
- LSHSearch: 局部敏感哈希检索
- SAXSearch: PAA+SAX 符号化检索
- DTWSearch: 动态时间规整检索
- MatrixProfileSearch: GPU Z-Norm 子序列检索
- TS2VecSearch: 深度对比学习检索
- RAGSearch: Siamese Cross-Attention RAG

所有模型接口：fit(X, Y) / predict(X)
"""

import torch
import numpy as np

# ================================================================
# 检索模型（Retrieval-based Models）
# ================================================================

class PatternSearchModel:
    """PatternSearch 模型包装器 - 欧氏距离 KNN 检索"""

    def __init__(self, args):
        from models.PatternSearch import PatternSearch
        self.args = args
        self.is_retrieval = True
        self._is_nn_module = False  # 标记不是 PyTorch Module

        device = self._get_device()
        self.model = PatternSearch(
            top_k=args.top_k,
            weighted=args.weighted,
            algorithm='kd_tree',
            device=device,
            predict_chunk_size=args.predict_chunk_size,
        )
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_features = args.enc_in

    def _get_device(self):
        if getattr(self.args, 'use_gpu', False) and torch.cuda.is_available():
            return f'cuda:{getattr(self.args, "gpu", 0)}'
        return 'cpu'

    def fit(self, X_train, Y_train):
        X = X_train.astype(np.float32)
        Y = Y_train.astype(np.float32)

        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)
        if Y.ndim == 2:
            Y = Y.reshape(Y.shape[0], -1, 1)

        self.seq_len = X.shape[1]
        self.pred_len = Y.shape[1]
        self.n_features = X.shape[2]

        self.model.seq_len = self.seq_len
        self.model.pred_len = self.pred_len
        self.model.n_features = self.n_features
        self.model.fit(X, Y)

    def predict(self, X_test):
        # 如果是 torch tensor，先转为 numpy
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.cpu().numpy()

        X = X_test.astype(np.float32)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)

        Y_pred = self.model.predict(X)

        if Y_pred.ndim == 2:
            Y_pred = Y_pred.reshape(Y_pred.shape[0], self.pred_len, self.n_features)

        return Y_pred

    def forward(self, x):
        return self.predict(x)


class LSHSearchModel:
    """LSHSearch 模型包装器 - 局部敏感哈希检索"""

    def __init__(self, args):
        from models.LSHSearch import LSHSearch
        self.args = args
        self.is_retrieval = True
        self._is_nn_module = False  # 标记不是 PyTorch Module

        device = self._get_device()
        self.model = LSHSearch(
            n_hash_funcs=args.n_hash_funcs,
            n_tables=args.n_tables,
            hamming_radius=args.hamming_radius,
            candidate_cap_per_table=args.candidate_cap_per_table,
            candidate_cap_total=args.candidate_cap_total,
            weighted=args.lsh_weighted,
            device=device,
        )
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_features = args.enc_in

    def _get_device(self):
        if getattr(self.args, 'use_gpu', False) and torch.cuda.is_available():
            return f'cuda:{getattr(self.args, "gpu", 0)}'
        return 'cpu'

    def fit(self, X_train, Y_train):
        X = X_train.astype(np.float32)
        Y = Y_train.astype(np.float32)

        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)
        if Y.ndim == 2:
            Y = Y.reshape(Y.shape[0], -1, 1)

        self.seq_len = X.shape[1]
        self.pred_len = Y.shape[1]
        self.n_features = X.shape[2]

        self.model.seq_len = self.seq_len
        self.model.pred_len = self.pred_len
        self.model.n_features = self.n_features
        self.model.fit(X, Y)

    def predict(self, X_test, top_k=None):
        # 如果是 torch tensor，先转为 numpy
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.cpu().numpy()

        X = X_test.astype(np.float32)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)

        k = top_k if top_k is not None else getattr(self.args, 'top_k', 10)
        Y_pred = self.model.predict(X, top_k=k)

        if Y_pred.ndim == 2:
            Y_pred = Y_pred.reshape(Y_pred.shape[0], self.pred_len, self.n_features)

        return Y_pred

    def forward(self, x):
        return self.predict(x)


class SAXSearchModel:
    """SAXSearch 模型包装器 - PAA+SAX 符号化检索"""

    def __init__(self, args):
        from models.SAXSearch import SAXSearch
        self.args = args
        self.is_retrieval = True
        self._is_nn_module = False  # 标记不是 PyTorch Module

        device = self._get_device()
        self.model = SAXSearch(
            word_size=args.word_size,
            alphabet_size=args.alphabet_size,
            epsilon_threshold=args.epsilon_threshold,
            bucket_top_k=args.bucket_top_k,
            weighted=args.sax_weighted,
            device=device,
        )
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_features = args.enc_in

    def _get_device(self):
        if getattr(self.args, 'use_gpu', False) and torch.cuda.is_available():
            return f'cuda:{getattr(self.args, "gpu", 0)}'
        return 'cpu'

    def fit(self, X_train, Y_train):
        X = X_train.astype(np.float32)
        Y = Y_train.astype(np.float32)

        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)
        if Y.ndim == 2:
            Y = Y.reshape(Y.shape[0], -1, 1)

        self.seq_len = X.shape[1]
        self.pred_len = Y.shape[1]
        self.n_features = X.shape[2]

        self.model.seq_len = self.seq_len
        self.model.pred_len = self.pred_len
        self.model.n_features = self.n_features
        self.model.fit(X, Y)

    def predict(self, X_test, top_k=None):
        # 如果是 torch tensor，先转为 numpy
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.cpu().numpy()

        X = X_test.astype(np.float32)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)

        k = top_k if top_k is not None else getattr(self.args, 'top_k', 10)
        Y_pred = self.model.predict(X, top_k=k)

        if Y_pred.ndim == 2:
            Y_pred = Y_pred.reshape(Y_pred.shape[0], self.pred_len, self.n_features)

        return Y_pred

    def forward(self, x):
        return self.predict(x)


class DTWSearchModel:
    """DTWSearch 模型包装器 - 动态时间规整检索"""

    def __init__(self, args):
        from models.DTWSearch import DTWSearch
        self.args = args
        self.is_retrieval = True
        self._is_nn_module = False  # 标记不是 PyTorch Module

        device = 'auto'  # DTWSearch 会自动检测
        self.model = DTWSearch(
            top_k=args.top_k,
            dtw_radius=args.dtw_radius,
            weighted=args.weighted,
            device=device,
            predict_chunk_size=args.predict_chunk_size,
        )
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_features = args.enc_in

    def fit(self, X_train, Y_train):
        X = X_train.astype(np.float32)
        Y = Y_train.astype(np.float32)

        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)
        if Y.ndim == 2:
            Y = Y.reshape(Y.shape[0], -1, 1)

        self.seq_len = X.shape[1]
        self.pred_len = Y.shape[1]
        self.n_features = X.shape[2]

        self.model.seq_len = self.seq_len
        self.model.pred_len = self.pred_len
        self.model.n_features = self.n_features
        self.model.fit(X, Y)

    def predict(self, X_test):
        # 如果是 torch tensor，先转为 numpy
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.cpu().numpy()

        X = X_test.astype(np.float32)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)

        Y_pred = self.model.predict(X)

        if Y_pred.ndim == 2:
            Y_pred = Y_pred.reshape(Y_pred.shape[0], self.pred_len, self.n_features)

        return Y_pred

    def forward(self, x):
        return self.predict(x)


class MatrixProfileSearchModel:
    """MatrixProfileSearch 模型包装器 - GPU Z-Norm 子序列检索"""

    def __init__(self, args):
        from models.MatrixProfileSearch import MatrixProfileSearch
        self.args = args
        self.is_retrieval = True
        self._is_nn_module = False  # 标记不是 PyTorch Module

        device = 'auto'  # MatrixProfileSearch 会自动检测
        self.model = MatrixProfileSearch(
            top_k=args.top_k,
            subsequence_length=getattr(args, 'subsequence_length', None),
            normalize=getattr(args, 'mp_normalize', True),
            device=device,
            predict_chunk_size=args.predict_chunk_size,
            train_chunk_size=getattr(args, 'train_chunk_size', 1024),
        )
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_features = args.enc_in

    def fit(self, X_train, Y_train):
        X = X_train.astype(np.float32)
        Y = Y_train.astype(np.float32)

        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)
        if Y.ndim == 2:
            Y = Y.reshape(Y.shape[0], -1, 1)

        self.seq_len = X.shape[1]
        self.pred_len = Y.shape[1]
        self.n_features = X.shape[2]

        self.model.seq_len = self.seq_len
        self.model.pred_len = self.pred_len
        self.model.n_features = self.n_features
        self.model.fit(X, Y)

    def predict(self, X_test):
        # 如果是 torch tensor，先转为 numpy
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.cpu().numpy()

        X = X_test.astype(np.float32)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)

        Y_pred = self.model.predict(X)

        if Y_pred.ndim == 2:
            Y_pred = Y_pred.reshape(Y_pred.shape[0], self.pred_len, self.n_features)

        return Y_pred

    def forward(self, x):
        return self.predict(x)


class TS2VecSearchModel:
    """TS2VecSearch 模型包装器 - 深度对比学习检索"""

    def __init__(self, args):
        from models.TS2VecSearch import TS2VecSearch
        self.args = args
        self.is_retrieval = True
        self._is_nn_module = False  # 标记不是 PyTorch Module

        device = 'auto'  # TS2VecSearch 会自动检测
        self.model = TS2VecSearch(
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            top_k=args.top_k,
            lr=args.lr,
            temperature=args.temperature,
            device=device,
            seed=getattr(args, 'seed', 42),
        )
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_features = args.enc_in

    def fit(self, X_train, Y_train):
        X = X_train.astype(np.float32)
        Y = Y_train.astype(np.float32)

        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)
        if Y.ndim == 2:
            Y = Y.reshape(Y.shape[0], -1, 1)

        self.seq_len = X.shape[1]
        self.pred_len = Y.shape[1]
        self.n_features = X.shape[2]

        self.model.seq_len = self.seq_len
        self.model.pred_len = self.pred_len
        self.model.n_features = self.n_features
        self.model.fit(X, Y)

    def predict(self, X_test):
        # 如果是 torch tensor，先转为 numpy
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.cpu().numpy()

        X = X_test.astype(np.float32)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)

        Y_pred = self.model.predict(X)

        if Y_pred.ndim == 2:
            Y_pred = Y_pred.reshape(Y_pred.shape[0], self.pred_len, self.n_features)

        return Y_pred

    def forward(self, x):
        return self.predict(x)


class RAGSearchModel:
    """RAGSearch 模型包装器 - Siamese Cross-Attention RAG"""

    def __init__(self, args):
        from models.RAGSearch import RAGSearch
        self.args = args
        self.is_retrieval = True
        self._is_nn_module = False  # 标记不是 PyTorch Module

        device = 'auto'  # RAGSearch 会自动检测
        self.model = RAGSearch(
            d_model=getattr(args, 'rag_d_model', 32),
            n_heads=getattr(args, 'rag_n_heads', 4),
            epochs=getattr(args, 'rag_epochs', 10),
            batch_size=getattr(args, 'rag_batch_size', 128),
            lr=getattr(args, 'rag_lr', 1e-3),
            weight_decay=getattr(args, 'rag_weight_decay', 1e-4),
            device=device,
            seed=getattr(args, 'seed', 42),
        )
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.n_features = args.enc_in

    def fit(self, X_train, Y_train):
        X = X_train.astype(np.float32)
        Y = Y_train.astype(np.float32)

        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)
        if Y.ndim == 2:
            Y = Y.reshape(Y.shape[0], -1, 1)

        self.seq_len = X.shape[1]
        self.pred_len = Y.shape[1]
        self.n_features = X.shape[2]

        self.model.seq_len = self.seq_len
        self.model.pred_len = self.pred_len
        self.model.n_features = self.n_features
        self.model.fit(X, Y)

    def predict(self, X_test):
        # 如果是 torch tensor，先转为 numpy
        if isinstance(X_test, torch.Tensor):
            X_test = X_test.cpu().numpy()

        X = X_test.astype(np.float32)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, 1)

        Y_pred = self.model.predict(X)

        if Y_pred.ndim == 2:
            Y_pred = Y_pred.reshape(Y_pred.shape[0], self.pred_len, self.n_features)

        return Y_pred

    def forward(self, x):
        return self.predict(x)


# ================================================================
# 模型注册表
# ================================================================

RETRIEVAL_MODELS = {
    'PatternSearch': PatternSearchModel,
    'LSHSearch': LSHSearchModel,
    'SAXSearch': SAXSearchModel,
    'DTWSearch': DTWSearchModel,
    'MatrixProfileSearch': MatrixProfileSearchModel,
    'TS2VecSearch': TS2VecSearchModel,
    'RAGSearch': RAGSearchModel,
}

# 全部模型（只保留7种检索模型）
MODEL_DICT = RETRIEVAL_MODELS


def get_model(args):
    """获取模型实例"""
    model_name = args.model
    if model_name not in MODEL_DICT:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_DICT.keys())}")
    return MODEL_DICT[model_name](args)
