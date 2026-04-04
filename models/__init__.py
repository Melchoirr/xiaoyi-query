"""
models/__init__.py - 统一模型注册表

支持两类模型：
1. 检索模型（Retrieval-based）：PatternSearch, LSHSearch, SAXSearch, DTWSearch,
   MatrixProfileSearch, TS2VecSearch, RAGSearch
   - 使用记忆库进行检索预测，无需梯度训练
   - 接口：fit(X, Y) / predict(X)

2. 深度学习模型（Deep Learning）：DLinear, PatchTST
   - 使用神经网络进行预测
   - 接口：forward(x) / train() / eval()
"""

import torch
import torch.nn as nn
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
# 深度学习模型（Deep Learning Models）
# ================================================================

class DLinearModel(nn.Module):
    """DLinear 模型 - 季节性/趋势分解 + 线性预测"""

    def __init__(self, args):
        super().__init__()
        from layers.RevIN import RevIN

        self.args = args
        self.is_retrieval = False

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len

        self.use_revin = getattr(args, 'revin', False)
        if self.use_revin:
            self.revin = RevIN(args.enc_in)

        self.decompsition = series_decomp(25)
        self.individual = args.individual
        self.channels = args.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        if self.use_revin:
            x = self.revin.normalize(x)

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.permute(0, 2, 1)

        if self.use_revin:
            x = self.revin.denormalize(x)
        return x


class PatchTSTModel(nn.Module):
    """PatchTST 模型 - Patch + Transformer 编码器"""

    def __init__(self, args):
        super().__init__()
        from layers.RevIN import RevIN
        from layers.Embed import PatchEmbedding

        self.args = args
        self.is_retrieval = False

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = args.enc_in

        self.use_revin = getattr(args, 'revin', False)
        if self.use_revin:
            self.revin = RevIN(args.enc_in)

        self.patch_len = args.patch_len
        self.stride = args.stride
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.e_layers = args.e_layers
        self.d_ff = args.d_ff
        self.dropout = args.dropout
        self.fc_dropout_rate = getattr(args, 'fc_dropout', args.dropout)
        self.head_dropout_rate = getattr(args, 'head_dropout', 0.0)

        self.padding = self.stride
        self.patch_num = int((self.seq_len - self.patch_len + self.padding) / self.stride + 1)

        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride,
            padding=self.padding, dropout=self.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.fc_dropout_rate,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.e_layers)

        self.head = nn.Linear(self.patch_num * self.d_model, self.pred_len)
        self.head_dropout = nn.Dropout(self.head_dropout_rate)

    def forward(self, x):
        if self.use_revin:
            x = self.revin.normalize(x)

        B, L, C = x.shape
        x_patch, _, _ = self.patch_embedding(x)
        x_patch = self.encoder(x_patch)
        x_patch = x_patch.reshape(B * C, -1)
        x_patch = self.head_dropout(x_patch)
        x_out = self.head(x_patch)
        x_out = x_out.reshape(B, C, self.pred_len)
        x_out = x_out.permute(0, 2, 1)

        if self.use_revin:
            x_out = self.revin.denormalize(x_out)

        return x_out


# ================================================================
# 辅助函数
# ================================================================

def moving_avg(kernel_size, stride):
    """移动平均"""
    pad = (kernel_size - 1) // 2
    return nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=pad)


def series_decomp(kernel_size):
    """序列分解 - 季节性/趋势分离"""
    class SeriesDecomp(nn.Module):
        def __init__(self):
            super().__init__()
            self.moving_avg = moving_avg(kernel_size, stride=1)

        def forward(self, x):
            moving_mean = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
            res = x - moving_mean
            return res, moving_mean
    return SeriesDecomp()


# ================================================================
# 模型注册表
# ================================================================

# 检索模型
RETRIEVAL_MODELS = {
    'PatternSearch': PatternSearchModel,
    'LSHSearch': LSHSearchModel,
    'SAXSearch': SAXSearchModel,
    'DTWSearch': DTWSearchModel,
    'MatrixProfileSearch': MatrixProfileSearchModel,
    'TS2VecSearch': TS2VecSearchModel,
    'RAGSearch': RAGSearchModel,
}

# 深度学习模型
DEEP_LEARNING_MODELS = {
    'DLinear': DLinearModel,
    'PatchTST': PatchTSTModel,
}

# 全部模型
MODEL_DICT = {**RETRIEVAL_MODELS, **DEEP_LEARNING_MODELS}


def get_model(args):
    """获取模型实例"""
    model_name = args.model
    if model_name not in MODEL_DICT:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_DICT.keys())}")
    return MODEL_DICT[model_name](args)
