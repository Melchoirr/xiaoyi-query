python run.py --model PatternSearch --data ETTh1 --seq_len 96 --pred_len 96 --top_k 5
python run.py --model LSHSearch --data ETTh1 --seq_len 96 --pred_len 96 --top_k 5 --recall_k 64
python run.py --model SAXSearch --data ETTh1 --seq_len 96 --pred_len 96 --top_k 5
python run.py --model DTWSearch --data ETTh1 --seq_len 96 --pred_len 96 --top_k 5 --dtw_radius 5
python run.py --model MatrixProfileSearch --data ETTh1 --seq_len 96 --pred_len 96 --top_k 5
python run.py --model TS2VecSearch --data ETTh1 --seq_len 96 --pred_len 96 --top_k 5 --ts2vec_epochs 5
python run.py --model RAGSearch --data ETTh1 --seq_len 96 --pred_len 96 --top_k 5 --rag_epochs 3
