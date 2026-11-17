

Bộ dữ liệu bắt buộc: UCI Bank Marketing (bank-full.csv), ID 222.
Tài liệu: https://archive.ics.uci.edu/dataset/222/bank+marketing
Chạy: python main.py --data data/bank-full.csv
"""

import argparse
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def eda(df: pd.DataFrame, out: Path) -> None:
    """Khám phá dữ liệu; không dùng kết quả từ test để fit bộ tiền xử lý."""
    (out / 'eda_tong_quan.txt').write_text(
        f"Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột\n"
        f"Cột và kiểu dữ liệu:\n{df.dtypes.to_string()}\n\n"
        f"Thiếu theo cột (NaN):\n{df.isna().sum().to_string()}\n\n"
        f"Trùng lặp: {df.duplicated().sum()}\n\n"
        f"Nhãn gốc:\n{df['y'].value_counts(dropna=False).to_string()}\n\n"
        f"Tỷ lệ nhãn:\n{df['y'].value_counts(normalize=True).to_string()}\n\n"
        f"Số giá trị 'unknown' ở các cột văn bản:\n"
        f"{df.select_dtypes(include='object').eq('unknown').sum().to_string()}\n\n"
        f"Thống kê đặc trưng số:\n{df.select_dtypes(include='number').describe().to_string()}\n",
        encoding='utf-8'
    )
    plt.figure(figsize=(6, 4))
    df['y'].value_counts().reindex(['no', 'yes'], fill_value=0).plot.bar()
    plt.title('Phan phoi nhan (no / yes)')
    plt.ylabel('So luong')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / '01_phan_phoi_nhan.png', dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    df['age'].hist(bins=25)
    plt.title('Phan phoi do tuoi khach hang')
    plt.xlabel('Tuoi')
    plt.ylabel('So luong')
    plt.tight_layout()
    plt.savefig(out / '02_phan_phoi_tuoi.png', dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    df.boxplot(column='balance', by='y', grid=False)
    plt.title('So du tai khoan theo nhan')
    plt.suptitle('')
    plt.ylabel('Balance (EUR)')
    plt.tight_layout()
    plt.savefig(out / '03_boxplot_balance.png', dpi=150)
    plt.close()

    if 'pdays' in df:
        n = (df['pdays'] == -1).sum()
        print(f"pdays = -1 (chưa được liên hệ trước đây): {n} bản ghi")


def build_features(df: pd.DataFrame, flow: str) -> pd.DataFrame:
    """Các phép biến đổi theo công thức cố định; KHÔNG fit trên dữ liệu test."""
    # duration chỉ biết sau cuộc gọi; loại để mô phỏng dự đoán trước khi gọi.
    x = df.drop(columns=['y', 'duration'], errors='ignore').copy()
    if flow == 'B':
        # Với bank-full.csv: -1 là 'chưa từng được liên hệ trước đó'.
        x['da_lien_he_truoc'] = (x['pdays'] != -1).astype(int)
        x['pdays'] = x['pdays'].replace(-1, 0)
        x['nhom_tuoi'] = pd.cut(x['age'], bins=[0, 25, 40, 60, np.inf],
                                    labels=['duoi_26', '26_40', '41_60', 'tren_60'])
        # Object để SimpleImputer + OneHotEncoder xử lý thống nhất.
        x['nhom_tuoi'] = x['nhom_tuoi'].astype('object')
    return x


def build_model(x_train: pd.DataFrame, kind: str, params=None) -> Pipeline:
    number_columns = x_train.select_dtypes(include=np.number).columns.tolist()
    category_columns = [col for col in x_train.columns if col not in number_columns]
    number_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    category_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    transform = ColumnTransformer([
        ('number', number_pipe, number_columns),
        ('category', category_pipe, category_columns),
    ])
    if kind == 'Cay quyet dinh':
        classifier = DecisionTreeClassifier(random_state=RANDOM_STATE,
                                            **(params or {}))
    elif kind == 'Naive Bayes':
        classifier = GaussianNB()
    elif kind == 'Luon du doan no':
        classifier = DummyClassifier(strategy='most_frequent')
    else:
        raise ValueError(f'Không hỗ trợ mô hình: {kind}')
    return Pipeline([('preprocess', transform), ('model', classifier)])


def evaluate(model, x, y):
    pred = model.predict(x)
    result = {
        'accuracy': accuracy_score(y, pred),
        'precision_yes': precision_score(y, pred, zero_division=0),
        'recall_yes': recall_score(y, pred, zero_division=0),
        'f1_yes': f1_score(y, pred, zero_division=0),
    }
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(x)[:, 1]
        result['roc_auc'] = roc_auc_score(y, prob)
        result['pr_auc'] = average_precision_score(y, prob)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/bank-full.csv',
                        help='Đường dẫn tệp UCI bank-full.csv')
    parser.add_argument('--out', default='outputs',
                        help='Thư mục ghi biểu đồ và kết quả')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Không thấy {data_path}. Hãy tải bank.zip từ UCI, "
            "giải nén và chép bank-full.csv vào thư mục data/."
        )
    df = pd.read_csv(data_path, sep=';')
    required = {'y', 'age', 'balance', 'pdays', 'duration'}
    if not required.issubset(df.columns):
        raise SystemExit(
            'File không đúng định dạng bank-full.csv (dấu phân cách ;). '
            f"Thiếu cột: {sorted(required - set(df.columns))}"
        )
    if not df['y'].isin(['yes', 'no']).all():
        raise SystemExit('Biến y chỉ được có hai giá trị yes và no.')
    y = (df['y'] == 'yes').astype(int)
    eda(df, out)

    # Giữ nguyên chỉ số để tất cả phương án cùng một cách chia; stratify giữ tỷ lệ yes/no.
    train_idx, other_idx = train_test_split(np.arange(len(df)), test_size=0.40,
                                           random_state=RANDOM_STATE, stratify=y)
    val_idx, test_idx = train_test_split(other_idx, test_size=0.50,
                                       random_state=RANDOM_STATE,
                                       stratify=y.iloc[other_idx])
    y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]
    by_flow = {flow: build_features(df, flow) for flow in ('A', 'B')}
    print(f"Kích thước: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print('Đã bỏ duration để tránh sử dụng dữ liệu chưa có trước cuộc gọi.')

    # Baseline đơn giản và so sánh đầy đủ 2 pipeline x 2 thuật toán đã học.
    configs = [
        ('Baseline: luon no', 'A', 'Luon du doan no', {}),
        ('A + Decision Tree', 'A', 'Cay quyet dinh', {'max_depth': 5, 'min_samples_leaf': 30}),
        ('A + GaussianNB', 'A', 'Naive Bayes', {}),
        ('B + Decision Tree', 'B', 'Cay quyet dinh', {'max_depth': 5, 'min_samples_leaf': 30}),
        ('B + GaussianNB', 'B', 'Naive Bayes', {}),
    ]
    # Thí nghiệm tham số của Cây quyết định trên validation, KHÔNG chạm tới test.
    for depth in (4, 6, 8):
        for leaf in (10, 30):
            configs.append((f'B + Tree (depth={depth}, leaf={leaf})', 'B',
                            'Cay quyet dinh', {'max_depth': depth, 'min_samples_leaf': leaf}))

    records = []
    best = None
    for label, flow, kind, params in configs:
        x = by_flow[flow]
        model = build_model(x.iloc[train_idx], kind, params)
        model.fit(x.iloc[train_idx], y_train)
        scores = evaluate(model, x.iloc[val_idx], y_val)
        records.append({'phuong_an': label, 'luong': flow, 'thuat_toan': kind,
                        'tham_so': json.dumps(params, ensure_ascii=False), **scores})
        # Chọn theo F1 của lớp yes. Loại baseline 'luôn no' khỏi lựa chọn cuối.
        if kind != 'Luon du doan no' and (best is None or scores['f1_yes'] > best[0]):
            best = (scores['f1_yes'], label, flow, kind, params)
        print(f"{label:38s} | validation F1(yes)={scores['f1_yes']:.4f}")

    table = pd.DataFrame(records).sort_values('f1_yes', ascending=False)
    table.to_csv(out / '04_so_sanh_validation.csv', index=False, encoding='utf-8-sig')
    plt.figure(figsize=(10, 6))
    scores_sorted = table.sort_values('f1_yes')
    plt.barh(scores_sorted['phuong_an'], scores_sorted['f1_yes'])
    plt.xlabel('F1-score - lop yes (validation)')
    plt.title('So sanh phuong an tren tap validation')
    plt.tight_layout()
    plt.savefig(out / '05_so_sanh_F1_validation.png', dpi=150)
    plt.close()

    _, name, flow, kind, params = best
    print(f"\nPhương án được chọn bởi validation: {name}")
    # Fit lại trên train+validation với cấu hình đã chọn, test DUY NHẤT MỘT LẦN.
    x = by_flow[flow]
    trainval_idx = np.concatenate([train_idx, val_idx])
    final = build_model(x.iloc[trainval_idx], kind, params)
    final.fit(x.iloc[trainval_idx], y.iloc[trainval_idx])
    pred = final.predict(x.iloc[test_idx])
    prob = final.predict_proba(x.iloc[test_idx])[:, 1]
    test_scores = evaluate(final, x.iloc[test_idx], y_test)
    pd.DataFrame([{'phuong_an': name, **test_scores}]).to_csv(
        out / '06_ket_qua_test.csv', index=False, encoding='utf-8-sig'
    )
    print('\nKẾT QUẢ TRÊN TEST:')
    print(classification_report(y_test, pred,
                                target_names=['no', 'yes'], zero_division=0))
    print(test_scores)
    with (out / '07_phan_tich_loi.txt').open('w', encoding='utf-8') as f:
        f.write(f'Phương án được chọn: {name}\n')
        f.write(f'Kích thước: train={len(train_idx)}, validation={len(val_idx)}, '
                f'test={len(test_idx)}\n')
        f.write(classification_report(y_test, pred,
                    target_names=['no', 'yes'], zero_division=0))
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        f.write(f'\nConfusion matrix (hàng: nhãn thật; cột: dự đoán):\n{cm}\n')
        f.write(f'False Negative: {cm[1, 0]} (yes bị dự đoán thành no)\n')
        f.write(f'False Positive: {cm[0, 1]} (no bị dự đoán thành yes)\n')

    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=['no', 'yes'],
                                            cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix - tap test')
    plt.tight_layout()
    plt.savefig(out / '08_confusion_matrix_test.png', dpi=150)
    plt.close()

    examples = df.iloc[test_idx].drop(columns=['duration']).copy()
    examples['nhan_thuc_te'] = np.where(y_test.to_numpy() == 1, 'yes', 'no')
    examples['du_doan'] = np.where(pred == 1, 'yes', 'no')
    examples['xac_suat_yes'] = prob
    errors = examples[examples['nhan_thuc_te'] != examples['du_doan']]
    errors.to_csv(out / '09_cac_mau_du_doan_sai.csv', index=False, encoding='utf-8-sig')

    joblib.dump(final, out / '10_mo_hinh_da_huan_luyen.joblib')
    (out / 'thong_tin_mo_hinh.json').write_text(
        json.dumps({'flow': flow, 'algorithm': kind, 'params': params,
                    'random_state': RANDOM_STATE, 'test_size': 0.20,
                    'selected_by': 'validation F1 yes',
                    'excluded_column': 'duration',
                    'input_features': x.columns.tolist()},
                   ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'\nĐã lưu toàn bộ biểu đồ, bảng và mô hình tại: {out.resolve()}')


if __name__ == '__main__':
    main()
