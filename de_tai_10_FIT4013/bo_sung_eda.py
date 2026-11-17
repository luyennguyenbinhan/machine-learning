
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Đọc dữ liệu gốc
df = pd.read_csv("data/bank-full.csv", sep=";")
out = Path("outputs")
out.mkdir(exist_ok=True)

# Biểu đồ 1: Tần suất nghề nghiệp
counts = df["job"].value_counts()

plt.figure(figsize=(10, 5))
counts.plot(kind="bar")
plt.title("Phan phoi nghe nghiep khach hang")
plt.xlabel("Nghe nghiep")
plt.ylabel("So luong khach hang")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(out / "eda_tan_suat_nghe_nghiep.png", dpi=200)
plt.close()

# Biểu đồ 2: Ma trận tương quan Spearman
# Không dùng duration vì là thông tin sau cuộc gọi
# Không dùng pdays vì -1 là mã nghiệp vụ đặc biệt
cols = ["age", "balance", "day", "campaign", "previous"]
data = df[cols].copy()
data["y_yes"] = (df["y"] == "yes").astype(int)

corr = data.corr(method="spearman")

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=9)

fig.colorbar(im, ax=ax)
plt.title("Ma tran tuong quan Spearman")
plt.tight_layout()
plt.savefig(out / "eda_ma_tran_tuong_quan.png", dpi=200)
plt.close()

print("Da tao thanh cong 2 bieu do EDA.")
