# spmv_cfg.py
# －－ SpMV 模擬共用設定（所有腳本都 import 這裡）－－

# 硬體 / 格式設定
LANES = 8
DENSE_THRESHOLD = 0.6
BCSR_BLOCK = 4

# 評估的格式與資料流（與 best_choices 完全一致）
FORMATS = ["Dense", "CSR", "BCSR", "DIA", "COO", "LIL"]
FLOWS   = ["IP", "OP"]

# 傳給 estimate_spmv 的固定參數
SPMV_KW = dict(
    lanes=LANES,
    block_size=BCSR_BLOCK,
    dense_threshold=DENSE_THRESHOLD,
)

# 方便其他腳本組出完整 label 空間（int label 的 pack/unpack）
def pack_label(fmt: str, flow: str) -> int:
    return FORMATS.index(fmt) * len(FLOWS) + FLOWS.index(flow)

def unpack_label(lbl: int) -> tuple[str, str]:
    i, j = divmod(int(lbl), len(FLOWS))
    return FORMATS[i], FLOWS[j]

def label_space() -> list[int]:
    return [pack_label(f, d) for f in FORMATS for d in FLOWS]
