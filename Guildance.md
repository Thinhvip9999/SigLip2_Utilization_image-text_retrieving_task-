# Hướng dẫn chạy notebook Colab trên server (cấu hình GPU RTX 5880, CUDA 12.8)

> Tài liệu này được điều chỉnh chính xác theo môi trường của bạn (Ubuntu + NVIDIA RTX 5880 Ada Generation, CUDA 12.8, driver 570.133.07). Bạn chỉ cần copy–paste theo thứ tự.

---

## 0) Yêu cầu tối thiểu

* **OS:** Ubuntu 20.04/22.04/24.04
* **Python:** 3.10 (khuyến nghị)
* **GPU:** NVIDIA RTX 5880 (đã xác nhận CUDA 12.8 hoạt động)
* **Ổ đĩa trống:** ≥ 10 GB

---

## 1) SSH & tạo thư mục dự án

```bash
ssh <user>@<server-ip>
mkdir -p ~/projects/siglip2 && cd ~/projects/siglip2
```

---

## 2) Cài gói hệ thống

```bash
sudo apt update && sudo apt install -y \
  python3-venv python3-dev build-essential git git-lfs \
  wget curl unzip ffmpeg libgl1 libglib2.0-0

git lfs install
```

---

## 3) Tạo môi trường ảo Python

Bạn có thể chọn **venv** (nhẹ, có sẵn) hoặc **conda** (quản lý môi trường mạnh hơn). Chọn **một** trong hai cách dưới đây.

### Cách A — venv (khuyến nghị nếu đã quen)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -V  # nên ra Python 3.10.x
```

### Cách B — conda (nếu bạn dùng Anaconda/Miniconda/Mamba)

```bash
# Tạo môi trường tên: siglip2 (Python 3.10)
conda create -y -n siglip2 python=3.10
conda activate siglip2
python -V
```

> Dù dùng venv hay conda, **các lệnh cài đặt phía dưới là giống nhau** (dùng `pip` bên trong môi trường đang kích hoạt).

---

## 4) Cài JAX + PyTorch & thư viện

Notebook của bạn dùng **JAX** (GPU) và cài thêm một số package từ repo **google-research/big\_vision**. Trên server CUDA 12.8, cài như sau:

### 4.1 Cài **JAX GPU** (CUDA 12.x)

```bash
pip install --upgrade pip wheel setuptools
# JAX GPU cho CUDA 12.x (Google cung cấp wheels tại jax_cuda_releases)
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

> Nếu sau đó `import jax; jax.devices()` không thấy GPU, kiểm tra lại driver/CUDA. (Trên máy bạn hiện CUDA 12.8 và driver 570.133.07.)

### 4.2 Cài **PyTorch** khớp CUDA 12.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4.3 Clone **big\_vision** và cài requirements của repo

```bash
# Ở thư mục dự án: ~/projects/siglip2
git clone --branch=main --depth=1 https://github.com/google-research/big_vision
pip install -r big_vision/big_vision/requirements.txt
```

### 4.4 Cài các thư viện bổ sung mà notebook của bạn có dùng

```bash
pip install --upgrade \
  transformers>=4.44,<5 accelerate pillow gdown faiss-cpu \
  numpy pandas matplotlib opencv-python tqdm pyyaml requests einops \
  jupyter nbconvert papermill ipykernel
```

### 4.5 Cài **gsutil** để tải checkpoint từ GCS (notebook của bạn có dùng)

```bash
# Chọn 1 trong 2 cách (khuyến nghị Cách 1)
# Cách 1: Cài Google Cloud SDK (có kèm gsutil)
# (Tham khảo thêm docs chính thức nếu cần xác thực tài khoản. Public bucket thì không cần.)
curl -sSL https://sdk.cloud.google.com | bash
exec -l $SHELL
which gsutil && gsutil version

# Cách 2: (ít khuyến nghị) cài gsutil qua pip
pip install gsutil
```

> Public bucket `gs://big_vision/siglip2/` có thể tải **không cần** đăng nhập. Nếu bạn gặp lỗi quota/auth, chạy `gcloud auth login` rồi `gcloud auth application-default login`.

---

## 5) Upload notebook & dữ liệu + clone repo

```bash
# Từ máy cá nhân:
scp "SigLip2_Process_with_non_squared_image.ipynb" <user>@<server-ip>:~/projects/siglip2/

# Trên server (đảm bảo đã clone repo ở bước 4.3):
mkdir -p data inputs outputs logs runs models
```

**Lưu ý tên thư mục**: notebook của bạn giả định có thư mục `big_vision/` (đã clone ở bước 4.3). Hãy giữ nguyên tên này để các import kiểu `big_vision.*` hoạt động đúng.

---

## 6) Biến môi trường (tăng tốc caching)

```bash
export HF_HOME="$PWD/.hf-cache"
export HUGGINGFACE_HUB_CACHE="$PWD/.hf-cache"
export TORCH_HOME="$PWD/.torch-cache"
mkdir -p "$HF_HOME" "$TORCH_HOME"
```

Thêm vào `.bashrc` để tự động:

```bash
echo "export HF_HOME=$PWD/.hf-cache" >> ~/.bashrc
echo "export HUGGINGFACE_HUB_CACHE=$PWD/.hf-cache" >> ~/.bashrc
echo "export TORCH_HOME=$PWD/.torch-cache" >> ~/.bashrc
```

---

## 7) Cách chạy notebook

### 7.1 Dùng **Papermill** (giống Colab “Run All”)

Notebook của bạn có cell cuối yêu cầu **nhập prompt**. Bạn có thể truyền prompt từ terminal thông qua tham số (không cần nhập tay).

**Thêm cell tham số** (ở đầu notebook) với nội dung:

```python
# Parameters
PROMPT = ""
OUTFILE = "020.jpg"  # tên file đầu ra mong muốn
```

**Sửa cell cuối** để dùng biến `PROMPT` thay vì `input()`:

```python
query = PROMPT if PROMPT else input("Nhập truy vấn: ")
top_results = search_keyframes(query, k=10)
# Tùy thuộc logic notebook, lưu ảnh top-1 ra OUTFILE
from pathlib import Path
import shutil
best = top_results[0]
Path("outputs").mkdir(exist_ok=True)
shutil.copy2(best["filepath"], f"outputs/{OUTFILE}")
print(f"Đã lưu ảnh kết quả: outputs/{OUTFILE}")
```

**Chạy bằng Papermill và truyền prompt + outfile từ terminal:**

```bash
source .venv/bin/activate  # hoặc conda activate siglip2
papermill \
  "SigLip2_Process_with_non_squared_image.ipynb" \
  "runs/$(date +%Y%m%d_%H%M%S)_output.ipynb" \
  -k python3 \
  -p PROMPT "A medium-shot, eye-level photo of a shoreline with a makeshift fence of dead, broken branches and wooden posts in the foreground, with the calm sea and a pale sky in the background. The text \"HTV9 HD\" and a timestamp are visible in the top right corner." \
  -p OUTFILE "020.jpg"
```

### 7.2 Dùng **script .py** (dễ cron/systemd + nhập prompt từ terminal)

Chuyển notebook sang script và thêm argparse để đọc prompt + outfile:

```bash
jupyter nbconvert --to script "SigLip2_Process_with_non_squared_image.ipynb"
```

Mở file `SigLip2_Process_with_non_squared_image.py`, chèn đoạn sau ở **phần main** (sau khi đã định nghĩa `search_keyframes`):

```python
import argparse, os, shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Câu truy vấn mà trước đây bạn nhập ở cell cuối")
    parser.add_argument("--outfile", type=str, default="020.jpg", help="Tên file ảnh đầu ra trong thư mục outputs/")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    results = search_keyframes(args.prompt, k=args.topk)
    assert len(results) > 0, "Không tìm thấy kết quả nào"
    best = results[0]

    Path("outputs").mkdir(exist_ok=True)
    outpath = os.path.join("outputs", args.outfile)
    shutil.copy2(best["filepath"], outpath)
    print(f"
✅ Đã lưu ảnh top-1 vào: {outpath}")

if __name__ == "__main__":
    main()
```

Chạy thử với prompt mẫu và kỳ vọng đầu ra `020.jpg`:

```bash
python SigLip2_Process_with_non_squared_image.py \
  --prompt "A medium-shot, eye-level photo of a shoreline with a makeshift fence of dead, broken branches and wooden posts in the foreground, with the calm sea and a pale sky in the background. The text \"HTV9 HD\" and a timestamp are visible in the top right corner." \
  --outfile 020.jpg
```

Kết quả sẽ lưu ở: `outputs/020.jpg`.

---

## 8) Kiểm tra GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

Kết quả nên báo `CUDA: True` và `Device: NVIDIA RTX 5880 Ada Generation`.

---

## 9) Cấu trúc thư mục gợi ý

```text
~/projects/siglip2/
├── .venv/
├── data/
├── inputs/
├── outputs/
├── logs/
├── models/
├── runs/
├── SigLip2_Process_with_non_squared_image.ipynb
├── SigLip2_Process_with_non_squared_image.py  (sau khi convert)
└── requirements.txt
```

---

## 10) Lưu môi trường để tái sử dụng (và nói rõ về `requirements.txt`)

Repo **big\_vision** đã có sẵn `big_vision/big_vision/requirements.txt` (bạn đã cài ở bước 4.3). File này bao gồm các phụ thuộc **cốt lõi** của dự án. Tuy nhiên, để chạy notebook của bạn trên server, ta còn cần một số phụ thuộc **bổ sung** (Papermill, FAISS, OpenCV, v.v.). Bạn có thể đóng gói chúng vào một file `requirements_extra.txt` như sau (đính kèm để tiện copy–paste):

```text
# requirements_extra.txt — bổ sung cho môi trường server chạy notebook
# (JAX/PyTorch cài riêng theo hướng dẫn vì phụ thuộc CUDA/driver)
transformers>=4.44,<5
accelerate
pillow
faiss-cpu
opencv-python
numpy
pandas
matplotlib
 tqdm
pyyaml
requests
einops
jupyter
nbconvert
papermill
ipykernel
# Tuỳ chọn: nếu tải checkpoint từ GCS
# gsutil  # hoặc cài Google Cloud SDK như hướng dẫn ở bước 4.5
```

Lưu file và cài:

```bash
pip install -r requirements_extra.txt
```

Để snapshot toàn bộ môi trường hiện tại (sau khi cài xong mọi thứ), bạn vẫn có thể:

```bash
pip freeze > requirements.txt
```

Sau này phục dựng:

```bash
# venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# hoặc conda
conda create -y -n siglip2 python=3.10 && conda activate siglip2
pip install -r requirements.txt
```

---

## 11) Ghi chú quan trọng theo đúng notebook của bạn

* Notebook **clone** `google-research/big_vision` và **import** theo dạng `big_vision.*` ⇒ đảm bảo thư mục `big_vision/` tồn tại ở **cùng cấp** với notebook.
* Notebook **tải checkpoint** từ `gs://big_vision/siglip2/{CKPT}` về `/tmp/` ⇒ hãy cài **gsutil** (bước 4.5). Với public bucket, không bắt buộc đăng nhập.
* Notebook dùng **FAISS (faiss-cpu)** để lập chỉ mục và truy vấn keyframes ⇒ đã thêm vào requirements.
* Khi chạy ở server, bạn có thể dùng `nohup`/`tmux` để chạy nền, hoặc convert sang `.py` và dùng `cron/systemd`.

---

### Kết thúc

Bạn có thể bắt đầu bằng Papermill (**7.1**) hoặc script (**7.2**). Prompt mẫu đã được nhúng sẵn; file kết quả kỳ vọng: **`outputs/020.jpg`**.
