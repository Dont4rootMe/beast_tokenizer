# BEAST Tokenizer

B-Spline токенизатор для преобразования траекторий в дискретные токены для VLM.

## Быстрый старт

### 1. Создание токенизатора

```python
from beast.beast_bspline_tokenizer import BEASTBsplineTokenizer
from beast.beast_bspline_tokenizer import BEASTBsplineTokenizer

tokenizer = BEASTBsplineTokenizer(
tokenizer = BEASTBsplineTokenizer(
    num_dof=14,             # Количество DoF
    num_basis=10,           # Количество базисных функций B-сплайна
    seq_len=50,             # Длина временной последовательности
    vocab_size=256,         # Размер словаря для дискретизации
    gripper_indices=[6, 13],    # Индексы DoF для гриппера
    gripper_zero_order=True,    # Специальный обработчик для гриппера 
    device="cuda"
)

# Обязательно установить размер словаря VLM
tokenizer.update_vlm_vocab_size(vlm_vocab_size=32000)
```

> `gripper_indices` является необязательным и в принципе требуется только, если указывается специальный обработчик для гриппера (gripper_zero_order=True)

Если в определении объекта BEASTBsplineTokenizer указывается размер словаря vlm (`update_vlm_vocab_size`), тогда токенизатор отображает токены как $\textrm{token} + \textrm{vlm\_vocab\_size} - \textrm{beast\_vocab\_size}$ – отправляет токены в конец словаря

### 1.1 Обучение BPE для BEAST

После подбора `w_min`/`w_max` можно обучить расширенный BEAST токенизатор,
который поверх дискретных бин-последовательностей строит текстовое
представление и обучает BPE-словарь по аналогии с FAST.

```python
from beast.beast_bspline_bpe_tokenizer import BEASTBsplineBPETokenizer

# 1) Используем обученный BSpline токенизатор (как выше)
...

# 2) Создаём BEAST + BPE токенизатор и обучаем внутренний BPE словарь
beast_tokenizer = BEASTBsplineBPETokenizer.from_beast(tokenizer, bpe_vocab_size=2048)
state = beast_tokenizer.fit_from_trajectories(train_dataloader)

# 3) При необходимости можно получить состояние BPE напрямую
print(state.min_token, state.max_token)
```

> Если вызвать методы кодирования до `fit_from_trajectories`,
> токенизатор сообщит об ошибке. Это гарантирует, что BPE
> словарь всегда обучен перед использованием.

---

## Основной workflow

### 2. Подготовка: фит параметров для нормализации bspline кодирования

```python
# Вариант A: Используя dataloader
tokenizer.fit_parameters(
    dataloader=train_dataloader,
    max_samples=1000,
    verbose=True
)

# Сохранить токенизатор с фитнутыми границами
tokenizer.save_pretrained("./saved_tokenizer")

# Вариант B: Загрузить готовый токенизатор
tokenizer = BEASTBsplineTokenizer.from_pretrained("./saved_tokenizer", device="cuda")
```

> Устанавливаются `w_min` и `w_max` для нормализации параметров B-сплайнов на основе ваших данных.

---

### 3. Обучение vlm: encode → tokens → LLM

```python

...
# обновляем размер словаря vlm
tokenizer.set_llm_vocab_size(
    llm_vocab_size=vlm.embeddings.weights.size(0)
)

...

# В training_step:
def training_step(self, batch):
    # 3.1. Encode: траектории → MP токены
    llm_tokens, params = tokenizer.encode(
        batch["actions"],           # [B, T, DoF]
        update_bounds=False         # False после fit_parameters!
    ) # [B, num_basis * num_dof]
    
    # 3.2. Передать llm_tokens в VLM для обучения
    loss = vlm(llm_tokens, ...)
    return loss
```

---

### 4. Инференс: LLM → tokens → траектории

```python
# В forward/inference:
def forward(self, obs):
    # 4.1. VLM генерирует LLM токены
    predicted_llm_tokens = vlm.generate(obs)  # [B, num_basis * num_dof]
    
    # 4.2. Decode: LLM токены → траектории
    actions = tokenizer.reconstruct_traj(
        predicted_llm_tokens,
        times=None,              # Использует self.times по умолчанию
        init_p=prev_action       # Опционально: для непрерывности траекторий
    )
    # actions: [B, seq_len, DoF]
    
    return actions
```

### 4.1 Работа с BEAST + BPE

```python
# BPE кодирование: траектории → текстовые токены
bpe_tokens, params = beast_tokenizer.encode(batch["actions"], update_bounds=False)

# Декодирование: BPE токены → траектория
reconstructed = beast_tokenizer.reconstruct_traj(bpe_tokens)

# При необходимости получить MP-представление без BPE
mp_tokens, _ = beast_tokenizer.encode_to_mp_tokens(batch["actions"], update_bounds=False)
```

### CLI-скрипт для обучения BEAST / BEAST + BPE

В каталоге `train/` лежит основной пайплайн `train_beast.py`. Он сначала фитит
базовый BSpline BEAST токенизатор (границы `w_min`/`w_max`), затем (по умолчанию)
обучает поверх него BPE и считает ошибку реконструкции на eval-датасетах.
Пример запуска (см. `train.sh`):

```
PYTHONPATH=.:MP_lite_PyTorch python train/train_beast.py \
    --batch-size 32 --num-basis 5 --vocab-size 256 --degree 3 \
    --fit-beast-max-samples 5000 --fit-bpe-max-samples 25000 \
    --bpe-vocab-size 2048 --max-eval-samples 2500 --num-workers 8
```

Дефолты: `--degree 3`, `--num-basis 5`, `--vocab-size 256` (как в статье BEAST,
но с `num_basis`, подобранным под чанк из 10 шагов). Требования: `num_basis >= degree + 1`
и `num_basis <= длина чанка`; при `num_basis > seq_len` токенизатор печатает предупреждение,
потому что фит вырождается в бининг с нулевым паддингом. `--num-dof 26` отрезает нулевой
паддинг 32-мерного пространства действий. `--fit-*-max-samples` и `--max-eval-samples`
считают батчи. `--no-train-bpe` пропускает BPE. Чекпоинты пишутся в
`beast_tokenizer_checkpoint` и `beast_bpe_tokenizer_checkpoint`, метрики в `eval_results/`.

### Перебор конфигураций и smoke-тест

```
PYTHONPATH=.:MP_lite_PyTorch python train/sweep_beast.py \
    --num-basis-grid 4,5,6,8 --degree-grid 2,3 \
    --fit-beast-max-samples 1000 --max-eval-samples 300 --out-dir sweep_results
```

Sweep один раз кеширует батчи в память, перебирает пары `(num_basis, degree)` без BPE
и после каждой пары дописывает `sweep_results/summary.{csv,json}` (столбцы: `config`,
`tokens_pre_bpe`, `mean_tokens`, `mean_l2`, `mean_l1`, `max_abs_err`). Опция
`--bpe-config 5,3` дообучает BPE для выбранной пары и добавляет строки с пост-BPE длиной.

`python train/smoke_synthetic.py` прогоняет весь стек на синтетических траекториях
без датасетов и печатает `SMOKE OK`.

Известные ограничения: BPE-токены возвращаются в диапазоне `[0, bpe_vocab_size)` и не
сдвигаются в словарь VLM внутри репозитория; `to(device)` не переносит `self.times`.

---

## Полезные методы визуализации

- `visualize_reconstruction_error(trajs)` - визуализировать качество реконструкции
- `visualize_reconstruction_error_with_llm_tokenizer(trajs)` - с учетом LLM токенизации
- `compute_reconstruction_error(trajs)` - вычислить MSE ошибку

## Пример полного пайплайна

```python
# === 1. Подготовка ===
tokenizer = BEASTBsplineTokenizer(num_dof=7, num_basis=10, gripper_indices=[6])
tokenizer = BEASTBsplineTokenizer(num_dof=7, num_basis=10, gripper_indices=[6])
tokenizer.update_vlm_vocab_size(32000)
tokenizer.fit_parameters(train_dataloader, max_samples=1000)
tokenizer.save_pretrained("./tokenizer_fitted")

# === 2. Обучение ===
tokenizer = BEASTBsplineTokenizer.from_pretrained("./tokenizer_fitted")
for batch in train_loader:
    mp_tokens, _ = tokenizer.encode(batch["actions"], update_bounds=False)
    llm_tokens = tokenizer.tokens_to_llm_tokens(mp_tokens)
    loss = train_vlm(llm_tokens)

# === 3. Инференс ===
tokenizer = BEASTBsplineTokenizer.from_pretrained("./tokenizer_fitted")
for obs in test_env:
    llm_tokens = vlm.generate(obs)
    actions = tokenizer.reconstruct_from_llm_tokens(llm_tokens)
    env.step(actions)
```

## Пример с BPE

```python
from beast.beast_bspline_bpe_tokenizer import BEASTBsplineBPETokenizer
import torch

# load from checkpoint
tokenizer = BEASTBsplineBPETokenizer.from_pretrained("./beast_bpe_tokenizer_checkpoint", device="cuda")
tokenizer.to('cpu')

# encode - decode actions
actions = torch.randn([1, 10, 32], device="cpu")
tokens, _ = tokenizer.encode(actions)
actions_reconstructed = tokenizer.reconstruct_traj(tokens)
```