# BEAST Tokenizer

B-Spline токенизатор для преобразования траекторий в дискретные токены для VLM.

## Быстрый старт

### 1. Создание токенизатора

```python
from beast.bspline_tokenizer import BSpline_Tokenizer

tokenizer = BSpline_Tokenizer(
    num_dof=14,             # Количество DoF
    num_basis=10,           # Количество базисных функций B-сплайна
    seq_len=50,             # Длина временной последовательности
    vocab_size=256,         # Размер словаря для дискретизации
    gripper_indices=[6, 13],    # Индексы DoF для гриппера
    device="cuda"
)

# Обязательно установить размер словаря VLM
tokenizer.update_vlm_vocab_size(vlm_vocab_size=32000)
```
---

## Основной workflow

### 2. Подготовка: фит границ параметров (один раз перед обучением)

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
tokenizer = BSpline_Tokenizer.from_pretrained("./saved_tokenizer", device="cuda")
```

> Устанавливаются `w_min` и `w_max` для нормализации параметров B-сплайнов на основе ваших данных.

---

### 3. Обучение: encode → tokens → LLM

```python
# В training_step:
def training_step(self, batch):
    # 3.1. Encode: траектории → MP токены
    mp_tokens, params = tokenizer.encode(
        batch["actions"],           # [B, T, DoF]
        update_bounds=False         # False после fit_parameters!
    )
    # mp_tokens: [B, num_basis * num_dof]
    
    # 3.2. Конвертация в LLM токены
    llm_tokens = tokenizer.tokens_to_llm_tokens(mp_tokens)
    # llm_tokens: [B, num_basis * num_dof], значения в диапазоне [0, vlm_vocab_size-1]
    
    # 3.3. Передать llm_tokens в VLM для обучения
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
    actions = tokenizer.reconstruct_from_llm_tokens(
        predicted_llm_tokens,
        times=None,              # Использует self.times по умолчанию
        init_p=prev_action       # Опционально: для непрерывности траекторий
    )
    # actions: [B, seq_len, DoF]
    
    return actions
```

---

## Схема методов

```
┌─────────────────────────────────────────────────────────────┐
│                    ПОДГОТОВКА (1 раз)                       │
├─────────────────────────────────────────────────────────────┤
│ 1. __init__()                     - создать токенизатор     │
│ 2. update_vlm_vocab_size()        - установить vocab VLM    │
│ 3. fit_parameters()               - фит границ w_min/w_max  │
│ 4. save_pretrained()              - сохранить               │
│ 5. from_pretrained()              - загрузить               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      ОБУЧЕНИЕ (цикл)                        │
├─────────────────────────────────────────────────────────────┤
│ 1. encode(trajs, update_bounds=False)                       │
│    ↓ [B, T, DoF] → [B, num_basis*num_dof]                   │
│ 2. tokens_to_llm_tokens(mp_tokens)                          │
│    ↓ MP tokens → LLM tokens                                 │
│ 3. VLM обучается на llm_tokens                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    ИНФЕРЕНС (цикл)                          │
├─────────────────────────────────────────────────────────────┤
│ 1. VLM генерирует llm_tokens                                │
│    ↓                                                        │
│ 2. reconstruct_from_llm_tokens(llm_tokens)                  │
│    ↓ LLM tokens → [B, seq_len, DoF]                         │
│ 3. Выполнить действия                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Полезные методы визуализации

- `visualize_reconstruction_error(trajs)` - визуализировать качество реконструкции
- `visualize_reconstruction_error_with_llm_tokenizer(trajs)` - с учетом LLM токенизации
- `compute_reconstruction_error(trajs)` - вычислить MSE ошибку

## Пример полного пайплайна

```python
# === 1. Подготовка ===
tokenizer = BSpline_Tokenizer(num_dof=7, num_basis=10, gripper_indices=[6])
tokenizer.update_vlm_vocab_size(32000)
tokenizer.fit_parameters(train_dataloader, max_samples=1000)
tokenizer.save_pretrained("./tokenizer_fitted")

# === 2. Обучение ===
tokenizer = BSpline_Tokenizer.from_pretrained("./tokenizer_fitted")
for batch in train_loader:
    mp_tokens, _ = tokenizer.encode(batch["actions"], update_bounds=False)
    llm_tokens = tokenizer.tokens_to_llm_tokens(mp_tokens)
    loss = train_vlm(llm_tokens)

# === 3. Инференс ===
tokenizer = BSpline_Tokenizer.from_pretrained("./tokenizer_fitted")
for obs in test_env:
    llm_tokens = vlm.generate(obs)
    actions = tokenizer.reconstruct_from_llm_tokens(llm_tokens)
    env.step(actions)
```
