# Bit-exact int emulation — pipeline per model

Zero training. The student is built once from the teacher and evaluated. Top-1 = 1.0000 on each
of the three models, KL = 0, Gumbel margin = 0 across ~320k held-out positions each
(see [full_int_model.md](full_int_model.md)).

Code paths: `src/difr_expt/train_emulate.py::build_models` (cast + patch),
`src/difr_expt/int_cast.py::IntLinear` (matmul forward),
`src/difr_expt/int_ops_bitexact.py::IntCommitWrap` (RMSNorm/SiLU commit).

---

## Qwen2.5-0.5B — per-row fp8 recipe

Teacher: `RedHatAI/Qwen2.5-0.5B-FP8-dynamic` (per-row fp8 e4m3 weights, per-token fp8 dynamic
activations). 24 transformer blocks, hidden=896, 14 heads × head_dim=64. **n_positions=318,091
→ top-1 = 1.0000.**

```mermaid
flowchart TB
  T0[Teacher CompressedLinear<br/>fp8 e4m3 weight + fp32 weight_scale per row]
  T0 -->|"W_bf16 = W_fp8 · weight_scale"| INIT[student.weight_fp ← W_bf16]
  INIT --> S0[Student IntLinear init]

  subgraph BLOCK[One transformer block, x24]
    direction TB
    X0[hidden_states bf16] --> N1[IntCommitWrap RMSNorm]
    N1 -- bf16 == int30·s --> QKV{q_proj / k_proj / v_proj<br/>IntLinear}
    QKV --> QN[IntCommitWrap q_norm/k_norm]
    QN --> RoPE[RoPE]
    RoPE --> SDPA["Q · K.T → softmax → P · V<br/>eager bf16 path"]
    SDPA --> O[IntLinear o_proj]
    O --> R1((+))
    X0 --> R1
    R1 --> N2[IntCommitWrap RMSNorm]
    N2 --> GU{gate_proj / up_proj<br/>IntLinear}
    GU --> SI[IntCommitWrap SiLU]
    SI --> MUL((×))
    MUL --> D[IntLinear down_proj]
    D --> R2((+))
    R1 --> R2
    R2 --> Y[hidden_states bf16]
  end

  S0 -.weight_fp.- BLOCK

  Y --> LMH[lm_head F.linear bf16] --> LOG[logits]
  LOG --> CMP[top-1 vs teacher = 1.0000]

  classDef commit fill:#cfe8d2,stroke:#1f9b54
  classDef matmul fill:#d6e4f4,stroke:#2a5fa0
  class N1,N2,QN,SI commit
  class QKV,O,GU,D,LMH matmul
```

**Inside each `IntLinear` (Qwen0.5B path):**

```mermaid
flowchart LR
  x[x bf16] --> A1[per-token fp8 e4m3 quant<br/>absmax in input dtype bf16<br/>round to 256-level grid]
  A1 -- x_q bf16 --> Q1[per-token int30 commit<br/>x_int = round x_q/scale<br/>scale = absmax/2^29]
  W[weight_fp bf16<br/>from init_from_teacher] --> Q2[per-row int30 commit<br/>w_int + w_scale]
  Q1 -- bf16 dequant identity --> FL[F.linear bf16 GEMM<br/>cuBLAS deterministic]
  Q2 -- bf16 dequant identity --> FL
  FL --> y[y bf16]

  classDef path fill:#cfe8d2,stroke:#1f9b54
  class A1,Q1,Q2,FL path
```

CLI: `--activation-fp8 --int-matmul-path --int-nonmatmul-bitexact`.

---

## Llama-3.1-8B-Instruct — same per-row fp8 recipe

Teacher: `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic` (per-row fp8 e4m3 weights, per-token
fp8 dynamic activations). 32 transformer blocks, hidden=4096, GQA 32 heads / 8 KV heads ×
head_dim=128. **n_positions=317,147 → top-1 = 1.0000.**

```mermaid
flowchart TB
  T0[Teacher CompressedLinear<br/>fp8 e4m3 weight + fp32 weight_scale per row]
  T0 -->|"W_bf16 = W_fp8 · weight_scale"| INIT[student.weight_fp ← W_bf16]
  INIT --> S0[Student IntLinear init]

  subgraph BLOCK[One transformer block, x32]
    direction TB
    X0[hidden_states bf16] --> N1[IntCommitWrap RMSNorm]
    N1 -- bf16 == int30·s --> QKV{q_proj / k_proj / v_proj<br/>IntLinear}
    QKV --> RoPE[RoPE no q_norm/k_norm in Llama]
    RoPE --> SDPA["Q · K.T → softmax → P · V<br/>eager bf16 path<br/>GQA: 8 KV heads repeated 4x"]
    SDPA --> O[IntLinear o_proj]
    O --> R1((+))
    X0 --> R1
    R1 --> N2[IntCommitWrap RMSNorm]
    N2 --> GU{gate_proj / up_proj<br/>IntLinear}
    GU --> SI[IntCommitWrap SiLU]
    SI --> MUL((×))
    MUL --> D[IntLinear down_proj]
    D --> R2((+))
    R1 --> R2
    R2 --> Y[hidden_states bf16]
  end

  S0 -.weight_fp.- BLOCK

  Y --> LMH[lm_head F.linear bf16] --> LOG[logits]
  LOG --> CMP[top-1 vs teacher = 1.0000]

  classDef commit fill:#cfe8d2,stroke:#1f9b54
  classDef matmul fill:#d6e4f4,stroke:#2a5fa0
  class N1,N2,SI commit
  class QKV,O,GU,D,LMH matmul
```

`IntLinear` internals identical to Qwen0.5B (per-token fp8 + int30 + F.linear).
CLI: `--activation-fp8 --int-matmul-path --int-nonmatmul-bitexact`.

---

## Qwen3-8B — block-fp8 kernel-path recipe

Teacher: `Qwen/Qwen3-8B-FP8` (per-128×128-tile fp8 e4m3 weights with `weight_scale_inv`, per-128-block
fp8 dynamic activations via Triton `act_quant`). 36 transformer blocks, hidden=4096, GQA 32 heads
/ 8 KV heads × head_dim=128, **with** `q_norm`/`k_norm`. **n_positions=325,731 → top-1 = 1.0000.**

```mermaid
flowchart TB
  T0[Teacher FP8Linear<br/>fp8 e4m3 weight + fp32 weight_scale_inv 128x128 tiles]
  T0 -->|"W_bf16 = W_fp8 · Sexp<br/>(Sexp = scale_inv repeat-interleaved)"| INIT[student.weight_fp ← W_bf16]
  T0 -->|"stash original fp8 + weight_scale_inv<br/>on IntLinear buffers"| STASH[block_fp8_weight, block_fp8_scale_inv]
  INIT --> S0[Student IntLinear init]
  STASH --> S0

  subgraph BLOCK[One transformer block, x36]
    direction TB
    X0[hidden_states bf16] --> N1[RMSNorm bf16<br/>wrappers off — see report]
    N1 --> QKV{q_proj / k_proj / v_proj<br/>IntLinear block_fp8_kernel_path}
    QKV --> QN[q_norm/k_norm bf16<br/>Qwen3-specific]
    QN --> RoPE[RoPE]
    RoPE --> SDPA["Q · K.T → softmax → P · V<br/>eager bf16 path<br/>GQA 32/8 x 128"]
    SDPA --> O[IntLinear o_proj block_fp8_kernel_path]
    O --> R1((+))
    X0 --> R1
    R1 --> N2[RMSNorm bf16]
    N2 --> GU{gate_proj / up_proj<br/>IntLinear block_fp8_kernel_path}
    GU --> SI[SiLU bf16]
    SI --> MUL((×))
    MUL --> D[IntLinear down_proj block_fp8_kernel_path]
    D --> R2((+))
    R1 --> R2
    R2 --> Y[hidden_states bf16]
  end

  S0 -.fp8 W + scale_inv.- BLOCK

  Y --> LMH[lm_head F.linear bf16] --> LOG[logits]
  LOG --> CMP[top-1 vs teacher = 1.0000]

  classDef matmul fill:#d6e4f4,stroke:#2a5fa0
  class QKV,O,GU,D,LMH matmul
```

**Inside each `IntLinear` (Qwen3 block-fp8 path):**

```mermaid
flowchart LR
  x[x bf16] --> AQ["act_quant Triton kernel<br/>per-128-block absmax in fp32<br/>x → fp8_e4m3fn + per-block fp32 scale"]
  AQ -- "qx fp8, sx fp32 [M, K/128]" --> KER
  STW[stashed block_fp8_weight<br/>fp8 e4m3] --> KER
  STS[stashed block_fp8_scale_inv<br/>fp32 N/128 × K/128] --> KER
  KER["w8a8_block_fp8_matmul Triton kernel<br/>for each K-block of 128:<br/>partial = tl.dot a_fp8, b_fp8 → fp32<br/>acc += partial · a_s · b_s<br/>(scales applied per K-block, after dot)"]
  KER --> y[y bf16]

  classDef path fill:#cfe8d2,stroke:#1f9b54
  class AQ,KER path
```

CLI: `--activation-block-fp8 --block-fp8-kernel-path` (no `--int-nonmatmul-bitexact` — the int30
wrapper on per-head `q_norm`/`k_norm` `[..., 128]` shape loses bf16 LSBs; the kernel path doesn't
need it). ZK spec: replace the Triton kernel with a per-K-block fp32 emulation (verified 99.99%
bf16-bit-exact in standalone test; see [full_int_model.md](full_int_model.md) § "Block-fp8 GEMM emulation").

---

## Surface → int commitment

| Surface | Qwen2.5-0.5B / Llama-3.1-8B | Qwen3-8B |
|---|---|---|
| Activation entering each Linear | per-token fp8 e4m3 (256-level → int8 LUT) | per-128-block fp8 e4m3 (Triton `act_quant`) |
| Weight in each Linear | per-row int30 + fp32 scale (round-trip identity on bf16) | per-128×128-tile fp8 + fp32 `weight_scale_inv` (stashed from teacher) |
| Matmul kernel | `F.linear` bf16 GEMM (deterministic) | `w8a8_block_fp8_matmul` (per-K-block fp32 accumulator) |
| RMSNorm / SiLU input | per-token int30 + fp32 scale (`IntCommitWrap`) | bf16 (kernel path covers the commitment via upstream IntLinear outputs) |
| Attention Q/K/V | int30 from upstream `q_proj`/`k_proj`/`v_proj` | per-128-block fp8 from upstream |
| Matmul output → next layer | bf16, re-committed at next IntLinear's input | same |
