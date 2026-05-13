# Full int model — pipeline diagrams

Diagrams render natively on GitHub. Code paths: `src/difr_expt/train_emulate.py` (build + eval),
`src/difr_expt/int_cast.py` (`IntLinear`), `src/difr_expt/int_ops_bitexact.py` (`IntCommitWrap`).

## 1. Top-level pipeline

```mermaid
flowchart LR
  T[Published fp8 teacher<br/>RedHatAI/...-FP8-dynamic or<br/>Qwen/Qwen3-8B-FP8]
  B[Base model bf16<br/>Qwen2.5-0.5B / Llama-3.1-8B / Qwen3-8B]
  T --> Init
  B --> Patch
  subgraph BUILD[build_models]
    Init[init_from_teacher<br/>copy fp8→bf16 dequant weights<br/>stash fp8+scale_inv for block-fp8]
    Patch[patch_model_int_cast<br/>nn.Linear → IntLinear x N]
    Wrap[patch_model_int_bitexact<br/>RMSNorm/SiLU → IntCommitWrap]
    Init --> Patch --> Wrap
  end
  Wrap --> S[Student model]
  S --> Loop{steps > 0?}
  Loop -- yes --> Train[Training loop<br/>logit-KL + Gumbel margin loss]
  Loop -- no --> Eval[evaluate]
  Train --> Eval
  Eval --> M[top-1, top-5, KL, Gumbel margin<br/>L1/L2/MAE, cosine]
  Eval -.compares.- T
```

## 2. Per-block forward pass — where int commits live

```mermaid
flowchart TB
  X0[hidden_states bf16] --> N1[IntCommitWrap<br/>input_layernorm RMSNorm]
  N1 -- bf16 = int30·s --> Q[IntLinear q_proj]
  N1 -- bf16 = int30·s --> K[IntLinear k_proj]
  N1 -- bf16 = int30·s --> V[IntLinear v_proj]
  Q --> QN[IntCommitWrap<br/>q_norm RMSNorm]
  K --> KN[IntCommitWrap<br/>k_norm RMSNorm]
  QN --> RoPE[RoPE rotation]
  KN --> RoPE
  RoPE --> SDPA["Q · K.T → softmax → P · V<br/>(eager bf16 path; Q/K/V already int30-committed)"]
  V --> SDPA
  SDPA --> O[IntLinear o_proj]
  O --> R1((+))
  X0 --> R1
  R1 --> N2[IntCommitWrap<br/>post_attention_layernorm RMSNorm]
  N2 --> G[IntLinear gate_proj]
  N2 --> U[IntLinear up_proj]
  G --> SI[IntCommitWrap<br/>act_fn SiLU]
  SI --> MUL((×))
  U --> MUL
  MUL --> D[IntLinear down_proj]
  D --> R2((+))
  R1 --> R2
  R2 --> Y[hidden_states bf16]

  style N1 fill:#cfe8d2,stroke:#1f9b54
  style N2 fill:#cfe8d2,stroke:#1f9b54
  style QN fill:#cfe8d2,stroke:#1f9b54
  style KN fill:#cfe8d2,stroke:#1f9b54
  style SI fill:#cfe8d2,stroke:#1f9b54
  style Q fill:#d6e4f4,stroke:#2a5fa0
  style K fill:#d6e4f4,stroke:#2a5fa0
  style V fill:#d6e4f4,stroke:#2a5fa0
  style O fill:#d6e4f4,stroke:#2a5fa0
  style G fill:#d6e4f4,stroke:#2a5fa0
  style U fill:#d6e4f4,stroke:#2a5fa0
  style D fill:#d6e4f4,stroke:#2a5fa0
```

Green nodes = `IntCommitWrap` (int30 bf16 round-trip = identity, makes commitment visible).
Blue nodes = `IntLinear` (int operands + matmul kernel; details below).

## 3. `IntLinear` forward — branches by activation scheme

```mermaid
flowchart TB
  X[x bf16] --> AS{activation_scheme}

  AS -->|fp8_e4m3<br/>per-token| AQ1[fake_quantize_per_token_fp8_e4m3_ste<br/>x → bf16 on fp8 e4m3 grid 256 levels<br/>per-token absmax computed in bf16]
  AS -->|block_fp8_e4m3<br/>per-128-block| AQ2[fake_quantize_block_fp8_e4m3_ste<br/>x → bf16 on fp8 e4m3 grid<br/>per-128-block absmax]
  AS -->|uniform default| AQ3[per-token int24 STE]

  AQ1 --> M1{int_matmul_path?}
  M1 -->|yes| IM["int30 quant x_q & W per-token/per-row<br/>round-trip identity on bf16<br/>F.linear on dequant bf16"]
  M1 -->|no| FL1["F.linear(x_q, W)"]

  AQ2 --> M2{block_fp8_kernel_path?}
  M2 -->|yes| BK["w8a8_block_fp8_matmul<br/>stashed teacher fp8 weight + scale_inv<br/>activation via teacher's act_quant Triton kernel<br/>per-K-block fp32 accumulator"]
  M2 -->|no| FL2["F.linear(x_q, W) on bf16 dequant W<br/>this is the 0.953 path (kernel mismatch)"]

  AQ3 --> STE[uniform int24 STE x @ W.T]

  IM --> Y[y bf16]
  FL1 --> Y
  BK --> Y
  FL2 --> Y
  STE --> Y

  style IM fill:#cfe8d2,stroke:#1f9b54
  style BK fill:#cfe8d2,stroke:#1f9b54
  style FL2 fill:#f5d4d0,stroke:#c4514a
  style STE fill:#fbe6c4,stroke:#b07013
```

Green = bit-exact-teacher paths used in this experiment.
Red = block-fp8 fallback that gives top-1 ≈ 0.953 (kernel mismatch).
Yellow = legacy uniform-int24 path (gives top-1 ≈ 0.93, not used in the final result).

## 4. Per-position eval

```mermaid
flowchart LR
  P[held-out prompt] --> TF[teacher forward<br/>fp8 dynamic]
  P --> SF[student forward<br/>full int model]
  TF --> ZR[z_ref logits]
  SF --> ZC[z_cand logits]
  G[Gumbel0,1 noise<br/>same draw for both] --> M
  ZR --> M[post_gumbel_margin<br/>δ = max ZR+g − ZR+g at argmax ZC+g]
  ZC --> M
  ZR --> T1[top-1 match: argmax ZR == argmax ZC]
  ZC --> T1
  ZR --> KL[KL ZR || ZC]
  ZC --> KL
  M --> Agg[Aggregate over ~320k positions per model]
  T1 --> Agg
  KL --> Agg
  Agg --> Out[top-1 / KL / margin tables and figure]
```

## What "full int" means at each surface

| Surface | int commitment | Kernel between commits |
|---|---|---|
| Activation entering each Linear | per-token fp8 e4m3 (256-level grid → int8 LUT) for Qwen0.5B/Llama; per-128-block fp8 e4m3 for Qwen3 | — |
| Weight in each Linear | per-row int30 + fp32 scale (or per-128×128-tile fp8 + scale_inv for Qwen3) | — |
| RMSNorm input | per-token int30 + fp32 scale (round-trip identity on bf16) | torch RMSNorm (cast to fp32, pow².mean, rsqrt, multiply by gamma) |
| SiLU input | per-token int30 + fp32 scale | torch sigmoid + multiply |
| Attention Q/K/V | int30 from upstream `q_proj`/`k_proj`/`v_proj` IntLinear outputs | `F.linear(bf16)` plus RoPE rotation, softmax, `F.linear(bf16)` |
| Matmul output | bf16 (next layer's commit point) | — |

Every kernel between commits is a deterministic function of its committed inputs (in a ZK
circuit, expand to fp32-on-int31 arithmetic with bf16-keyed LUTs for `rsqrt`, `exp`, `sigmoid`,
`sin`, `cos`). The fp8 dynamic activation quant is itself a 256-entry public LUT.
