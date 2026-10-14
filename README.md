# Hybrid Quantum Regression with a Reuploading Circuit (HQR-RC)

Code for the article *"Hybrid quantum regression with data reuploading for tea aroma quality prediction using E-nose sensor data"* (Applied Computing and Informatics, under review).

HQR-RC predicts the panel aroma score of green tea from twelve E-nose channels (ten MOS sensors, humidity and temperature). It is evaluated under a leakage-free protocol, together with 20 classical, kernel and quantum-kernel baselines and parameter-matched controls in which the circuit is removed.

## Proposed method

HQR-RC is a hybrid quantum–classical regressor with four parts.

1. **Amplitude-encoded initial state.** The standardised 12-dimensional input is zero-padded to 16 entries and ℓ2-normalised. The result is loaded as the initial state of 4 qubits.
2. **Learned angle encoder.** A shared Conv1D encoder (16 and 32 filters, kernel size 3, batch normalisation, GELU, max-pooling, linear layer to a 32-dimensional code **h**) feeds twelve linear heads. Head *b* produces the data angles of block *b*: **θ**⁽ᵇ⁾ = W_b **h** + **c**_b ∈ ℝ¹². Each block therefore receives 3 angles per qubit, and every one of the 144 angles depends nonlinearly on all twelve sensors.
3. **Data-reuploading circuit.** The circuit has 12 blocks. Each block is U⁽ᵇ⁾ = E_b · V(**φ**⁽ᵇ⁾) · S(**θ**⁽ᵇ⁾(**x**)), with:
   - S: the data upload, R_X → R_Y → R_Z on every qubit;
   - V: six trainable rotations per qubit;
   - E_b: a ring entangler on the pairs (q, q+1 mod 4), using CNOT gates in odd blocks and CZ gates in even blocks.

   Re-uploading the data in every block enlarges the Fourier spectrum the model can represent, without adding qubits.
4. **Readout and head.**
   - The expectation values ⟨Z⟩, ⟨X⟩ and ⟨Y⟩ of every qubit (three measurement settings) are expanded into 40 readout features: z, x, y, zx, zy, xy, z², x², y² and zxy for each qubit.
   - A regression head (128 tanh units, dropout 0.2) maps these features to the aroma score.
   - All parts are trained jointly, giving k = 15,249 trainable parameters in total.

The circuits were designed in Qiskit 2.4.2 / Qiskit Aer 0.17.2. For batched GPU training, the same gate sequence runs in a PyTorch statevector simulator, which computes exact expectation values and gradients by reverse-mode automatic differentiation. Finite shots and depolarising noise are simulated at evaluation time.

### Variants and controls used in the paper

| Model | Description |
|---|---|
| HQR-BC | Data rotations in block 1 only (no reuploading). Everything else is identical to HQR-RC_MLP. |
| HQR-RC_MLP | Reuploading in every block, with a separate MLP angle generator per block. |
| **HQR-RC (ours)** | Reuploading in every block, with the shared Conv1D angle encoder. |
| C-wide / C-angles / C-trig | Parameter-matched controls without the circuit. C-trig feeds the Fourier basis [cos θ, sin θ] of the same angles to the head. |

## Repository structure

```
1_HQR-Preprocessing-Analysis.ipynb     data hierarchy, PCA, correlations, leakage diagnostic
2_HQR-RC-Main-Experiment-V7.ipynb      main experiment: HQR variants, 20 baselines, ablations,
                                       refit/ensemble, lot-grouped test, shot/depolarising and sensor noise
3_HQR-RC-Controls-Timing-V7b.ipynb     parameter-matched controls, search budget, CPU/GPU timing
4_HQR-RC-Timing-SameDevice-V7c.ipynb   fit/inference timing of PLS, KRR, GPR and quantum kernels
models/                                model definitions loaded by notebooks 2-4
                                       (HQR circuit and Qiskit equivalent, classical MLP/CNN/linear models)
config/folds_session.json              session-grouped fold assignment (5 folds x seeds 42, 7, 123)
config/folds_lot.json                  lot-grouped fold assignment used for the stress test
paper/make_tables.py                   regenerates the table bodies of the paper from the run folders
paper/make_figs.py                     regenerates the figure panels of the paper from the run folders
datasets.csv                           E-nose dataset used in the paper
requirements.txt
```

`datasets.csv` contains the E-nose dataset (see Data). Results are not part of the repository. Every notebook writes its outputs to `runs/<experiment>/run_<date-time>/` and records the latest run in `LATEST.txt`.

## Data

The E-nose dataset collected by the authors is included in this repository as `datasets.csv` (10,409 rows, 274 measurement sessions, 69 production lots). The notebooks read it from the repository root.

| Column(s) | Meaning |
|---|---|
| `MQ3`, `TGS822`, `TGS2602`, `MQ5`, `MQ138`, `TGS2620`, `TGS813`, `TGS2600`, `TGS2611`, `TGS2603` | MOS sensor channels (model inputs) |
| `Humidity`, `Celsius` | Environment channels (model inputs) |
| `Chop_ID` | Production lot |
| `Sampling_ID` | Measurement session (one 60-s measurement cycle) |
| `Aroma` | Panel aroma score (target) |
| `Taste`, `Color`, `Appearance`, `Dreg` | Other sensory scores of the panel (not used in the paper) |
| `Kategori` | Quality category A–E (not used in the paper) |
| `Standar Kualitas` | Lot grade: `Baik` (Good) or `Cacat Mutu` (Defect); descriptive only |
| `*_mean`, `*_min`, `*_max`, `*_std`, `*_skew`, `*_kurtosis`, `*_auc` | Session-level summary statistics (not used as model inputs) |

If you use the dataset or the code, please cite the article.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # install a CUDA build of torch for GPU use
```

## Reproducing the results

Run the notebooks in order from the repository root.

| Step | Notebook | Main outputs → paper |
|---|---|---|
| 1 | `1_HQR-Preprocessing-Analysis.ipynb` | `runs/preproc/` → Table I, Figures II–III, Table IV and Figure V (leakage) |
| 2 | `2_HQR-RC-Main-Experiment-V7.ipynb` | `runs/hqrrc_v7/` → Tables III, V, VI, VIII–XI and Figures VI–IX |
| 3 | `3_HQR-RC-Controls-Timing-V7b.ipynb` | `runs/hqrrc_v7b/` → Tables II, VII and XII, Figure X |
| 4 | `4_HQR-RC-Timing-SameDevice-V7c.ipynb` | `runs/hqrrc_v7c/` → timing rows of Table XII and Figure X |

Then regenerate the tables and figures:

```bash
python paper/make_tables.py runs paper_outputs/tables
python paper/make_figs.py runs datasets.csv paper_outputs/figures
```

Figures I (workflow) and IV (circuits) were drawn manually and are not generated by the scripts.

### Useful environment variables

| Variable | Effect |
|---|---|
| `HQR_DEVICE=cuda` | Train on the GPU (default `auto`; the paper used CPU and GPU) |
| `HQR_SMOKE=1` | Quick functional check: 2 epochs and freshly generated folds; the results are not meaningful |
| `HQR_RUN_ID=auto\|new\|<name>` | Resume the last unfinished run with the same configuration (default), start a new run, or use a named run folder |
| `HQR_V7_RUN=<run_folder>` | Notebooks 3–4: use a specific V7 run instead of `runs/hqrrc_v7/LATEST.txt` |

### Notes

- **Evaluation protocol.**
  - Session-grouped GroupKFold, 5 folds × 3 seeds (15 outer folds), with 20% of the training sessions held out for early stopping and selection.
  - Each model is also refitted on 100% of the training sessions, and a three-seed ensemble is reported.
  - Paired models are compared with two-sided Wilcoxon signed-rank tests and Holm correction.
- **Trainer (all neural models).**
  - Adam, learning rate 10⁻³, weight decay 10⁻⁴ (none on circuit angles), batch size 128.
  - At most 60 epochs: 10 warm-up epochs, then cosine decay.
  - Early stopping with patience 20, gradient clipping 1.0, weight EMA 0.99.
- **Runtime.** The full run of notebook 2 trains several hundred folds. One HQR-RC epoch takes about 14 s on a 20-thread CPU, so plan for a long run or use `HQR_SMOKE=1` first. Unfinished runs resume automatically.
- **Exact numbers.** GPU non-determinism can change the last digits of individual folds; the fold assignment in `config/` is the one used in the paper.
- **Search budget.** Table II of the paper also counts earlier development stages (V3–V6), which are not part of this repository. A fresh run therefore reports only the budget of notebooks 2–3.

## License

MIT (see `LICENSE`).
