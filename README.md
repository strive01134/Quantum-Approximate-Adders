# Quantum-Approximate-Adders


Python/Qiskit implementations for the design and simulation of approximate quantum adders.

This repository includes:

* Five Approximate Quantum Adder (AQA) architectures
* A hybrid AQA–Cuccaro Ripple-Carry Adder
* Ideal and noisy quantum circuit simulation
* Arithmetic error evaluation using ER, MED, NMED, and MaxED

The project was developed to investigate the trade-off between quantum circuit simplification and arithmetic accuracy.

---

## Repository Structure

```text
Quantum-Approximate-Adders/
├── README.md
├── aqa_variants.py
└── hybrid_aqa_cuccaro.py
```

### `aqa_variants.py`

Implements five approximate quantum adder architectures:

| Design | Approximate Sum | Carry-out                  |
| ------ | --------------- | -------------------------- |
| AQA1   | A               | 0                          |
| AQA2   | A XOR B         | 0                          |
| AQA3   | A               | MSB of B                   |
| AQA4   | A XOR B         | MSB of B                   |
| AQA5   | A XOR B         | AND of the MSBs of A and B |

Each architecture is constructed using Qiskit quantum circuits and evaluated over combinations of input operands.

---

### `hybrid_aqa_cuccaro.py`

Implements a hybrid quantum adder consisting of two regions:

```text
LSB Region                     MSB Region
┌─────────────────┐          ┌────────────────────┐
│ Approximate AQA │  ─────▶  │ Cuccaro Adder     │
│ AQA1 ~ AQA5     │          │ MAJ / UMA blocks  │
└─────────────────┘          └────────────────────┘
```

The least significant bits are processed using one of the approximate AQA architectures, while the upper bits are processed using a Cuccaro ripple-carry adder.

The number of approximate bits and the AQA architecture can be selected at runtime.

---

## Simulation Flow

```text
Input operands
      │
      ▼
Quantum circuit generation
      │
      ▼
Approximate / Hybrid adder
      │
      ▼
Qiskit transpilation
      │
      ▼
AerSimulator
      │
      ├── Ideal
      ├── Thermal relaxation
      ├── Depolarizing
      └── Bit-flip
      │
      ▼
Measurement result
      │
      ▼
Error analysis
```

---

## Noise Models

The simulations support four execution environments.

### Ideal

No quantum noise is applied.

### Thermal Relaxation

A thermal relaxation error model is constructed using T1, T2, and gate-time parameters.

### Depolarizing Noise

Depolarizing errors are applied to single-qubit and two-qubit operations.

### Bit-Flip Noise

Pauli-X errors are applied probabilistically to the quantum gates.

---

## Evaluation Metrics

The simulated arithmetic result is compared with the exact classical sum.

### Error Distance

```text
ED = |Approximate Result - Exact Result|
```

### Error Rate (ER)

Ratio of input cases that generate an incorrect arithmetic result.

```text
ER = Number of erroneous cases / Total number of cases
```

### Mean Error Distance (MED)

Average arithmetic error distance over the evaluated input cases.

```text
MED = Average(ED)
```

### Normalized Mean Error Distance (NMED)

Normalized form of MED.

### Maximum Error Distance (MaxED)

Largest observed arithmetic error.

```text
MaxED = max(ED)
```

---

## Requirements

* Python 3
* Qiskit
* Qiskit Aer

Install the required packages using:

```bash
pip install qiskit qiskit-aer
```

---

## Usage

### 1. AQA Architecture Evaluation

Run:

```bash
python aqa_variants.py
```

The program requests:

```text
Bit-width
Number of shots
Noise model
```

It then evaluates AQA1 through AQA5 and reports:

```text
Error Rate
MED
NMED
MaxED
```

Example:

```text
===== Approximate Quantum Adder Evaluation =====

-- AQA1 --
Error Rate : ...
MED        : ...
NMED       : ...
Max ED     : ...

-- AQA2 --
...
```

---

### 2. Hybrid AQA–Cuccaro Evaluation

Run:

```bash
python hybrid_aqa_cuccaro.py
```

The program allows configuration of:

```text
Total bit-width
Approximate bit-width
AQA type
Number of shots
Noise model
```

Example configuration:

```text
Total bits : 8
AQA bits   : 2
AQA type   : AQA2
Upper bits : Cuccaro ripple-carry adder
Noise      : Depolarizing
```

The circuit is evaluated over the input operand space and arithmetic error metrics are reported.

---

## Technologies

| Category               | Technology                      |
| ---------------------- | ------------------------------- |
| Programming            | Python                          |
| Quantum SDK            | Qiskit                          |
| Simulation             | Qiskit Aer                      |
| Quantum Arithmetic     | Approximate Quantum Adders      |
| Exact Arithmetic Block | Cuccaro Ripple-Carry Adder      |
| Noise Simulation       | Thermal, Depolarizing, Bit-Flip |
| Evaluation             | ER, MED, NMED, MaxED            |

---

## Project Scope

This repository demonstrates:

* Quantum circuit construction using Qiskit
* Implementation of quantum arithmetic circuits
* Approximate computing concepts
* Cuccaro ripple-carry adder implementation
* Quantum noise-model simulation
* Exhaustive arithmetic evaluation
* Quantitative error analysis

---

## Implementation Note

The current `aqa_variants.py` evaluation routine contains an 8-bit-specific result parsing and normalization section. Therefore, the current version should be treated as an 8-bit evaluation implementation unless those sections are generalized for other bit-widths.

The repository contains research and educational implementations intended for quantum arithmetic simulation and design-space evaluation.
