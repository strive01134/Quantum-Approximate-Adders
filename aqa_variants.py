from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, pauli_error

# ============================================================
# Noise Model Generator
# ============================================================
def make_noise_model(choice):
    """Create one of the supported noise models:
       0: Ideal, 1: Thermal, 2: Depolarizing, 3: Bit-flip."""
    nm = NoiseModel()
    if choice == 1:
        print("Thermal Relaxation noise enabled")
        T1, T2 = 50e3, 70e3
        g1, g2 = 100, 300
        e1 = thermal_relaxation_error(T1, T2, g1)
        e2 = thermal_relaxation_error(T1, T2, g2)
        nm.add_all_qubit_quantum_error(e1, ['x', 'sx', 'id'])
        nm.add_all_qubit_quantum_error(e2.tensor(e2), ['cx'])
    elif choice == 2:
        print("Depolarizing noise enabled (p1=0.001, p2=0.01)")
        e1 = depolarizing_error(0.001, 1)
        e2 = depolarizing_error(0.01, 2)
        nm.add_all_qubit_quantum_error(e1, ['x', 'sx', 'id'])
        nm.add_all_qubit_quantum_error(e2, ['cx'])
    elif choice == 3:
        print("Bit-flip noise enabled (p=0.005)")
        e1 = pauli_error([("X", 0.005), ("I", 0.995)])
        e2 = e1.tensor(e1)
        nm.add_all_qubit_quantum_error(e1, ['x', 'sx', 'id'])
        nm.add_all_qubit_quantum_error(e2, ['cx'])
    else:
        print("Ideal environment (no noise)")
        nm = None
    return nm


# ============================================================
# Cuccaro Adder Primitives
# ============================================================
def maj(qc, a, b, c):
    """MAJ block used in Cuccaro ripple-carry adder."""
    qc.cx(a, b)
    qc.cx(a, c)
    qc.ccx(c, b, a)


def uma(qc, a, b, c):
    """UMA block used in Cuccaro ripple-carry adder."""
    qc.ccx(c, b, a)
    qc.cx(a, c)
    qc.cx(c, b)


# ============================================================
# Cuccaro Block (used for the MSB region)
# ============================================================
def cuccaro_block(qc, A, B, Cin, Cout, start, end):
    """Perform Cuccaro addition for bits [start:end] inclusive."""
    maj(qc, A[start], B[start], Cin)
    for i in range(start + 1, end + 1):
        maj(qc, A[i], B[i], A[i - 1])
    qc.cx(A[end], Cout)
    for i in range(end, start, -1):
        uma(qc, A[i], B[i], A[i - 1])
    uma(qc, A[start], B[start], Cin)


# ============================================================
# Approximate Quantum Adder (AQA) Block
# ============================================================
def build_aqa_block(qc, A, B, Cout, n_aqa, aqa_type):
    """Construct an AQA block for the least significant bits."""
    if aqa_type == 1:
        # AQA1: Sum = A
        for i in range(n_aqa):
            qc.cx(A[i], B[i])
            qc.cx(B[i], A[i])
            qc.cx(A[i], B[i])
    elif aqa_type == 2:
        # AQA2: Sum = A ⊕ B
        for i in range(n_aqa):
            qc.cx(A[i], B[i])
    elif aqa_type == 3:
        # AQA3: Sum = A, Cout = b_{n_aqa-1}
        qc.cx(B[n_aqa - 1], Cout)
        for i in range(n_aqa):
            qc.cx(A[i], B[i])
            qc.cx(B[i], A[i])
            qc.cx(A[i], B[i])
    elif aqa_type == 4:
        # AQA4: Sum = A ⊕ B, Cout = b_{n_aqa-1}
        qc.cx(B[n_aqa - 1], Cout)
        for i in range(n_aqa):
            qc.cx(A[i], B[i])
    elif aqa_type == 5:
        # AQA5: Sum = A ⊕ B, Cout = a_{n_aqa-1} · b_{n_aqa-1}
        qc.ccx(A[n_aqa - 1], B[n_aqa - 1], Cout)
        for i in range(n_aqa):
            qc.cx(A[i], B[i])
    else:
        raise ValueError("Invalid AQA type (1~5)")


# ============================================================
# Hybrid AQA–Cuccaro Adder
# ============================================================
def build_hybrid(a_val, b_val, n_total, n_aqa, aqa_type):
    """Combine AQA for LSB region and Cuccaro for MSB region."""
    A = QuantumRegister(n_total, 'A')
    B = QuantumRegister(n_total, 'B')
    Cin = QuantumRegister(1, 'Cin')
    Cout = QuantumRegister(1, 'Cout')
    creg = ClassicalRegister(n_total + 1, 'c')
    qc = QuantumCircuit(A, B, Cin, Cout, creg)
    qc.reset(Cin[0])
    qc.reset(Cout[0])

    # Initialize input values
    for i in range(n_total):
        if (a_val >> i) & 1:
            qc.x(A[i])
        if (b_val >> i) & 1:
            qc.x(B[i])

    # Lower bits → approximate (AQA)
    build_aqa_block(qc, A, B, Cin[0], n_aqa, aqa_type)

    # Carry connection between AQA and Cuccaro region
    c_mid = Cout[0]

    # Upper bits → exact (Cuccaro)
    if n_aqa < n_total:
        cuccaro_block(qc, A, B, c_mid, Cout[0], n_aqa, n_total - 1)

    # Measurement
    for i in range(n_total):
        qc.measure(B[i], creg[i])
    qc.measure(Cout[0], creg[n_total])
    return qc


# ============================================================
# Evaluation
# ============================================================
def parse_counts(counts, n):
    """Extract numerical output from measurement counts."""
    key = max(counts, key=counts.get)
    bits = [int(ch) for ch in key[::-1]]
    s = sum((bits[i] << i) for i in range(n))
    c = bits[n]
    return s + (c << n)


def evaluate(sim, n_total, n_aqa, aqa_type, shots=1):
    """Run full evaluation over all input pairs."""
    pairs = [(a, b) for a in range(1 << n_total) for b in range(1 << n_total)]
    truths = [a + b for a, b in pairs]

    total = 0
    err = 0
    MED = 0.0
    maxED = 0
    denom = (1 << n_total) - 1

    circs = [build_hybrid(a, b, n_total, n_aqa, aqa_type) for a, b in pairs]
    tqc = transpile(circs, backend=sim, optimization_level=0)
    res = sim.run(tqc, shots=shots).result()
    outs = res.get_counts()

    if isinstance(outs, dict):
        outs = [outs]

    for cnt, truth in zip(outs, truths):
        approx = parse_counts(cnt, n_total)
        ED = abs(approx - truth)
        total += 1
        if ED:
            err += 1
            if ED > maxED:
                maxED = ED
        MED += (ED - MED) / total

    ER = err / total
    NMED = MED / denom
    return {"ER": ER, "MED": MED, "NMED": NMED, "MaxED": maxED, "Total": total}


# ============================================================
# Main Routine
# ============================================================
if __name__ == "__main__":
    n_total = int(input("Enter total bit-width (e.g., 8): "))
    n_aqa = int(input(f"Enter AQA bit-width (0~{n_total - 1}): "))
    while n_aqa >= n_total:
        print("AQA bit-width must be smaller than total width.")
        n_aqa = int(input(f"Enter AQA bit-width (0~{n_total - 1}): "))

    aqa_type = int(input("Select AQA type (1~5): "))
    shots = int(input("Enter number of shots (e.g., 1, 10, 100): "))
    print("Noise mode: [0] Ideal  [1] Thermal  [2] Depolarizing  [3] Bit-flip")
    noise_choice = int(input("Select: "))

    nm = make_noise_model(noise_choice)
    sim = AerSimulator(noise_model=nm) if nm else AerSimulator()

    print("\n===== Hybrid AQA–Cuccaro Evaluation =====")
    print(f"Total bits: {n_total} | AQA: {n_aqa}-bit (AQA{aqa_type}) | "
          f"Cuccaro: {n_total - n_aqa}-bit | Noise: {noise_choice}")

    stats = evaluate(sim, n_total, n_aqa, aqa_type, shots)
    print(f"\nError Rate : {stats['ER']:.6f}")
    print(f"MED        : {stats['MED']:.6f}")
    print(f"NMED       : {stats['NMED']:.6f}")
    print(f"Max ED     : {stats['MaxED']}")
    print(f"Total      : {stats['Total']}")
