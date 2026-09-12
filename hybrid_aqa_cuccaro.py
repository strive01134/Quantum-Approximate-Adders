#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python-Based Carry-Dependency-Aware Relative-Phase Toffoli
Placement and Noise Evaluation Framework for QRCA

Version: 1.0

Main functions
--------------
1. Construct a Cuccaro quantum ripple-carry adder.
2. Enumerate forward and backward Toffoli locations.
3. Calculate the carry-dependency metric P(l_i).
4. Automatically select the forward Toffoli location with minimum P(l_i).
5. Replace only the selected exact Toffoli gate with RCCX.
6. Verify ideal computational-basis arithmetic correctness.
7. Compare conventional and proposed QRCA designs under:
   - Thermal relaxation noise
   - Depolarizing noise
   - Bit-flip noise
   - Lindblad-like composite noise
8. Calculate MED, NMED, ER, execution error probability,
   modal MaxED, and observed MaxED.

Excluded functions
------------------
This software does not calculate:
- quantum cost
- gate count
- circuit depth
- resource reduction

Execution
---------
    python qrca_carry_dependency_noise_framework.py
"""

from __future__ import annotations

import gc
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    transpile,
)
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    phase_amplitude_damping_error,
    thermal_relaxation_error,
)


PROGRAM_TITLE = (
    "Python-Based Carry-Dependency-Aware Relative-Phase Toffoli "
    "Placement and Noise Evaluation Framework for QRCA"
)

PROGRAM_VERSION = "1.0"

EXECUTION_BASIS = [
    "rz",
    "sx",
    "x",
    "cx",
]


@dataclass(frozen=True)
class ToffoliSite:
    site_id: int
    stage: str
    role: str
    logical_bit: int
    control0: str
    control1: str
    target: str
    analyzed: bool


@dataclass(frozen=True)
class DependencyEdge:
    source_site_id: int
    dependent_site_id: int
    relation: str


@dataclass(frozen=True)
class CircuitDesign:
    label: str
    selected_site_ids: Tuple[int, ...]
    design_type: str


@dataclass
class NoiseParameters:
    T1_us: float = 120.0
    T2_us: float = 90.0
    one_q_gate_time_ns: float = 35.0
    two_q_gate_time_ns: float = 250.0

    depolarizing_1q: float = 0.001
    depolarizing_2q: float = 0.010

    bitflip_probability: float = 0.001

    lindblad_amp: float = 0.0045
    lindblad_phase: float = 0.0090
    lindblad_correlated_2q: float = 0.0005


@dataclass
class ExperimentConfig:
    mode: str
    n: int

    preset: str
    shots: int
    repeats: int

    max_cases: int
    case_mode: str
    batch_size: int
    seed: int

    site_policy: str
    manual_sites: List[int]

    noise_models: List[str]
    progress_every: int

    output_dir: str
    noise_parameters: NoiseParameters


PRESETS = {
    "demo": {
        "shots": 10,
        "repeats": 1,
        "n4_cases": 64,
        "n6_cases": 256,
        "large_cases": 512,
    },
    "paper": {
        "shots": 100,
        "repeats": 1,
        "n4_cases": 256,
        "n6_cases": 4096,
        "large_cases": 4096,
    },
    "high": {
        "shots": 300,
        "repeats": 3,
        "n4_cases": 256,
        "n6_cases": 4096,
        "large_cases": 4096,
    },
}


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def normalize_counts(counts):
    if isinstance(counts, dict):
        return [counts]

    return counts


def counts_to_distribution(
    counts: Dict[str, int],
    shots: int,
) -> Dict[int, float]:
    return {
        int(bitstring.replace(" ", ""), 2): count / shots
        for bitstring, count in counts.items()
    }


def safe_percentage(
    value: float,
    baseline: float,
) -> float:
    if abs(baseline) < 1e-15:
        return float("nan")

    return (value - baseline) / baseline * 100.0


def validate_probability(
    name: str,
    value: float,
) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )


def validate_noise_parameters(
    parameters: NoiseParameters,
) -> None:
    if parameters.T1_us <= 0:
        raise ValueError("T1 must be positive.")

    if parameters.T2_us <= 0:
        raise ValueError("T2 must be positive.")

    if parameters.T2_us > 2.0 * parameters.T1_us:
        raise ValueError(
            "Thermal relaxation requires T2 <= 2*T1."
        )

    if parameters.one_q_gate_time_ns <= 0:
        raise ValueError(
            "1Q gate time must be positive."
        )

    if parameters.two_q_gate_time_ns <= 0:
        raise ValueError(
            "2Q gate time must be positive."
        )

    validate_probability(
        "depolarizing_1q",
        parameters.depolarizing_1q,
    )
    validate_probability(
        "depolarizing_2q",
        parameters.depolarizing_2q,
    )
    validate_probability(
        "bitflip_probability",
        parameters.bitflip_probability,
    )
    validate_probability(
        "lindblad_amp",
        parameters.lindblad_amp,
    )
    validate_probability(
        "lindblad_phase",
        parameters.lindblad_phase,
    )
    validate_probability(
        "lindblad_correlated_2q",
        parameters.lindblad_correlated_2q,
    )

    if (
        parameters.lindblad_amp
        + parameters.lindblad_phase
        > 1.0
    ):
        raise ValueError(
            "lindblad_amp + lindblad_phase must not exceed 1."
        )


def build_noise_model(
    noise_name: str,
    parameters: NoiseParameters,
) -> NoiseModel:
    validate_noise_parameters(parameters)

    noise_model = NoiseModel()

    if noise_name == "thermal":
        T1_ns = parameters.T1_us * 1_000.0
        T2_ns = parameters.T2_us * 1_000.0

        one_q_error = thermal_relaxation_error(
            T1_ns,
            T2_ns,
            parameters.one_q_gate_time_ns,
        )

        two_q_single = thermal_relaxation_error(
            T1_ns,
            T2_ns,
            parameters.two_q_gate_time_ns,
        )

        two_q_error = two_q_single.tensor(
            two_q_single
        )

    elif noise_name == "depolarizing":
        one_q_error = depolarizing_error(
            parameters.depolarizing_1q,
            1,
        )

        two_q_error = depolarizing_error(
            parameters.depolarizing_2q,
            2,
        )

    elif noise_name == "bitflip":
        probability = parameters.bitflip_probability

        one_q_error = pauli_error(
            [
                ("X", probability),
                ("I", 1.0 - probability),
            ]
        )

        two_q_error = pauli_error(
            [
                ("IX", probability / 2.0),
                ("XI", probability / 2.0),
                ("II", 1.0 - probability),
            ]
        )

    elif noise_name == "lindblad":
        one_q_error = phase_amplitude_damping_error(
            parameters.lindblad_amp,
            parameters.lindblad_phase,
        )

        damping_single = phase_amplitude_damping_error(
            parameters.lindblad_amp,
            parameters.lindblad_phase,
        )

        independent_2q = damping_single.tensor(
            damping_single
        )

        correlated_probability = (
            parameters.lindblad_correlated_2q
        )

        correlated_error = pauli_error(
            [
                ("ZZ", correlated_probability),
                ("II", 1.0 - correlated_probability),
            ]
        )

        two_q_error = independent_2q.compose(
            correlated_error
        )

    else:
        raise ValueError(
            f"Unsupported noise model: {noise_name}"
        )

    for gate in [
        "rz",
        "sx",
        "x",
    ]:
        noise_model.add_all_qubit_quantum_error(
            one_q_error,
            [gate],
        )

    noise_model.add_all_qubit_quantum_error(
        two_q_error,
        ["cx"],
    )

    return noise_model


def enumerate_qrca_sites(
    n: int,
) -> List[ToffoliSite]:
    if n < 4:
        raise ValueError("QRCA requires n >= 4.")

    sites: List[ToffoliSite] = []

    sites.append(
        ToffoliSite(
            site_id=0,
            stage="forward",
            role="initial_carry_generate",
            logical_bit=0,
            control0="a[0]",
            control1="b[0]",
            target="x",
            analyzed=True,
        )
    )

    sites.append(
        ToffoliSite(
            site_id=1,
            stage="forward",
            role="forward_head",
            logical_bit=1,
            control0="x",
            control1="b[1]",
            target="a[1]",
            analyzed=True,
        )
    )

    site_id = 2

    for logical_bit in range(2, n - 2):
        sites.append(
            ToffoliSite(
                site_id=site_id,
                stage="forward",
                role="forward_propagate",
                logical_bit=logical_bit,
                control0=f"a[{logical_bit - 1}]",
                control1=f"b[{logical_bit}]",
                target=f"a[{logical_bit}]",
                analyzed=True,
            )
        )

        site_id += 1

    sites.append(
        ToffoliSite(
            site_id=site_id,
            stage="forward",
            role="final_carry_prepare",
            logical_bit=n - 2,
            control0=f"a[{n - 3}]",
            control1=f"b[{n - 2}]",
            target=f"a[{n - 2}]",
            analyzed=True,
        )
    )

    site_id += 1

    sites.append(
        ToffoliSite(
            site_id=site_id,
            stage="forward",
            role="final_carry_out",
            logical_bit=n - 1,
            control0=f"a[{n - 2}]",
            control1=f"b[{n - 1}]",
            target="z",
            analyzed=True,
        )
    )

    site_id += 1

    sites.append(
        ToffoliSite(
            site_id=site_id,
            stage="backward",
            role="final_carry_cleanup",
            logical_bit=n - 2,
            control0=f"a[{n - 3}]",
            control1=f"b[{n - 2}]",
            target=f"a[{n - 2}]",
            analyzed=False,
        )
    )

    site_id += 1

    for logical_bit in reversed(
        range(2, n - 2)
    ):
        sites.append(
            ToffoliSite(
                site_id=site_id,
                stage="backward",
                role="backward_recover",
                logical_bit=logical_bit,
                control0=f"a[{logical_bit - 1}]",
                control1=f"b[{logical_bit}]",
                target=f"a[{logical_bit}]",
                analyzed=False,
            )
        )

        site_id += 1

    sites.append(
        ToffoliSite(
            site_id=site_id,
            stage="backward",
            role="backward_head",
            logical_bit=1,
            control0="x",
            control1="b[1]",
            target="a[1]",
            analyzed=False,
        )
    )

    site_id += 1

    sites.append(
        ToffoliSite(
            site_id=site_id,
            stage="backward",
            role="initial_carry_cleanup",
            logical_bit=0,
            control0="a[0]",
            control1="b[0]",
            target="x",
            analyzed=False,
        )
    )

    expected_count = 2 * n - 1

    if len(sites) != expected_count:
        raise AssertionError(
            f"QRCA site count mismatch: "
            f"expected={expected_count}, actual={len(sites)}"
        )

    return sites


def analyze_qrca_dependencies(
    n: int,
    sites: Sequence[ToffoliSite],
) -> Tuple[
    List[DependencyEdge],
    Dict[int, int],
]:
    analyzed_sites = [
        site
        for site in sites
        if site.analyzed
    ]

    forward_sites = {
        site.logical_bit: site
        for site in sites
        if site.stage == "forward"
    }

    backward_sites = [
        site
        for site in sites
        if site.stage == "backward"
    ]

    edges: List[DependencyEdge] = []

    for source in analyzed_sites:
        logical_bit = source.logical_bit

        for later_bit in range(
            logical_bit + 1,
            n,
        ):
            dependent = forward_sites[later_bit]

            edges.append(
                DependencyEdge(
                    source_site_id=source.site_id,
                    dependent_site_id=dependent.site_id,
                    relation="forward_carry_dependency",
                )
            )

        for dependent in backward_sites:
            if dependent.logical_bit >= logical_bit:
                edges.append(
                    DependencyEdge(
                        source_site_id=source.site_id,
                        dependent_site_id=dependent.site_id,
                        relation="backward_restore_dependency",
                    )
                )

    p_values = {
        source.site_id: sum(
            edge.source_site_id == source.site_id
            for edge in edges
        )
        for source in analyzed_sites
    }

    if n == 6:
        calculated = [
            p_values[site.site_id]
            for site in analyzed_sites
        ]

        expected = [
            10,
            8,
            6,
            4,
            2,
            0,
        ]

        if calculated != expected:
            raise AssertionError(
                f"QRCA P(l_i) check failed: "
                f"calculated={calculated}, expected={expected}"
            )

    return edges, p_values


def select_minimum_site(
    sites: Sequence[ToffoliSite],
    p_values: Dict[int, int],
) -> ToffoliSite:
    analyzed_sites = [
        site
        for site in sites
        if site.analyzed
    ]

    return min(
        analyzed_sites,
        key=lambda site: (
            p_values[site.site_id],
            -site.logical_bit,
            site.site_id,
        ),
    )


def create_designs(
    sites: Sequence[ToffoliSite],
    p_values: Dict[int, int],
    site_policy: str,
    manual_sites: Sequence[int],
) -> List[CircuitDesign]:
    designs = [
        CircuitDesign(
            label="conventional_ccx",
            selected_site_ids=tuple(),
            design_type="conventional",
        )
    ]

    analyzed_ids = {
        site.site_id
        for site in sites
        if site.analyzed
    }

    if site_policy == "min_p":
        selected = select_minimum_site(
            sites,
            p_values,
        )

        designs.append(
            CircuitDesign(
                label=f"proposed_minP_l{selected.site_id}",
                selected_site_ids=(selected.site_id,),
                design_type="proposed",
            )
        )

    elif site_policy == "manual":
        selected_ids = tuple(
            sorted(set(manual_sites))
        )

        invalid = [
            site_id
            for site_id in selected_ids
            if site_id not in analyzed_ids
        ]

        if invalid:
            raise ValueError(
                f"Invalid QRCA site IDs: {invalid}; "
                f"available={sorted(analyzed_ids)}"
            )

        if not selected_ids:
            raise ValueError(
                "At least one manual site is required."
            )

        label_text = "_".join(
            str(value)
            for value in selected_ids
        )

        designs.append(
            CircuitDesign(
                label=f"manual_rccx_l{label_text}",
                selected_site_ids=selected_ids,
                design_type="manual",
            )
        )

    elif site_policy == "all_sites":
        analyzed_sites = sorted(
            [
                site
                for site in sites
                if site.analyzed
            ],
            key=lambda site: (
                p_values[site.site_id],
                -site.logical_bit,
                site.site_id,
            ),
        )

        for site in analyzed_sites:
            designs.append(
                CircuitDesign(
                    label=(
                        f"rccx_l{site.site_id}_"
                        f"P{p_values[site.site_id]}"
                    ),
                    selected_site_ids=(site.site_id,),
                    design_type="site_sweep",
                )
            )

    else:
        raise ValueError(
            f"Unsupported site policy: {site_policy}"
        )

    return designs


def apply_toffoli(
    qc: QuantumCircuit,
    control0,
    control1,
    target,
    site_id: int,
    selected_site_ids: Set[int],
) -> None:
    if site_id in selected_site_ids:
        qc.rccx(
            control0,
            control1,
            target,
        )
    else:
        qc.ccx(
            control0,
            control1,
            target,
        )


def build_qrca(
    n: int,
    operand_a: int,
    operand_b: int,
    selected_site_ids: Iterable[int],
    measurement: str,
) -> QuantumCircuit:
    selected = set(selected_site_ids)

    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    x = QuantumRegister(1, "x")
    z = QuantumRegister(1, "z")

    qc = QuantumCircuit(
        a,
        b,
        x,
        z,
    )

    for index in range(n):
        if (operand_a >> index) & 1:
            qc.x(a[index])

        if (operand_b >> index) & 1:
            qc.x(b[index])

    def T(
        site_id: int,
        control0,
        control1,
        target,
    ) -> None:
        apply_toffoli(
            qc,
            control0,
            control1,
            target,
            site_id,
            selected,
        )

    for index in range(1, n):
        qc.cx(
            a[index],
            b[index],
        )

    qc.cx(a[1], x[0])

    T(
        0,
        a[0],
        b[0],
        x[0],
    )

    qc.cx(a[2], a[1])

    T(
        1,
        x[0],
        b[1],
        a[1],
    )

    qc.cx(a[3], a[2])

    site_id = 2

    for index in range(2, n - 2):
        T(
            site_id,
            a[index - 1],
            b[index],
            a[index],
        )

        site_id += 1

        qc.cx(
            a[index + 2],
            a[index + 1],
        )

    T(
        site_id,
        a[n - 3],
        b[n - 2],
        a[n - 2],
    )

    site_id += 1

    qc.cx(
        a[n - 1],
        z[0],
    )

    T(
        site_id,
        a[n - 2],
        b[n - 1],
        z[0],
    )

    site_id += 1

    for index in range(1, n - 1):
        qc.x(b[index])

    qc.cx(
        x[0],
        b[1],
    )

    for index in range(2, n):
        qc.cx(
            a[index - 1],
            b[index],
        )

    T(
        site_id,
        a[n - 3],
        b[n - 2],
        a[n - 2],
    )

    site_id += 1

    for index in reversed(
        range(2, n - 2)
    ):
        T(
            site_id,
            a[index - 1],
            b[index],
            a[index],
        )

        site_id += 1

        qc.cx(
            a[index + 2],
            a[index + 1],
        )

        qc.x(
            b[index + 1]
        )

    T(
        site_id,
        x[0],
        b[1],
        a[1],
    )

    site_id += 1

    qc.cx(a[3], a[2])
    qc.x(b[2])

    T(
        site_id,
        a[0],
        b[0],
        x[0],
    )

    site_id += 1

    qc.cx(a[2], a[1])
    qc.x(b[1])

    qc.cx(a[1], x[0])

    for index in range(n):
        qc.cx(
            a[index],
            b[index],
        )

    if site_id != 2 * n - 1:
        raise AssertionError(
            "QRCA site-ID assignment mismatch."
        )

    if measurement == "none":
        return qc

    if measurement == "sum":
        classical = ClassicalRegister(
            n + 1,
            "sum",
        )

        qc.add_register(classical)

        for index in range(n):
            qc.measure(
                b[index],
                classical[index],
            )

        qc.measure(
            z[0],
            classical[n],
        )

        return qc

    if measurement == "all":
        classical = ClassicalRegister(
            qc.num_qubits,
            "all",
        )

        qc.add_register(classical)

        for index, qubit in enumerate(qc.qubits):
            qc.measure(
                qubit,
                classical[index],
            )

        return qc

    raise ValueError(
        f"Unsupported measurement: {measurement}"
    )


def automatic_case_settings(
    n: int,
    preset: str,
) -> Tuple[int, str, int]:
    values = PRESETS[preset]

    total_cases = 2 ** (2 * n)

    if n == 4:
        requested = values["n4_cases"]
        batch_size = 64

    elif n <= 6:
        requested = values["n6_cases"]
        batch_size = 64

    else:
        requested = values["large_cases"]
        batch_size = 16 if n >= 8 else 32

    max_cases = min(
        requested,
        total_cases,
    )

    case_mode = (
        "exhaustive"
        if max_cases >= total_cases
        else "random"
    )

    return max_cases, case_mode, batch_size


def make_cases(
    n: int,
    max_cases: int,
    case_mode: str,
    seed: int,
) -> List[Tuple[int, int]]:
    dimension = 2**n
    total_cases = dimension * dimension

    if max_cases <= 0 or max_cases >= total_cases:
        return list(
            itertools.product(
                range(dimension),
                range(dimension),
            )
        )

    if case_mode == "random":
        rng = np.random.default_rng(seed)

        indices = rng.choice(
            total_cases,
            size=max_cases,
            replace=False,
        )

        indices.sort()

        return [
            (
                int(index // dimension),
                int(index % dimension),
            )
            for index in indices
        ]

    if case_mode == "first":
        return list(
            itertools.islice(
                itertools.product(
                    range(dimension),
                    range(dimension),
                ),
                max_cases,
            )
        )

    raise ValueError(
        f"Unsupported case mode: {case_mode}"
    )


def expected_all_integer(
    n: int,
    operand_a: int,
    operand_b: int,
) -> int:
    total = operand_a + operand_b

    low_sum = total & (
        (1 << n) - 1
    )

    carry = (
        total >> n
    ) & 1

    # Register order: A, B, X, Z.
    return (
        operand_a
        | (low_sum << n)
        | (carry << (2 * n + 1))
    )


def exact_verify_design(
    design: CircuitDesign,
    n: int,
    cases: Sequence[Tuple[int, int]],
    batch_size: int,
    seed: int,
    progress_every: int,
) -> Dict[str, object]:
    simulator = AerSimulator(
        method="matrix_product_state",
        max_parallel_experiments=1,
        max_parallel_shots=1,
        max_parallel_threads=1,
    )

    checked = 0
    start_time = time.time()

    for batch_start in range(
        0,
        len(cases),
        batch_size,
    ):
        batch_cases = list(
            cases[
                batch_start:
                batch_start + batch_size
            ]
        )

        circuits = [
            build_qrca(
                n,
                operand_a,
                operand_b,
                design.selected_site_ids,
                "all",
            )
            for operand_a, operand_b
            in batch_cases
        ]

        transpiled_circuits = transpile(
            circuits,
            simulator,
            optimization_level=0,
        )

        result = simulator.run(
            transpiled_circuits,
            shots=1,
            seed_simulator=seed + batch_start,
        ).result()

        counts_list = normalize_counts(
            result.get_counts()
        )

        for (
            operand_a,
            operand_b,
        ), counts in zip(
            batch_cases,
            counts_list,
        ):
            if len(counts) != 1:
                raise AssertionError(
                    f"Non-deterministic ideal result: {counts}"
                )

            bitstring = next(
                iter(counts)
            ).replace(" ", "")

            measured = int(
                bitstring,
                2,
            )

            expected = expected_all_integer(
                n,
                operand_a,
                operand_b,
            )

            if measured != expected:
                raise AssertionError(
                    "QRCA exact verification failed: "
                    f"design={design.label}, "
                    f"a={operand_a}, b={operand_b}, "
                    f"measured={measured}, expected={expected}"
                )

        checked += len(batch_cases)

        if (
            progress_every > 0
            and (
                checked == len(cases)
                or checked % progress_every == 0
            )
        ):
            print(
                f"      exact {design.label}: "
                f"{checked}/{len(cases)}",
                flush=True,
            )

        del circuits
        del transpiled_circuits
        del result
        del counts_list

        gc.collect()

    elapsed = time.time() - start_time

    del simulator
    gc.collect()

    return {
        "design": design.label,
        "design_type": design.design_type,
        "selected_site_ids": ",".join(
            str(value)
            for value in design.selected_site_ids
        ),
        "n": n,
        "cases": len(cases),
        "passed": True,
        "time_s": elapsed,
    }


def run_noisy_design(
    design: CircuitDesign,
    noise_name: str,
    config: ExperimentConfig,
    cases: Sequence[Tuple[int, int]],
) -> Tuple[
    List[Dict[str, object]],
    Dict[str, object],
]:
    noise_model = build_noise_model(
        noise_name,
        config.noise_parameters,
    )

    simulator = AerSimulator(
        method="matrix_product_state",
        noise_model=noise_model,
        max_parallel_experiments=1,
        max_parallel_shots=1,
        max_parallel_threads=1,
    )

    repeat_rows = []

    for repeat_index in range(config.repeats):
        total_med = 0.0
        modal_error_count = 0
        execution_error_total = 0.0

        max_ed_mode = 0
        max_ed_observed = 0

        completed = 0
        start_time = time.time()

        for batch_start in range(
            0,
            len(cases),
            config.batch_size,
        ):
            batch_cases = list(
                cases[
                    batch_start:
                    batch_start + config.batch_size
                ]
            )

            circuits = [
                build_qrca(
                    config.n,
                    operand_a,
                    operand_b,
                    design.selected_site_ids,
                    "sum",
                )
                for operand_a, operand_b
                in batch_cases
            ]

            transpiled_circuits = transpile(
                circuits,
                basis_gates=EXECUTION_BASIS,
                optimization_level=0,
            )

            simulation_seed = (
                config.seed
                + repeat_index * 100_000
                + batch_start
            )

            result = simulator.run(
                transpiled_circuits,
                shots=config.shots,
                seed_simulator=simulation_seed,
            ).result()

            counts_list = normalize_counts(
                result.get_counts()
            )

            for (
                operand_a,
                operand_b,
            ), counts in zip(
                batch_cases,
                counts_list,
            ):
                expected = operand_a + operand_b

                distribution = counts_to_distribution(
                    counts,
                    config.shots,
                )

                total_med += sum(
                    probability
                    * abs(output - expected)
                    for output, probability
                    in distribution.items()
                )

                execution_error_total += (
                    1.0
                    - distribution.get(
                        expected,
                        0.0,
                    )
                )

                modal_output = int(
                    max(
                        counts.items(),
                        key=lambda item: (
                            item[1],
                            -int(
                                item[0].replace(" ", ""),
                                2,
                            ),
                        ),
                    )[0].replace(" ", ""),
                    2,
                )

                modal_ed = abs(
                    modal_output - expected
                )

                if modal_output != expected:
                    modal_error_count += 1

                max_ed_mode = max(
                    max_ed_mode,
                    modal_ed,
                )

                for output, probability in distribution.items():
                    if probability > 0:
                        max_ed_observed = max(
                            max_ed_observed,
                            abs(output - expected),
                        )

            completed += len(batch_cases)

            if (
                config.progress_every > 0
                and (
                    completed == len(cases)
                    or completed
                    % config.progress_every
                    == 0
                )
            ):
                print(
                    f"      {noise_name} "
                    f"repeat={repeat_index + 1}/"
                    f"{config.repeats} "
                    f"cases={completed}/{len(cases)}",
                    flush=True,
                )

            del circuits
            del transpiled_circuits
            del result
            del counts_list

            gc.collect()

        med = total_med / len(cases)

        nmed = med / (
            (2 ** (config.n + 1)) - 1
        )

        er = modal_error_count / len(cases)

        execution_error_probability = (
            execution_error_total
            / len(cases)
        )

        elapsed = time.time() - start_time

        repeat_rows.append(
            {
                "noise": noise_name,
                "design": design.label,
                "design_type": design.design_type,
                "selected_site_ids": ",".join(
                    str(value)
                    for value in design.selected_site_ids
                ),
                "n": config.n,
                "repeat": repeat_index + 1,
                "cases": len(cases),
                "case_mode": config.case_mode,
                "shots": config.shots,
                "MED": med,
                "NMED": nmed,
                "ER": er,
                "execution_error_probability": (
                    execution_error_probability
                ),
                "MaxED_mode": max_ed_mode,
                "MaxED_observed": max_ed_observed,
                "time_s": elapsed,
            }
        )

    med_values = np.array(
        [
            row["MED"]
            for row in repeat_rows
        ],
        dtype=float,
    )

    nmed_values = np.array(
        [
            row["NMED"]
            for row in repeat_rows
        ],
        dtype=float,
    )

    er_values = np.array(
        [
            row["ER"]
            for row in repeat_rows
        ],
        dtype=float,
    )

    execution_values = np.array(
        [
            row["execution_error_probability"]
            for row in repeat_rows
        ],
        dtype=float,
    )

    summary = {
        "noise": noise_name,
        "design": design.label,
        "design_type": design.design_type,
        "selected_site_ids": ",".join(
            str(value)
            for value in design.selected_site_ids
        ),
        "n": config.n,
        "cases": len(cases),
        "case_mode": config.case_mode,
        "shots": config.shots,
        "repeats": config.repeats,
        "MED_mean": float(np.mean(med_values)),
        "MED_std": (
            float(np.std(med_values, ddof=1))
            if config.repeats > 1
            else 0.0
        ),
        "NMED_mean": float(np.mean(nmed_values)),
        "NMED_std": (
            float(np.std(nmed_values, ddof=1))
            if config.repeats > 1
            else 0.0
        ),
        "ER_mean": float(np.mean(er_values)),
        "ER_std": (
            float(np.std(er_values, ddof=1))
            if config.repeats > 1
            else 0.0
        ),
        "execution_error_probability_mean": float(
            np.mean(execution_values)
        ),
        "execution_error_probability_std": (
            float(np.std(execution_values, ddof=1))
            if config.repeats > 1
            else 0.0
        ),
        "MaxED_mode": max(
            row["MaxED_mode"]
            for row in repeat_rows
        ),
        "MaxED_observed": max(
            row["MaxED_observed"]
            for row in repeat_rows
        ),
    }

    del simulator
    del noise_model

    gc.collect()

    return repeat_rows, summary


def create_comparison_summary(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if summary_df.empty:
        return pd.DataFrame()

    for noise_name, group in summary_df.groupby("noise"):
        baseline_rows = group[
            group["design"] == "conventional_ccx"
        ]

        if baseline_rows.empty:
            continue

        baseline = baseline_rows.iloc[0]

        proposed_rows = group[
            group["design"] != "conventional_ccx"
        ]

        for _, proposed in proposed_rows.iterrows():
            rows.append(
                {
                    "noise": noise_name,
                    "baseline": baseline["design"],
                    "proposed": proposed["design"],
                    "selected_site_ids": (
                        proposed["selected_site_ids"]
                    ),
                    "baseline_MED": baseline["MED_mean"],
                    "proposed_MED": proposed["MED_mean"],
                    "delta_MED_pct": safe_percentage(
                        proposed["MED_mean"],
                        baseline["MED_mean"],
                    ),
                    "baseline_NMED": baseline["NMED_mean"],
                    "proposed_NMED": proposed["NMED_mean"],
                    "delta_NMED_pct": safe_percentage(
                        proposed["NMED_mean"],
                        baseline["NMED_mean"],
                    ),
                    "baseline_ER": baseline["ER_mean"],
                    "proposed_ER": proposed["ER_mean"],
                    "delta_ER_pct": safe_percentage(
                        proposed["ER_mean"],
                        baseline["ER_mean"],
                    ),
                }
            )

    return pd.DataFrame(rows)


def prompt_int(
    message: str,
    default: int,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    while True:
        raw = input(
            f"{message} [{default}]: "
        ).strip()

        try:
            value = default if not raw else int(raw)
        except ValueError:
            print("  Enter an integer.")
            continue

        if minimum is not None and value < minimum:
            print(f"  Minimum value is {minimum}.")
            continue

        if maximum is not None and value > maximum:
            print(f"  Maximum value is {maximum}.")
            continue

        return value


def prompt_float(
    message: str,
    default: float,
) -> float:
    while True:
        raw = input(
            f"{message} [{default}]: "
        ).strip()

        try:
            return default if not raw else float(raw)
        except ValueError:
            print("  Enter a numeric value.")


def prompt_choice(
    message: str,
    options: Sequence[str],
    default_index: int,
) -> str:
    while True:
        print(f"\n{message}")

        for index, option in enumerate(
            options,
            start=1,
        ):
            mark = (
                " (default)"
                if index == default_index
                else ""
            )

            print(
                f"  [{index}] {option}{mark}"
            )

        raw = input("Select: ").strip()

        if not raw:
            return options[default_index - 1]

        try:
            index = int(raw)
        except ValueError:
            print("  Enter a menu number.")
            continue

        if 1 <= index <= len(options):
            return options[index - 1]

        print("  Invalid selection.")


def prompt_noise_models() -> List[str]:
    options = [
        "thermal",
        "depolarizing",
        "bitflip",
        "lindblad",
    ]

    print("\nNoise models")
    print("  [0] All four")

    for index, option in enumerate(
        options,
        start=1,
    ):
        print(
            f"  [{index}] {option}"
        )

    raw = input(
        "Select comma-separated numbers [0]: "
    ).strip()

    if not raw or raw == "0":
        return options

    try:
        indices = [
            int(value.strip())
            for value in raw.split(",")
        ]
    except ValueError:
        return options

    selected = [
        options[index - 1]
        for index in indices
        if 1 <= index <= len(options)
    ]

    return selected or options


def prompt_noise_parameters() -> NoiseParameters:
    parameters = NoiseParameters()

    use_defaults = prompt_choice(
        "Use paper-oriented default noise parameters?",
        [
            "yes",
            "no",
        ],
        default_index=1,
    )

    if use_defaults == "yes":
        return parameters

    parameters.T1_us = prompt_float(
        "T1 in microseconds",
        parameters.T1_us,
    )

    parameters.T2_us = prompt_float(
        "T2 in microseconds",
        parameters.T2_us,
    )

    parameters.one_q_gate_time_ns = prompt_float(
        "1Q gate time in ns",
        parameters.one_q_gate_time_ns,
    )

    parameters.two_q_gate_time_ns = prompt_float(
        "2Q gate time in ns",
        parameters.two_q_gate_time_ns,
    )

    parameters.depolarizing_1q = prompt_float(
        "1Q depolarizing probability",
        parameters.depolarizing_1q,
    )

    parameters.depolarizing_2q = prompt_float(
        "2Q depolarizing probability",
        parameters.depolarizing_2q,
    )

    parameters.bitflip_probability = prompt_float(
        "Bit-flip probability",
        parameters.bitflip_probability,
    )

    parameters.lindblad_amp = prompt_float(
        "Amplitude-damping probability",
        parameters.lindblad_amp,
    )

    parameters.lindblad_phase = prompt_float(
        "Phase-damping probability",
        parameters.lindblad_phase,
    )

    parameters.lindblad_correlated_2q = prompt_float(
        "Correlated 2Q phase probability",
        parameters.lindblad_correlated_2q,
    )

    validate_noise_parameters(parameters)

    return parameters


def build_interactive_config() -> ExperimentConfig:
    print("=" * 88)
    print(PROGRAM_TITLE)
    print(f"Version {PROGRAM_VERSION}")
    print("=" * 88)

    print(
        "Press Enter to use each recommended default."
    )

    n = prompt_int(
        "\nQRCA bit-width",
        default=6,
        minimum=4,
        maximum=12,
    )

    mode = prompt_choice(
        "Execution mode",
        [
            "complete",
            "analyze",
            "exact",
            "noise",
        ],
        default_index=1,
    )

    preset = prompt_choice(
        "Experiment preset",
        [
            "demo",
            "paper",
            "high",
        ],
        default_index=2,
    )

    max_cases, case_mode, batch_size = (
        automatic_case_settings(
            n,
            preset,
        )
    )

    shots = PRESETS[preset]["shots"]
    repeats = PRESETS[preset]["repeats"]

    site_policy = prompt_choice(
        "RCCX placement policy",
        [
            "min_p",
            "manual",
            "all_sites",
        ],
        default_index=1,
    )

    manual_sites: List[int] = []

    if site_policy == "manual":
        raw = input(
            f"Site IDs [default={n - 1}]: "
        ).strip()

        manual_sites = [
            int(value.strip())
            for value in (
                raw or str(n - 1)
            ).split(",")
            if value.strip()
        ]

    noise_models = prompt_noise_models()
    noise_parameters = prompt_noise_parameters()

    seed = prompt_int(
        "\nRandom seed",
        default=20260805,
        minimum=0,
    )

    progress_every = prompt_int(
        "Progress interval",
        default=256,
        minimum=0,
    )

    default_output = (
        f"qrca_dependency_noise_n{n}_"
        f"{preset}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )

    output_value = input(
        f"Output directory [{default_output}]: "
    ).strip()

    return ExperimentConfig(
        mode=mode,
        n=n,
        preset=preset,
        shots=shots,
        repeats=repeats,
        max_cases=max_cases,
        case_mode=case_mode,
        batch_size=batch_size,
        seed=seed,
        site_policy=site_policy,
        manual_sites=manual_sites,
        noise_models=noise_models,
        progress_every=progress_every,
        output_dir=output_value or default_output,
        noise_parameters=noise_parameters,
    )


def save_noise_checkpoint(
    output_dir: Path,
    repeat_rows: Sequence[Dict[str, object]],
    summary_rows: Sequence[Dict[str, object]],
) -> None:
    repeat_df = pd.DataFrame(repeat_rows)
    summary_df = pd.DataFrame(summary_rows)

    repeat_df.to_csv(
        output_dir / "qrca_noise_repeat_results.csv",
        index=False,
    )

    summary_df.to_csv(
        output_dir / "qrca_noise_summary.csv",
        index=False,
    )

    create_comparison_summary(
        summary_df
    ).to_csv(
        output_dir / "qrca_comparison_summary.csv",
        index=False,
    )


def main() -> None:
    config = build_interactive_config()

    validate_noise_parameters(
        config.noise_parameters
    )

    output_dir = Path(config.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        output_dir / "run_config.json",
        {
            "program_title": PROGRAM_TITLE,
            "program_version": PROGRAM_VERSION,
            **asdict(config),
        },
    )

    sites = enumerate_qrca_sites(config.n)

    edges, p_values = analyze_qrca_dependencies(
        config.n,
        sites,
    )

    designs = create_designs(
        sites,
        p_values,
        config.site_policy,
        config.manual_sites,
    )

    selected_ids = {
        site_id
        for design in designs
        if design.design_type != "conventional"
        for site_id in design.selected_site_ids
    }

    pd.DataFrame(
        [asdict(site) for site in sites]
    ).to_csv(
        output_dir / "qrca_toffoli_site_catalog.csv",
        index=False,
    )

    pd.DataFrame(
        [asdict(edge) for edge in edges]
    ).to_csv(
        output_dir / "qrca_dependency_edges.csv",
        index=False,
    )

    propagation_rows = []

    for site in sites:
        if not site.analyzed:
            continue

        propagation_rows.append(
            {
                **asdict(site),
                "P_value": p_values[site.site_id],
                "selected": site.site_id in selected_ids,
            }
        )

        mark = (
            " <SELECTED>"
            if site.site_id in selected_ids
            else ""
        )

        print(
            f"l{site.site_id}: "
            f"P(l_i)={p_values[site.site_id]}, "
            f"bit={site.logical_bit}, "
            f"role={site.role}{mark}"
        )

    pd.DataFrame(
        propagation_rows
    ).to_csv(
        output_dir / "qrca_propagation_depth.csv",
        index=False,
    )

    selected_rows = []

    for design in designs:
        for site_id in design.selected_site_ids:
            site = next(
                value
                for value in sites
                if value.site_id == site_id
            )

            selected_rows.append(
                {
                    "design": design.label,
                    "site_id": site_id,
                    "logical_bit": site.logical_bit,
                    "role": site.role,
                    "P_value": p_values[site_id],
                }
            )

    pd.DataFrame(
        selected_rows
    ).to_csv(
        output_dir / "qrca_selected_sites.csv",
        index=False,
    )

    if config.mode == "analyze":
        return

    cases = make_cases(
        config.n,
        config.max_cases,
        config.case_mode,
        config.seed,
    )

    if config.mode in (
        "complete",
        "exact",
    ):
        exact_rows = []

        for design in designs:
            print(
                f"\nIdeal verification: {design.label}"
            )

            exact_rows.append(
                exact_verify_design(
                    design,
                    config.n,
                    cases,
                    config.batch_size,
                    config.seed,
                    config.progress_every,
                )
            )

            pd.DataFrame(
                exact_rows
            ).to_csv(
                output_dir
                / "qrca_ideal_correctness.csv",
                index=False,
            )

        if config.mode == "exact":
            return

    if config.mode in (
        "complete",
        "noise",
    ):
        repeat_rows = []
        summary_rows = []

        for noise_name in config.noise_models:
            print(
                f"\nNoise model: {noise_name}"
            )

            for design in designs:
                print(
                    f"  Running {design.label}"
                )

                new_repeat_rows, summary = (
                    run_noisy_design(
                        design,
                        noise_name,
                        config,
                        cases,
                    )
                )

                repeat_rows.extend(
                    new_repeat_rows
                )

                summary_rows.append(
                    summary
                )

                save_noise_checkpoint(
                    output_dir,
                    repeat_rows,
                    summary_rows,
                )

    print(
        f"\nResults saved to: {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
