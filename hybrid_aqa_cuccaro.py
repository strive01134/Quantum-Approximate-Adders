#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dependency-Aware QRCA Demo
==========================

A portfolio-oriented Qiskit demonstration of a quantum
ripple-carry adder (QRCA) with dependency-aware RCCX placement.

This program demonstrates:

1. Construction of an n-bit QRCA.
2. Enumeration of Toffoli locations in the carry network.
3. Simple carry-dependency analysis for forward Toffoli sites.
4. Automatic selection of the least-dependent forward site.
5. Replacement of the selected CCX gate with RCCX.
6. Computational-basis correctness verification.
7. Comparison of conventional and RCCX-based designs under:
       - Thermal relaxation noise
       - Depolarizing noise
       - Bit-flip noise
       - Damping / correlated-phase noise
8. Calculation of:
       - Mean Error Distance (MED)
       - Normalized MED (NMED)
       - Modal Error Rate
       - Shot Error Probability
       - Maximum Observed Error Distance

Example
-------
python qrca_dependency_demo.py --bits 4 --shots 100 --noise thermal

python qrca_dependency_demo.py --bits 6 --cases 512 \
    --shots 100 --noise all
"""

from __future__ import annotations

import argparse
import itertools

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

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


# ============================================================
# Global Simulation Basis
# ============================================================

BASIS_GATES = [
    "rz",
    "sx",
    "x",
    "cx",
]


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class DemoConfig:
    """Configuration for a QRCA demonstration."""

    bits: int = 4
    shots: int = 100
    cases: int = 0
    batch_size: int = 64

    noise: str = "thermal"

    seed: int = 42

    # -1 means automatic minimum-dependency site
    manual_site: int = -1

    def validate(self) -> None:

        if self.bits < 4:
            raise ValueError(
                "QRCA demo requires at least 4 bits."
            )

        if self.bits > 12:
            raise ValueError(
                "For this demonstration, use <= 12 bits."
            )

        if self.shots < 1:
            raise ValueError(
                "shots must be at least 1."
            )

        if self.cases < 0:
            raise ValueError(
                "cases must be >= 0."
            )

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        allowed_noise = {
            "ideal",
            "thermal",
            "depolarizing",
            "bitflip",
            "damping",
            "all",
        }

        if self.noise not in allowed_noise:
            raise ValueError(
                f"Unsupported noise mode: {self.noise}"
            )


@dataclass(frozen=True)
class ToffoliLocation:
    """
    Description of one Toffoli location
    in the QRCA carry network.
    """

    site_id: int
    stage: str
    logical_bit: int
    role: str
    analyzed: bool


@dataclass
class EvaluationMetrics:
    """Noise-evaluation result."""

    med: float
    nmed: float

    modal_error_rate: float
    shot_error_probability: float

    max_observed_ed: int

    cases: int
    shots: int


# ============================================================
# Dependency Analysis
# ============================================================

class CarryDependencyAnalyzer:
    """
    Analyze how strongly each forward Toffoli site is connected
    to later carry-generation and carry-restoration operations.
    """

    @staticmethod
    def enumerate_sites(
        n: int,
    ) -> List[ToffoliLocation]:

        if n < 4:
            raise ValueError(
                "n must be >= 4."
            )

        locations: List[ToffoliLocation] = []

        # ----------------------------------------------------
        # Forward carry network
        # ----------------------------------------------------

        locations.append(
            ToffoliLocation(
                site_id=0,
                stage="forward",
                logical_bit=0,
                role="initial_carry",
                analyzed=True,
            )
        )

        locations.append(
            ToffoliLocation(
                site_id=1,
                stage="forward",
                logical_bit=1,
                role="forward_head",
                analyzed=True,
            )
        )

        next_id = 2

        for logical_bit in range(
            2,
            n - 2,
        ):

            locations.append(
                ToffoliLocation(
                    site_id=next_id,
                    stage="forward",
                    logical_bit=logical_bit,
                    role="carry_propagation",
                    analyzed=True,
                )
            )

            next_id += 1

        locations.append(
            ToffoliLocation(
                site_id=next_id,
                stage="forward",
                logical_bit=n - 2,
                role="final_carry_prepare",
                analyzed=True,
            )
        )

        next_id += 1

        locations.append(
            ToffoliLocation(
                site_id=next_id,
                stage="forward",
                logical_bit=n - 1,
                role="carry_out",
                analyzed=True,
            )
        )

        next_id += 1

        # ----------------------------------------------------
        # Backward restoration network
        # ----------------------------------------------------

        locations.append(
            ToffoliLocation(
                site_id=next_id,
                stage="backward",
                logical_bit=n - 2,
                role="final_restore",
                analyzed=False,
            )
        )

        next_id += 1

        for logical_bit in reversed(
            range(
                2,
                n - 2,
            )
        ):

            locations.append(
                ToffoliLocation(
                    site_id=next_id,
                    stage="backward",
                    logical_bit=logical_bit,
                    role="carry_restore",
                    analyzed=False,
                )
            )

            next_id += 1

        locations.append(
            ToffoliLocation(
                site_id=next_id,
                stage="backward",
                logical_bit=1,
                role="backward_head",
                analyzed=False,
            )
        )

        next_id += 1

        locations.append(
            ToffoliLocation(
                site_id=next_id,
                stage="backward",
                logical_bit=0,
                role="initial_restore",
                analyzed=False,
            )
        )

        expected = 2 * n - 1

        if len(locations) != expected:
            raise AssertionError(
                f"Expected {expected} Toffoli sites, "
                f"but found {len(locations)}."
            )

        return locations

    @staticmethod
    def calculate_scores(
        n: int,
        sites: Sequence[ToffoliLocation],
    ) -> Dict[int, int]:
        """
        Calculate a simple dependency count D(l).

        For each analyzed forward location, count:

        1. Later forward carry operations that depend on
           the propagated carry.

        2. Backward restoration operations whose state
           depends on that carry history.
        """

        forward = [
            site
            for site in sites
            if site.stage == "forward"
        ]

        backward = [
            site
            for site in sites
            if site.stage == "backward"
        ]

        analyzed = [
            site
            for site in forward
            if site.analyzed
        ]

        scores: Dict[int, int] = {}

        for source in analyzed:

            dependency_count = 0

            # Later forward carry operations.
            for target in forward:

                if (
                    target.logical_bit
                    > source.logical_bit
                ):
                    dependency_count += 1

            # Restoration operations influenced by
            # this position or later positions.
            for target in backward:

                if (
                    target.logical_bit
                    >= source.logical_bit
                ):
                    dependency_count += 1

            scores[source.site_id] = (
                dependency_count
            )

        return scores

    @staticmethod
    def choose_site(
        sites: Sequence[ToffoliLocation],
        scores: Dict[int, int],
    ) -> ToffoliLocation:
        """
        Select the least-dependent forward Toffoli location.

        Tie breaking:
            1. smaller dependency score
            2. more significant logical bit
            3. smaller site ID
        """

        candidates = [
            site
            for site in sites
            if site.analyzed
        ]

        if not candidates:
            raise ValueError(
                "No candidate sites found."
            )

        return min(
            candidates,
            key=lambda site: (
                scores[site.site_id],
                -site.logical_bit,
                site.site_id,
            ),
        )


# ============================================================
# Noise Models
# ============================================================

class NoiseFactory:
    """Build noise models for the portfolio demonstration."""

    @staticmethod
    def thermal() -> NoiseModel:

        model = NoiseModel()

        # Example coherence parameters.
        # Units: nanoseconds.
        t1 = 120_000.0
        t2 = 90_000.0

        gate_1q = 35.0
        gate_2q = 250.0

        error_1q = thermal_relaxation_error(
            t1,
            t2,
            gate_1q,
        )

        one_component_2q = (
            thermal_relaxation_error(
                t1,
                t2,
                gate_2q,
            )
        )

        error_2q = one_component_2q.tensor(
            one_component_2q
        )

        for gate in [
            "rz",
            "sx",
            "x",
        ]:
            model.add_all_qubit_quantum_error(
                error_1q,
                [gate],
            )

        model.add_all_qubit_quantum_error(
            error_2q,
            ["cx"],
        )

        return model

    @staticmethod
    def depolarizing() -> NoiseModel:

        model = NoiseModel()

        error_1q = depolarizing_error(
            0.001,
            1,
        )

        error_2q = depolarizing_error(
            0.010,
            2,
        )

        for gate in [
            "rz",
            "sx",
            "x",
        ]:
            model.add_all_qubit_quantum_error(
                error_1q,
                [gate],
            )

        model.add_all_qubit_quantum_error(
            error_2q,
            ["cx"],
        )

        return model

    @staticmethod
    def bitflip() -> NoiseModel:

        model = NoiseModel()

        probability = 0.001

        error_1q = pauli_error(
            [
                ("X", probability),
                ("I", 1.0 - probability),
            ]
        )

        error_2q = pauli_error(
            [
                ("IX", probability / 2.0),
                ("XI", probability / 2.0),
                ("II", 1.0 - probability),
            ]
        )

        for gate in [
            "rz",
            "sx",
            "x",
        ]:
            model.add_all_qubit_quantum_error(
                error_1q,
                [gate],
            )

        model.add_all_qubit_quantum_error(
            error_2q,
            ["cx"],
        )

        return model

    @staticmethod
    def damping() -> NoiseModel:
        """
        Demonstration model combining:

        - amplitude/phase damping
        - weak correlated ZZ phase error
        """

        model = NoiseModel()

        amplitude = 0.0045
        phase = 0.0090
        correlated = 0.0005

        error_1q = phase_amplitude_damping_error(
            amplitude,
            phase,
        )

        independent_2q = (
            error_1q.tensor(
                error_1q
            )
        )

        correlated_phase = pauli_error(
            [
                ("ZZ", correlated),
                ("II", 1.0 - correlated),
            ]
        )

        error_2q = independent_2q.compose(
            correlated_phase
        )

        for gate in [
            "rz",
            "sx",
            "x",
        ]:
            model.add_all_qubit_quantum_error(
                error_1q,
                [gate],
            )

        model.add_all_qubit_quantum_error(
            error_2q,
            ["cx"],
        )

        return model

    @classmethod
    def create(
        cls,
        name: str,
    ) -> Optional[NoiseModel]:

        if name == "ideal":
            return None

        if name == "thermal":
            return cls.thermal()

        if name == "depolarizing":
            return cls.depolarizing()

        if name == "bitflip":
            return cls.bitflip()

        if name == "damping":
            return cls.damping()

        raise ValueError(
            f"Unknown noise model: {name}"
        )


# ============================================================
# QRCA Circuit
# ============================================================

class QRCACircuit:
    """
    Quantum ripple-carry adder with optional
    relative-phase Toffoli substitution.
    """

    def __init__(
        self,
        n: int,
        relative_phase_sites: Iterable[int] = (),
    ):

        if n < 4:
            raise ValueError(
                "QRCA requires n >= 4."
            )

        self.n = n

        self.relative_phase_sites: Set[int] = set(
            relative_phase_sites
        )

    def _toffoli(
        self,
        circuit: QuantumCircuit,
        control_0,
        control_1,
        target,
        site_id: int,
    ) -> None:

        if site_id in self.relative_phase_sites:

            circuit.rccx(
                control_0,
                control_1,
                target,
            )

        else:

            circuit.ccx(
                control_0,
                control_1,
                target,
            )

    @staticmethod
    def _initialize_integer(
        circuit: QuantumCircuit,
        register: QuantumRegister,
        value: int,
    ) -> None:

        for bit in range(
            len(register)
        ):

            if (
                value >> bit
            ) & 1:

                circuit.x(
                    register[bit]
                )

    def build(
        self,
        operand_a: int,
        operand_b: int,
        measure: bool = True,
    ) -> QuantumCircuit:

        n = self.n

        maximum = (
            1 << n
        ) - 1

        if not (
            0 <= operand_a <= maximum
        ):
            raise ValueError(
                "operand_a is outside range."
            )

        if not (
            0 <= operand_b <= maximum
        ):
            raise ValueError(
                "operand_b is outside range."
            )

        # ----------------------------------------------------
        # Registers
        # ----------------------------------------------------

        a = QuantumRegister(
            n,
            "a",
        )

        b = QuantumRegister(
            n,
            "b",
        )

        workspace = QuantumRegister(
            1,
            "work",
        )

        carry_out = QuantumRegister(
            1,
            "cout",
        )

        circuit = QuantumCircuit(
            a,
            b,
            workspace,
            carry_out,
        )

        # ----------------------------------------------------
        # Input encoding
        # ----------------------------------------------------

        self._initialize_integer(
            circuit,
            a,
            operand_a,
        )

        self._initialize_integer(
            circuit,
            b,
            operand_b,
        )

        # ----------------------------------------------------
        # Forward carry preparation
        # ----------------------------------------------------

        for bit in range(
            1,
            n,
        ):

            circuit.cx(
                a[bit],
                b[bit],
            )

        circuit.cx(
            a[1],
            workspace[0],
        )

        # Site 0
        self._toffoli(
            circuit,
            a[0],
            b[0],
            workspace[0],
            0,
        )

        circuit.cx(
            a[2],
            a[1],
        )

        # Site 1
        self._toffoli(
            circuit,
            workspace[0],
            b[1],
            a[1],
            1,
        )

        circuit.cx(
            a[3],
            a[2],
        )

        site_id = 2

        # ----------------------------------------------------
        # Forward carry propagation
        # ----------------------------------------------------

        for bit in range(
            2,
            n - 2,
        ):

            self._toffoli(
                circuit,
                a[bit - 1],
                b[bit],
                a[bit],
                site_id,
            )

            site_id += 1

            circuit.cx(
                a[bit + 2],
                a[bit + 1],
            )

        # Final carry preparation
        self._toffoli(
            circuit,
            a[n - 3],
            b[n - 2],
            a[n - 2],
            site_id,
        )

        site_id += 1

        # ----------------------------------------------------
        # Carry output
        # ----------------------------------------------------

        circuit.cx(
            a[n - 1],
            carry_out[0],
        )

        self._toffoli(
            circuit,
            a[n - 2],
            b[n - 1],
            carry_out[0],
            site_id,
        )

        site_id += 1

        # ----------------------------------------------------
        # Sum / reverse network preparation
        # ----------------------------------------------------

        for bit in range(
            1,
            n - 1,
        ):

            circuit.x(
                b[bit]
            )

        circuit.cx(
            workspace[0],
            b[1],
        )

        for bit in range(
            2,
            n,
        ):

            circuit.cx(
                a[bit - 1],
                b[bit],
            )

        # ----------------------------------------------------
        # Backward restoration
        # ----------------------------------------------------

        self._toffoli(
            circuit,
            a[n - 3],
            b[n - 2],
            a[n - 2],
            site_id,
        )

        site_id += 1

        for bit in reversed(
            range(
                2,
                n - 2,
            )
        ):

            self._toffoli(
                circuit,
                a[bit - 1],
                b[bit],
                a[bit],
                site_id,
            )

            site_id += 1

            circuit.cx(
                a[bit + 2],
                a[bit + 1],
            )

            circuit.x(
                b[bit + 1]
            )

        self._toffoli(
            circuit,
            workspace[0],
            b[1],
            a[1],
            site_id,
        )

        site_id += 1

        circuit.cx(
            a[3],
            a[2],
        )

        circuit.x(
            b[2]
        )

        self._toffoli(
            circuit,
            a[0],
            b[0],
            workspace[0],
            site_id,
        )

        site_id += 1

        circuit.cx(
            a[2],
            a[1],
        )

        circuit.x(
            b[1]
        )

        circuit.cx(
            a[1],
            workspace[0],
        )

        # ----------------------------------------------------
        # Final sum formation
        # ----------------------------------------------------

        for bit in range(
            n
        ):

            circuit.cx(
                a[bit],
                b[bit],
            )

        expected_sites = (
            2 * n - 1
        )

        if site_id != expected_sites:

            raise AssertionError(
                "QRCA Toffoli site indexing failed."
            )

        # ----------------------------------------------------
        # Measurement
        # ----------------------------------------------------

        if measure:

            result = ClassicalRegister(
                n + 1,
                "sum",
            )

            circuit.add_register(
                result
            )

            for bit in range(
                n
            ):

                circuit.measure(
                    b[bit],
                    result[bit],
                )

            circuit.measure(
                carry_out[0],
                result[n],
            )

        return circuit


# ============================================================
# Test Case Generation
# ============================================================

def create_cases(
    n: int,
    requested_cases: int,
    seed: int,
) -> List[Tuple[int, int]]:
    """
    Generate test input pairs.

    requested_cases == 0:
        n <= 4 : exhaustive
        n >  4 : 512 random unique pairs
    """

    dimension = (
        1 << n
    )

    total = (
        dimension
        * dimension
    )

    if requested_cases == 0:

        if n <= 4:
            requested_cases = total

        else:
            requested_cases = min(
                512,
                total,
            )

    if requested_cases >= total:

        return list(
            itertools.product(
                range(dimension),
                range(dimension),
            )
        )

    rng = np.random.default_rng(
        seed
    )

    flat_indices = rng.choice(
        total,
        size=requested_cases,
        replace=False,
    )

    flat_indices.sort()

    return [
        (
            int(index // dimension),
            int(index % dimension),
        )
        for index in flat_indices
    ]


# ============================================================
# Evaluator
# ============================================================

class QRCAEvaluator:

    def __init__(
        self,
        config: DemoConfig,
    ):

        self.config = config

    @staticmethod
    def _normalize_counts(
        counts,
    ):

        if isinstance(
            counts,
            dict,
        ):

            return [
                counts
            ]

        return counts

    @staticmethod
    def _decode(
        bitstring: str,
    ) -> int:

        clean = bitstring.replace(
            " ",
            "",
        )

        return int(
            clean,
            2,
        )

    def verify_ideal(
        self,
        circuit_builder: QRCACircuit,
        cases: Sequence[Tuple[int, int]],
    ) -> bool:
        """
        Verify computational-basis arithmetic correctness.
        """

        simulator = AerSimulator()

        total_cases = len(
            cases
        )

        for batch_start in range(
            0,
            total_cases,
            self.config.batch_size,
        ):

            batch = list(
                cases[
                    batch_start:
                    batch_start
                    + self.config.batch_size
                ]
            )

            circuits = [
                circuit_builder.build(
                    a,
                    b,
                    measure=True,
                )
                for a, b in batch
            ]

            compiled = transpile(
                circuits,
                backend=simulator,
                optimization_level=0,
            )

            result = simulator.run(
                compiled,
                shots=1,
                seed_simulator=(
                    self.config.seed
                    + batch_start
                ),
            ).result()

            counts_list = (
                self._normalize_counts(
                    result.get_counts()
                )
            )

            for (
                operand_a,
                operand_b,
            ), counts in zip(
                batch,
                counts_list,
            ):

                if len(counts) != 1:

                    raise AssertionError(
                        "Ideal QRCA produced a "
                        "non-deterministic result."
                    )

                measured = self._decode(
                    next(
                        iter(counts)
                    )
                )

                expected = (
                    operand_a
                    + operand_b
                )

                if measured != expected:

                    raise AssertionError(
                        "Ideal verification failed: "
                        f"A={operand_a}, "
                        f"B={operand_b}, "
                        f"measured={measured}, "
                        f"expected={expected}"
                    )

        return True

    def evaluate_noise(
        self,
        circuit_builder: QRCACircuit,
        cases: Sequence[Tuple[int, int]],
        noise_name: str,
    ) -> EvaluationMetrics:

        noise_model = (
            NoiseFactory.create(
                noise_name
            )
        )

        if noise_model is None:

            simulator = AerSimulator()

        else:

            simulator = AerSimulator(
                noise_model=noise_model
            )

        total_expected_ed = 0.0
        total_modal_errors = 0

        total_execution_error = 0.0

        max_observed_ed = 0

        total_cases = len(
            cases
        )

        for batch_start in range(
            0,
            total_cases,
            self.config.batch_size,
        ):

            batch = list(
                cases[
                    batch_start:
                    batch_start
                    + self.config.batch_size
                ]
            )

            circuits = [
                circuit_builder.build(
                    a,
                    b,
                    measure=True,
                )
                for a, b in batch
            ]

            # Decompose CCX/RCCX to the same execution basis
            # so that CX-based noise is applied consistently.
            compiled = transpile(
                circuits,
                basis_gates=BASIS_GATES,
                optimization_level=0,
            )

            result = simulator.run(
                compiled,
                shots=self.config.shots,
                seed_simulator=(
                    self.config.seed
                    + batch_start
                ),
            ).result()

            counts_list = (
                self._normalize_counts(
                    result.get_counts()
                )
            )

            for (
                operand_a,
                operand_b,
            ), counts in zip(
                batch,
                counts_list,
            ):

                expected = (
                    operand_a
                    + operand_b
                )

                observed_shots = sum(
                    counts.values()
                )

                # ------------------------------------------------
                # Shot-weighted MED
                # ------------------------------------------------

                expected_ed = 0.0

                correct_probability = 0.0

                for (
                    bitstring,
                    frequency,
                ) in counts.items():

                    output = self._decode(
                        bitstring
                    )

                    probability = (
                        frequency
                        / observed_shots
                    )

                    ed = abs(
                        output
                        - expected
                    )

                    expected_ed += (
                        probability
                        * ed
                    )

                    if output == expected:

                        correct_probability += (
                            probability
                        )

                    if (
                        probability > 0
                    ):

                        max_observed_ed = max(
                            max_observed_ed,
                            ed,
                        )

                total_expected_ed += (
                    expected_ed
                )

                total_execution_error += (
                    1.0
                    - correct_probability
                )

                # ------------------------------------------------
                # Modal ER
                # ------------------------------------------------

                modal_bitstring = max(
                    counts.items(),
                    key=lambda item: (
                        item[1],
                        -self._decode(
                            item[0]
                        ),
                    ),
                )[0]

                modal_output = (
                    self._decode(
                        modal_bitstring
                    )
                )

                if modal_output != expected:

                    total_modal_errors += 1

        med = (
            total_expected_ed
            / total_cases
        )

        # Normalize by full unsigned range of
        # the (n+1)-bit sum register.
        normalization = (
            (1 << (self.config.bits + 1))
            - 1
        )

        nmed = (
            med
            / normalization
        )

        modal_error_rate = (
            total_modal_errors
            / total_cases
        )

        shot_error_probability = (
            total_execution_error
            / total_cases
        )

        return EvaluationMetrics(
            med=med,
            nmed=nmed,
            modal_error_rate=modal_error_rate,
            shot_error_probability=(
                shot_error_probability
            ),
            max_observed_ed=max_observed_ed,
            cases=total_cases,
            shots=self.config.shots,
        )


# ============================================================
# Circuit Resource Summary
# ============================================================

def circuit_summary(
    circuit_builder: QRCACircuit,
) -> Dict[str, object]:
    """
    Return structural information for a representative QRCA.
    """

    circuit = circuit_builder.build(
        0,
        0,
        measure=False,
    )

    basis_circuit = transpile(
        circuit,
        basis_gates=BASIS_GATES,
        optimization_level=0,
    )

    return {
        "logical_qubits": circuit.num_qubits,
        "original_depth": circuit.depth(),
        "original_ops": dict(
            circuit.count_ops()
        ),
        "basis_depth": basis_circuit.depth(),
        "basis_ops": dict(
            basis_circuit.count_ops()
        ),
    }


# ============================================================
# Display Helpers
# ============================================================

def print_dependency_table(
    sites: Sequence[ToffoliLocation],
    scores: Dict[int, int],
    selected_site: int,
) -> None:

    print(
        "\n========== Carry Dependency =========="
    )

    print(
        f"{'Site':<8}"
        f"{'Bit':<8}"
        f"{'Dependency':<14}"
        f"{'Role':<24}"
        f"{'Selected'}"
    )

    print(
        "-" * 68
    )

    for site in sites:

        if not site.analyzed:
            continue

        selected = (
            "YES"
            if site.site_id
            == selected_site
            else ""
        )

        print(
            f"l{site.site_id:<7}"
            f"{site.logical_bit:<8}"
            f"{scores[site.site_id]:<14}"
            f"{site.role:<24}"
            f"{selected}"
        )


def print_metrics(
    label: str,
    metrics: EvaluationMetrics,
) -> None:

    print(
        f"\n[{label}]"
    )

    print(
        f"MED                    : "
        f"{metrics.med:.8f}"
    )

    print(
        f"NMED                   : "
        f"{metrics.nmed:.8f}"
    )

    print(
        f"Modal Error Rate       : "
        f"{metrics.modal_error_rate:.8f}"
    )

    print(
        f"Shot Error Probability : "
        f"{metrics.shot_error_probability:.8f}"
    )

    print(
        f"Max Observed ED        : "
        f"{metrics.max_observed_ed}"
    )


# ============================================================
# CLI
# ============================================================

def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Dependency-aware QRCA / RCCX "
            "portfolio demonstration"
        )
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help=(
            "QRCA operand width "
            "(default: 4)"
        ),
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=100,
        help=(
            "Shots per input pair "
            "(default: 100)"
        ),
    )

    parser.add_argument(
        "--cases",
        type=int,
        default=0,
        help=(
            "Number of input pairs. "
            "0 = automatic "
            "(exhaustive for 4-bit, "
            "sampled for larger widths)"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Circuits per simulation batch "
            "(default: 64)"
        ),
    )

    parser.add_argument(
        "--noise",
        choices=[
            "ideal",
            "thermal",
            "depolarizing",
            "bitflip",
            "damping",
            "all",
        ],
        default="thermal",
        help="Noise model",
    )

    parser.add_argument(
        "--site",
        type=int,
        default=-1,
        help=(
            "Manually selected forward RCCX "
            "site ID. Default -1 selects the "
            "minimum-dependency site."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = create_parser().parse_args()

    config = DemoConfig(
        bits=args.bits,
        shots=args.shots,
        cases=args.cases,
        batch_size=args.batch_size,
        noise=args.noise,
        seed=args.seed,
        manual_site=args.site,
    )

    config.validate()

    print(
        "\n=============================================="
    )

    print(
        " Dependency-Aware QRCA Portfolio Demo"
    )

    print(
        "=============================================="
    )

    print(
        f"QRCA width     : "
        f"{config.bits}-bit"
    )

    print(
        f"Shots/input    : "
        f"{config.shots}"
    )

    # --------------------------------------------------------
    # Dependency analysis
    # --------------------------------------------------------

    sites = (
        CarryDependencyAnalyzer.enumerate_sites(
            config.bits
        )
    )

    scores = (
        CarryDependencyAnalyzer.calculate_scores(
            config.bits,
            sites,
        )
    )

    automatically_selected = (
        CarryDependencyAnalyzer.choose_site(
            sites,
            scores,
        )
    )

    valid_forward_sites = {
        site.site_id
        for site in sites
        if site.analyzed
    }

    if config.manual_site >= 0:

        if (
            config.manual_site
            not in valid_forward_sites
        ):

            raise ValueError(
                "Manual RCCX site must be "
                "one of the analyzed "
                f"forward sites: "
                f"{sorted(valid_forward_sites)}"
            )

        selected_site = (
            config.manual_site
        )

    else:

        selected_site = (
            automatically_selected.site_id
        )

    print_dependency_table(
        sites,
        scores,
        selected_site,
    )

    selected_location = next(
        site
        for site in sites
        if site.site_id
        == selected_site
    )

    print(
        "\nSelected RCCX location"
    )

    print(
        f"  Site       : l{selected_site}"
    )

    print(
        f"  Logical bit: "
        f"{selected_location.logical_bit}"
    )

    print(
        f"  Role       : "
        f"{selected_location.role}"
    )

    print(
        f"  Dependency : "
        f"{scores[selected_site]}"
    )

    # --------------------------------------------------------
    # Build two designs
    # --------------------------------------------------------

    conventional = QRCACircuit(
        n=config.bits,
        relative_phase_sites=(),
    )

    proposed = QRCACircuit(
        n=config.bits,
        relative_phase_sites=(
            selected_site,
        ),
    )

    # --------------------------------------------------------
    # Circuit summaries
    # --------------------------------------------------------

    conventional_summary = (
        circuit_summary(
            conventional
        )
    )

    proposed_summary = (
        circuit_summary(
            proposed
        )
    )

    print(
        "\n========== Circuit Summary =========="
    )

    print(
        "\nConventional QRCA"
    )

    for key, value in (
        conventional_summary.items()
    ):

        print(
            f"  {key:<16}: {value}"
        )

    print(
        "\nDependency-aware RCCX QRCA"
    )

    for key, value in (
        proposed_summary.items()
    ):

        print(
            f"  {key:<16}: {value}"
        )

    # --------------------------------------------------------
    # Input cases
    # --------------------------------------------------------

    cases = create_cases(
        n=config.bits,
        requested_cases=config.cases,
        seed=config.seed,
    )

    print(
        "\n========== Verification =========="
    )

    print(
        f"Input pairs: {len(cases):,}"
    )

    evaluator = QRCAEvaluator(
        config
    )

    print(
        "Checking conventional QRCA..."
    )

    evaluator.verify_ideal(
        conventional,
        cases,
    )

    print(
        "  PASS"
    )

    print(
        "Checking RCCX QRCA..."
    )

    evaluator.verify_ideal(
        proposed,
        cases,
    )

    print(
        "  PASS"
    )

    # --------------------------------------------------------
    # Noise selection
    # --------------------------------------------------------

    if config.noise == "all":

        noise_names = [
            "thermal",
            "depolarizing",
            "bitflip",
            "damping",
        ]

    else:

        noise_names = [
            config.noise
        ]

    # --------------------------------------------------------
    # Noise evaluation
    # --------------------------------------------------------

    print(
        "\n========== Noise Evaluation =========="
    )

    for noise_name in noise_names:

        print(
            f"\n### Noise: {noise_name}"
        )

        baseline_metrics = (
            evaluator.evaluate_noise(
                conventional,
                cases,
                noise_name,
            )
        )

        proposed_metrics = (
            evaluator.evaluate_noise(
                proposed,
                cases,
                noise_name,
            )
        )

        print_metrics(
            "Conventional CCX",
            baseline_metrics,
        )

        print_metrics(
            "Dependency-Aware RCCX",
            proposed_metrics,
        )

        # ----------------------------------------------------
        # Relative MED change
        # ----------------------------------------------------

        if baseline_metrics.med > 0:

            med_change = (
                (
                    proposed_metrics.med
                    - baseline_metrics.med
                )
                / baseline_metrics.med
                * 100.0
            )

            print(
                "\nMED change "
                f"(RCCX vs CCX): "
                f"{med_change:+.2f}%"
            )

        else:

            print(
                "\nMED change "
                "(RCCX vs CCX): N/A"
            )

    print(
        "\n=============================================="
    )

    print(
        " Demo completed."
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()
