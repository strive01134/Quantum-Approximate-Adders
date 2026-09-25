"""
Hybrid Approximate Quantum Adder 
=====================================

Portfolio-oriented implementation of a hybrid quantum adder.

The circuit combines:
    1. An approximate addition stage for the least-significant bits
    2. A Cuccaro ripple-carry adder for the remaining upper bits

Supported evaluation environments:
    - Ideal
    - Thermal relaxation noise
    - Depolarizing noise
    - Bit-flip noise

Evaluation metrics:
    - Error Rate (ER)
    - Mean Error Distance (MED)
    - Normalized MED (NMED)
    - Maximum observed Error Distance (MaxED)

Example:
    python hybrid_quantum_adder_demo.py \
        --bits 4 \
        --approx-bits 2 \
        --aqa-type 2 \
        --shots 100 \
        --noise ideal
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import argparse

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)

from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    thermal_relaxation_error,
    depolarizing_error,
    pauli_error,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    """Configuration for one hybrid-adder experiment."""

    total_bits: int = 4
    approx_bits: int = 2
    aqa_type: int = 2
    shots: int = 100
    noise_mode: str = "ideal"
    batch_size: int = 256
    seed: int = 42

    def validate(self) -> None:
        if self.total_bits < 2:
            raise ValueError("total_bits must be at least 2.")

        if not 0 <= self.approx_bits < self.total_bits:
            raise ValueError(
                "approx_bits must satisfy 0 <= approx_bits < total_bits."
            )

        if self.aqa_type not in {1, 2, 3, 4, 5}:
            raise ValueError("aqa_type must be between 1 and 5.")

        if self.shots < 1:
            raise ValueError("shots must be at least 1.")

        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        valid_noise_modes = {
            "ideal",
            "thermal",
            "depolarizing",
            "bitflip",
        }

        if self.noise_mode not in valid_noise_modes:
            raise ValueError(
                f"noise_mode must be one of {sorted(valid_noise_modes)}."
            )


# ============================================================
# Noise Model Factory
# ============================================================

class NoiseFactory:
    """Generate noise models used in the demonstration."""

    @staticmethod
    def create(mode: str) -> Optional[NoiseModel]:

        if mode == "ideal":
            return None

        noise_model = NoiseModel()

        if mode == "thermal":
            # Example device-level parameters in nanoseconds.
            t1 = 50_000
            t2 = 70_000

            single_gate_time = 100
            two_qubit_gate_time = 300

            single_error = thermal_relaxation_error(
                t1,
                t2,
                single_gate_time,
            )

            two_qubit_single_error = thermal_relaxation_error(
                t1,
                t2,
                two_qubit_gate_time,
            )

            two_qubit_error = (
                two_qubit_single_error.tensor(
                    two_qubit_single_error
                )
            )

            noise_model.add_all_qubit_quantum_error(
                single_error,
                ["x", "sx", "id"],
            )

            noise_model.add_all_qubit_quantum_error(
                two_qubit_error,
                ["cx"],
            )

        elif mode == "depolarizing":

            single_error = depolarizing_error(
                param=0.001,
                num_qubits=1,
            )

            two_qubit_error = depolarizing_error(
                param=0.01,
                num_qubits=2,
            )

            noise_model.add_all_qubit_quantum_error(
                single_error,
                ["x", "sx", "id"],
            )

            noise_model.add_all_qubit_quantum_error(
                two_qubit_error,
                ["cx"],
            )

        elif mode == "bitflip":

            single_error = pauli_error(
                [
                    ("X", 0.005),
                    ("I", 0.995),
                ]
            )

            two_qubit_error = single_error.tensor(
                single_error
            )

            noise_model.add_all_qubit_quantum_error(
                single_error,
                ["x", "sx", "id"],
            )

            noise_model.add_all_qubit_quantum_error(
                two_qubit_error,
                ["cx"],
            )

        return noise_model


# ============================================================
# Cuccaro Adder Primitives
# ============================================================

class CuccaroStage:
    """Primitive gates for a Cuccaro ripple-carry adder."""

    @staticmethod
    def majority(
        circuit: QuantumCircuit,
        a,
        b,
        carry,
    ) -> None:
        """
        Majority (MAJ) operation.

        The carry information is propagated through
        the A register.
        """

        circuit.cx(a, b)
        circuit.cx(a, carry)
        circuit.ccx(carry, b, a)

    @staticmethod
    def unmajority_add(
        circuit: QuantumCircuit,
        a,
        b,
        carry,
    ) -> None:
        """
        Unmajority-and-add (UMA) operation.

        Restores the workspace while storing
        the sum in the B register.
        """

        circuit.ccx(carry, b, a)
        circuit.cx(a, carry)
        circuit.cx(carry, b)

    @classmethod
    def apply(
        cls,
        circuit: QuantumCircuit,
        reg_a: QuantumRegister,
        reg_b: QuantumRegister,
        carry_in,
        carry_out,
        start: int,
        end: int,
    ) -> None:
        """
        Apply exact Cuccaro addition over
        bit positions [start, end].
        """

        if start > end:
            return

        # Forward carry propagation.
        cls.majority(
            circuit,
            reg_a[start],
            reg_b[start],
            carry_in,
        )

        for bit in range(start + 1, end + 1):
            cls.majority(
                circuit,
                reg_a[bit],
                reg_b[bit],
                reg_a[bit - 1],
            )

        # Copy final carry.
        circuit.cx(
            reg_a[end],
            carry_out,
        )

        # Reverse operation and sum generation.
        for bit in range(end, start, -1):
            cls.unmajority_add(
                circuit,
                reg_a[bit],
                reg_b[bit],
                reg_a[bit - 1],
            )

        cls.unmajority_add(
            circuit,
            reg_a[start],
            reg_b[start],
            carry_in,
        )


# ============================================================
# Approximate Adder Stage
# ============================================================

class ApproximateStage:
    """
    Approximate processing for the least-significant bits.

    The output sum is stored in the B register.

    Five demonstration variants are provided.
    """

    @staticmethod
    def _swap(
        circuit: QuantumCircuit,
        q1,
        q2,
    ) -> None:
        """
        SWAP implemented explicitly using three CNOT gates.

        This keeps the low-level gate structure visible
        for portfolio demonstrations.
        """

        circuit.cx(q1, q2)
        circuit.cx(q2, q1)
        circuit.cx(q1, q2)

    @classmethod
    def apply(
        cls,
        circuit: QuantumCircuit,
        reg_a: QuantumRegister,
        reg_b: QuantumRegister,
        carry,
        width: int,
        mode: int,
    ) -> None:
        """
        Apply the selected approximate strategy.

        mode 1:
            output B = original A

        mode 2:
            output B = A XOR B

        mode 3:
            output B = original A
            carry = MSB(B_approx)

        mode 4:
            output B = A XOR B
            carry = MSB(B_approx)

        mode 5:
            output B = A XOR B
            carry = MSB(A_approx) AND MSB(B_approx)
        """

        if width == 0:
            return

        msb = width - 1

        # ----------------------------------------------------
        # AQA-1
        # ----------------------------------------------------
        if mode == 1:

            for bit in range(width):
                cls._swap(
                    circuit,
                    reg_a[bit],
                    reg_b[bit],
                )

        # ----------------------------------------------------
        # AQA-2
        # ----------------------------------------------------
        elif mode == 2:

            for bit in range(width):
                circuit.cx(
                    reg_a[bit],
                    reg_b[bit],
                )

        # ----------------------------------------------------
        # AQA-3
        # ----------------------------------------------------
        elif mode == 3:

            # Generate approximate carry before modifying B.
            circuit.cx(
                reg_b[msb],
                carry,
            )

            for bit in range(width):
                cls._swap(
                    circuit,
                    reg_a[bit],
                    reg_b[bit],
                )

        # ----------------------------------------------------
        # AQA-4
        # ----------------------------------------------------
        elif mode == 4:

            # Carry is copied from the original B MSB.
            circuit.cx(
                reg_b[msb],
                carry,
            )

            for bit in range(width):
                circuit.cx(
                    reg_a[bit],
                    reg_b[bit],
                )

        # ----------------------------------------------------
        # AQA-5
        # ----------------------------------------------------
        elif mode == 5:

            # Approximate carry from the most significant
            # bit pair of the approximate region.
            circuit.ccx(
                reg_a[msb],
                reg_b[msb],
                carry,
            )

            for bit in range(width):
                circuit.cx(
                    reg_a[bit],
                    reg_b[bit],
                )

        else:
            raise ValueError(
                "Unsupported approximate-adder mode."
            )


# ============================================================
# Hybrid Quantum Adder
# ============================================================

class HybridQuantumAdder:
    """
    Build a hybrid circuit consisting of:

        LSB region -> approximate stage
        MSB region -> exact Cuccaro stage
    """

    def __init__(
        self,
        config: ExperimentConfig,
    ):
        self.config = config
        self.config.validate()

    @staticmethod
    def _load_unsigned_integer(
        circuit: QuantumCircuit,
        register: QuantumRegister,
        value: int,
    ) -> None:
        """Encode a classical unsigned integer into a register."""

        for bit in range(len(register)):
            if (value >> bit) & 1:
                circuit.x(register[bit])

    def build(
        self,
        value_a: int,
        value_b: int,
        measure: bool = True,
    ) -> QuantumCircuit:

        n = self.config.total_bits
        approx_width = self.config.approx_bits

        max_value = (1 << n) - 1

        if not 0 <= value_a <= max_value:
            raise ValueError("value_a is outside register range.")

        if not 0 <= value_b <= max_value:
            raise ValueError("value_b is outside register range.")

        # ----------------------------------------------------
        # Registers
        # ----------------------------------------------------

        reg_a = QuantumRegister(
            n,
            "A",
        )

        reg_b = QuantumRegister(
            n,
            "B",
        )

        # carry[0] : carry from approximate region
        # carry[1] : final carry output
        carry = QuantumRegister(
            2,
            "carry",
        )

        if measure:
            classical = ClassicalRegister(
                n + 1,
                "result",
            )

            circuit = QuantumCircuit(
                reg_a,
                reg_b,
                carry,
                classical,
            )

        else:
            circuit = QuantumCircuit(
                reg_a,
                reg_b,
                carry,
            )

        # ----------------------------------------------------
        # Input preparation
        # ----------------------------------------------------

        self._load_unsigned_integer(
            circuit,
            reg_a,
            value_a,
        )

        self._load_unsigned_integer(
            circuit,
            reg_b,
            value_b,
        )

        # ----------------------------------------------------
        # Approximate LSB region
        # ----------------------------------------------------

        ApproximateStage.apply(
            circuit=circuit,
            reg_a=reg_a,
            reg_b=reg_b,
            carry=carry[0],
            width=approx_width,
            mode=self.config.aqa_type,
        )

        # ----------------------------------------------------
        # Exact MSB region
        # ----------------------------------------------------

        exact_start = approx_width
        exact_end = n - 1

        CuccaroStage.apply(
            circuit=circuit,
            reg_a=reg_a,
            reg_b=reg_b,
            carry_in=carry[0],
            carry_out=carry[1],
            start=exact_start,
            end=exact_end,
        )

        # ----------------------------------------------------
        # Measurement
        # ----------------------------------------------------

        if measure:

            for bit in range(n):
                circuit.measure(
                    reg_b[bit],
                    classical[bit],
                )

            circuit.measure(
                carry[1],
                classical[n],
            )

        return circuit


# ============================================================
# Evaluation Metrics
# ============================================================

@dataclass
class EvaluationResult:
    """Container for evaluated error metrics."""

    error_rate: float
    med: float
    nmed: float
    max_error_distance: int
    input_pairs: int
    shots_per_input: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "ER": self.error_rate,
            "MED": self.med,
            "NMED": self.nmed,
            "MaxED": self.max_error_distance,
            "InputPairs": self.input_pairs,
            "ShotsPerInput": self.shots_per_input,
        }


class HybridAdderEvaluator:
    """Exhaustive evaluator for the hybrid adder."""

    # Force decomposition of CCX gates so that the
    # CX noise model is actually applied to Toffoli logic.
    TRANSPILE_BASIS = [
        "id",
        "rz",
        "sx",
        "x",
        "cx",
    ]

    def __init__(
        self,
        config: ExperimentConfig,
    ):
        self.config = config
        self.config.validate()

        self.builder = HybridQuantumAdder(
            config
        )

        noise_model = NoiseFactory.create(
            config.noise_mode
        )

        if noise_model is None:
            self.simulator = AerSimulator()
        else:
            self.simulator = AerSimulator(
                noise_model=noise_model
            )

    @staticmethod
    def decode_output(
        bit_string: str,
    ) -> int:
        """
        Convert Qiskit's classical output string
        directly to an unsigned integer.

        Classical mapping:
            result[0:n] -> B register
            result[n]   -> final carry
        """

        clean = bit_string.replace(
            " ",
            "",
        )

        return int(clean, 2)

    def _evaluate_batch(
        self,
        pairs: List[Tuple[int, int]],
    ) -> Tuple[float, float, int]:

        circuits = [
            self.builder.build(
                value_a=a,
                value_b=b,
                measure=True,
            )
            for a, b in pairs
        ]

        transpiled = transpile(
            circuits,
            basis_gates=self.TRANSPILE_BASIS,
            optimization_level=0,
        )

        result = self.simulator.run(
            transpiled,
            shots=self.config.shots,
            seed_simulator=self.config.seed,
        ).result()

        counts_collection = result.get_counts()

        if isinstance(
            counts_collection,
            dict,
        ):
            counts_collection = [
                counts_collection
            ]

        error_probability_sum = 0.0
        expected_error_distance_sum = 0.0
        max_error_distance = 0

        for (
            value_a,
            value_b,
        ), counts in zip(
            pairs,
            counts_collection,
        ):

            exact_sum = value_a + value_b

            total_shots = sum(
                counts.values()
            )

            input_error_probability = 0.0
            input_expected_ed = 0.0

            for (
                bit_string,
                frequency,
            ) in counts.items():

                approximate_sum = (
                    self.decode_output(
                        bit_string
                    )
                )

                error_distance = abs(
                    approximate_sum
                    - exact_sum
                )

                probability = (
                    frequency
                    / total_shots
                )

                if error_distance != 0:
                    input_error_probability += (
                        probability
                    )

                input_expected_ed += (
                    probability
                    * error_distance
                )

                max_error_distance = max(
                    max_error_distance,
                    error_distance,
                )

            error_probability_sum += (
                input_error_probability
            )

            expected_error_distance_sum += (
                input_expected_ed
            )

        return (
            error_probability_sum,
            expected_error_distance_sum,
            max_error_distance,
        )

    def evaluate(self) -> EvaluationResult:
        """
        Exhaustively evaluate every input combination.

        Circuits are processed in batches to avoid
        excessive memory use for larger bit widths.
        """

        n = self.config.total_bits

        input_range = range(
            1 << n
        )

        batch: List[
            Tuple[int, int]
        ] = []

        total_error_probability = 0.0
        total_expected_ed = 0.0
        global_max_ed = 0

        processed_inputs = 0

        total_inputs = (
            1 << n
        ) ** 2

        for value_a in input_range:
            for value_b in input_range:

                batch.append(
                    (
                        value_a,
                        value_b,
                    )
                )

                if (
                    len(batch)
                    >= self.config.batch_size
                ):

                    (
                        batch_error,
                        batch_med,
                        batch_max,
                    ) = self._evaluate_batch(
                        batch
                    )

                    total_error_probability += (
                        batch_error
                    )

                    total_expected_ed += (
                        batch_med
                    )

                    global_max_ed = max(
                        global_max_ed,
                        batch_max,
                    )

                    processed_inputs += len(
                        batch
                    )

                    print(
                        f"\rProcessed "
                        f"{processed_inputs:,}"
                        f"/{total_inputs:,} "
                        f"input pairs",
                        end="",
                    )

                    batch = []

        # Remaining circuits.
        if batch:

            (
                batch_error,
                batch_med,
                batch_max,
            ) = self._evaluate_batch(
                batch
            )

            total_error_probability += (
                batch_error
            )

            total_expected_ed += (
                batch_med
            )

            global_max_ed = max(
                global_max_ed,
                batch_max,
            )

            processed_inputs += len(
                batch
            )

        print()

        error_rate = (
            total_error_probability
            / total_inputs
        )

        med = (
            total_expected_ed
            / total_inputs
        )

        # Same normalization convention used by
        # the original demonstration code.
        normalization_factor = (
            (1 << n) - 1
        )

        nmed = (
            med
            / normalization_factor
        )

        return EvaluationResult(
            error_rate=error_rate,
            med=med,
            nmed=nmed,
            max_error_distance=global_max_ed,
            input_pairs=total_inputs,
            shots_per_input=self.config.shots,
        )


# ============================================================
# Circuit Information
# ============================================================

def print_circuit_summary(
    config: ExperimentConfig,
) -> None:
    """
    Print structural information for one representative circuit.
    """

    builder = HybridQuantumAdder(
        config
    )

    circuit = builder.build(
        value_a=0,
        value_b=0,
        measure=False,
    )

    decomposed = transpile(
        circuit,
        basis_gates=HybridAdderEvaluator.TRANSPILE_BASIS,
        optimization_level=0,
    )

    print("\n===== Representative Circuit =====")
    print(
        f"Logical qubits : "
        f"{circuit.num_qubits}"
    )

    print(
        f"Original depth : "
        f"{circuit.depth()}"
    )

    print(
        f"Basis depth    : "
        f"{decomposed.depth()}"
    )

    print(
        f"Basis gates    : "
        f"{dict(decomposed.count_ops())}"
    )


# ============================================================
# CLI
# ============================================================

def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Hybrid Approximate Quantum "
            "Adder portfolio demonstration"
        )
    )

    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Total operand bit width (default: 4)",
    )

    parser.add_argument(
        "--approx-bits",
        type=int,
        default=2,
        help=(
            "Number of approximate "
            "least-significant bits "
            "(default: 2)"
        ),
    )

    parser.add_argument(
        "--aqa-type",
        type=int,
        choices=[
            1,
            2,
            3,
            4,
            5,
        ],
        default=2,
        help=(
            "Approximate stage type "
            "1-5 (default: 2)"
        ),
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=100,
        help=(
            "Simulation shots per "
            "input pair (default: 100)"
        ),
    )

    parser.add_argument(
        "--noise",
        choices=[
            "ideal",
            "thermal",
            "depolarizing",
            "bitflip",
        ],
        default="ideal",
        help="Noise environment",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help=(
            "Number of circuits processed "
            "per simulator batch "
            "(default: 256)"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Aer simulator seed",
    )

    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = build_argument_parser()
    args = parser.parse_args()

    config = ExperimentConfig(
        total_bits=args.bits,
        approx_bits=args.approx_bits,
        aqa_type=args.aqa_type,
        shots=args.shots,
        noise_mode=args.noise,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    config.validate()

    print(
        "\n=============================================="
    )
    print(
        " Hybrid Approximate Quantum Adder Demo"
    )
    print(
        "=============================================="
    )

    print(
        f"Total width       : "
        f"{config.total_bits}-bit"
    )

    print(
        f"Approximate region: "
        f"{config.approx_bits}-bit"
    )

    print(
        f"Exact region      : "
        f"{config.total_bits - config.approx_bits}-bit"
    )

    print(
        f"AQA type          : "
        f"{config.aqa_type}"
    )

    print(
        f"Noise model       : "
        f"{config.noise_mode}"
    )

    print(
        f"Shots/input       : "
        f"{config.shots}"
    )

    print_circuit_summary(
        config
    )

    print(
        "\n===== Exhaustive Evaluation ====="
    )

    evaluator = HybridAdderEvaluator(
        config
    )

    result = evaluator.evaluate()

    print(
        "\n=============== Results ==============="
    )

    print(
        f"Error Rate : "
        f"{result.error_rate:.8f}"
    )

    print(
        f"MED        : "
        f"{result.med:.8f}"
    )

    print(
        f"NMED       : "
        f"{result.nmed:.8f}"
    )

    print(
        f"Max ED     : "
        f"{result.max_error_distance}"
    )

    print(
        f"Input pairs: "
        f"{result.input_pairs:,}"
    )

    print(
        f"Shots/input: "
        f"{result.shots_per_input}"
    )

    print(
        "======================================="
    )


if __name__ == "__main__":
    main()
