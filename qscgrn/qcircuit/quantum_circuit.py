import numpy as np
import pandas as pd
from .utils import *
from .gates import *
from ..run import *
from ..utils import info_print
from qscgrn.qcircuit import cnot_gate, ry_gate


__all__ = ['quantum_circuit']


class quantum_circuit(qscgrn_model):
    """
    Attributes
    ----------
    ngenes : int
        Number of genes in the Quantum GRN.
    genes : list
        Gene list for Quantum GRN modeling.
    theta : pd.Series
        Theta values given edges in the Quantum GRN.
    edges : list
        Edges for the Quantum GRN.
    indexes : list
        Numerical values of the edges for usage in the quantum circuit.
    encoder : np.array
        Matrix representation of the encoder layer `L_enc`.
    regulation : np.array
        Matrix representation of the regulation layers `L_k`. The
        encoder layers are grouped in a single array.
    circuit : np.array
        A single matrix representation for the quantum circuit
        transformation.
    input : np.array
        The quantum state as input for the quantum circuit model.
    derivatives : pd.DataFrame
        Derivatives of the quantum state in the output register
        with respect to each parameter.
    drop_zero : bool
        If True, a normalization step for `p^out` that sets the `|0>_n`
        state to 0, and rescales the rest of the distribution.

    Methods
    -------
    compute_encoder()
        Computes the transformation matrix for the `L_enc` into the
        encoder attribute.
    compute_regulation()
        Computes the transformation matrix for the `L_k` into the
        regulation attribute.
    generate_circuit()
        Compute the transformation matrix for the `L_enc` and `L_k`.
    transform_matrix()
        Compute the transformation matrix for the entire quantum
        circuit.
    output_state()
        Compute the quantum state in the output register given an
        input state.
    output_probabilities(drop_zeros)
        Compute the probability distribution in the output register.
        If drop_zero is True, a normalization step is done.
    create_derivatives()
        Creates a pd.DataFrame to store the derivatives of the
        output state with respect to the parameters.
    der_encoder()
        Computes the derivatives with respect to the parameters
        in the `L_enc` layer.
    der_regulation()
        Computes the derivatives with respect to the parameters
        in the `L_k` layers.
    compute_derivatives()
        Computes the derivatives by calling the der_encoder
        and the der_regulation methods.
    """

    def __init__(self, genes, theta, edges, drop_zero=True):
        """
        Parameters
        ----------
        genes : list
            Gene list for Quantum GRN modeling.
        theta : pd.Series
            Theta values given edges in the Quantum GRN.
        edges : list
            Edges for the Quantum GRN.
        drop_zero : bool
            If True, a normalization step for `p^out` that sets the
            `|0>_n` state to 0, and rescales the rest of the
            distribution.
        """
        super().__init__(genes, theta, edges, drop_zero)
        # numerical indexes are needed to construct the circuit
        self.indexes = edges_to_index(genes, edges)
        # array storage for the quantum circuit (Lenc, Lk and
        # and transformation matrix)
        self.encoder = None
        self.regulation = None
        self.circuit = False
        # parameters for quantum circuit such as input state,
        # derivatives and drop_zero
        self.input = np.zeros((2**self.ngenes, 1))
        self.input[0, 0] = 1.
        self.derivatives = None

    def __str__(self):
        return ("Quantum circuit for {ngenes} genes for GRN"
                " modeling".format(ngenes=len(self.genes)))

    def _circuit_is_empty(self):
        """
        Validates whether the quantum circuit is initialized or not.
        Raises
        ------
        AttributeError
            If circuit attribute is a None object.
        """
        if not self.circuit:
            info_print("The Quantum GRN model is not initialized")
            raise AttributeError("The quantum circuit for GRN model "
                                 "is not constructed")

    def _der_is_not_empty(self):
        """
        Validates if the derivatives for the quantum circuit are
        not initialized.
        Raises
        ------
        AttributeError
            If derivatives is not a None object
        """
        if self.derivatives is not None:
            info_print("Derivatives for the Quantum GRN are already "
                        "initialized", level="E")
            raise AttributeError("The quantum circuit for GRN model "
                                 "has derivatives initialized")

    def _der_is_empty(self):
        """
        Validates if the derivatives for the quantum circuit are
        initialized
        Raises
        ------
        AttributeError
            If derivatives is a None object
        """
        if self.derivatives is None:
            info_print("Derivatives for the Quantum GRN are not "
                        "initialized", level="E")
            raise AttributeError("The quantum circuit for GRN model "
                                 "does not have derivatives "
                                 "initialized")

    def compute_encoder(self):
        """
        Computes the transformation matrices of each gate in `L_enc`
        layer and saves the result into self.encoder
        """
        RR = np.zeros((len(self.genes), 2, 2))

        for idx, gene in enumerate(self.genes):
            RR[idx] = ry_gate(self.theta[(gene, gene)])

        self.encoder = RR

    def compute_regulation(self):
        """
        Computes the transformation matrices of each gate in `L_k`
        layer and saves the result into self.regulation
        """
        arr = np.zeros((len(self.edges), 2**self.ngenes, 2**self.ngenes))

        for i, edge in enumerate(self.edges):
            idx = self.indexes[i]
            control, target = idx[0], idx[1]
            theta_edge = self.theta[edge]

            # Decompose the controlled-Ry gate into CNOT and rotation gates
            cnot = cnot_gate(control, target)
            ry = ry_gate(theta_edge)

            arr[i] = np.dot(cnot, np.dot(ry, cnot))

        self.regulation = arr

    def generate_circuit(self):
        """
        Generates the entire quantum circuit by computing the
        transformation matrix for the encoder and regulation layers.
        """
        self._der_is_empty()
        self.compute_encoder()
        self.compute_regulation()
        self.circuit = self.transform_matrix()

    def transform_matrix(self):
        """
        Computes the transformation matrix for the entire quantum
        circuit by multiplying the encoder and regulation matrices.
        Returns
        -------
        np.ndarray
            The transformation matrix representing the quantum circuit.
        """
        self._circuit_is_empty()

        if self.encoder.shape[0] == 1:
            return self.regulation[0]
        else:
            regulation_matrix = np.eye(2**self.ngenes)

            for R in self.regulation:
                regulation_matrix = np.dot(R, regulation_matrix)

            return regulation_matrix

    def output_state(self):
        """
        Computes the quantum state in the output register given an
        input state.
        Returns
        -------
        np.ndarray
            The quantum state in the output register.
        """
        self._circuit_is_empty()
        return np.dot(self.circuit, self.input)

    def output_probabilities(self, drop_zeros=False):
        """
        Computes the probability distribution in the output register.
        If `drop_zeros` is True, a normalization step is performed
        to set the `|0>_n` state to 0, and rescale the rest of the
        distribution.
        Parameters
        ----------
        drop_zeros : bool, optional
            If True, a normalization step is performed.
        Returns
        -------
        np.ndarray
            The probability distribution in the output register.
        """
        self._circuit_is_empty()
        p_out = np.abs(self.output_state()) ** 2

        if drop_zeros:
            p_out /= np.sum(p_out[1:])

        return p_out

    def create_derivatives(self):
        """
        Creates a DataFrame to store the derivatives of the
        output state with respect to each parameter.
        """
        self._der_is_not_empty()
        self.derivatives = pd.DataFrame(columns=self.theta.index,
                                         index=np.arange(2**self.ngenes))
        self.derivatives.iloc[:, :] = 0

    def der_encoder(self):
        """
        Computes the derivatives with respect to the parameters
        in the `L_enc` layer.
        """
        self._der_is_empty()
        der_R = np.zeros((len(self.genes), 2, 2))

        for idx, gene in enumerate(self.genes):
            der_R[idx] = der_ry_gate(self.theta[(gene, gene)])

        self.der_encoder = der_R

    def der_regulation(self):
        """
        Computes the derivatives with respect to the parameters
        in the `L_k` layers.
        """
        self._der_is_empty()
        der_R = np.zeros((len(self.edges), 2**self.ngenes, 2**self.ngenes))

        for i, edge in enumerate(self.edges):
            idx = self.indexes[i]
            control, target = idx[0], idx[1]
            theta_edge = self.theta[edge]

            cnot = cnot_gate(control, target)
            der_ry = der_ry_gate(theta_edge)

            der_R[i] = np.dot(cnot, np.dot(der_ry, cnot))

        self.der_regulation = der_R

    def compute_derivatives(self):
        """
        Computes the derivatives by calling the der_encoder
        and the der_regulation methods.
        """
        self._der_is_empty()
        self.der_encoder()
        self.der_regulation()

        for i, gene in enumerate(self.genes):
            for j, edge in enumerate(self.edges):
                self.derivatives[gene] += np.dot(
                    np.dot(self.regulation[j], self.der_encoder[i]),
                    self.input.reshape(-1))

        for i, edge in enumerate(self.edges):
            self.derivatives[edge] += np.dot(
                np.dot(self.der_regulation[i], self.encoder),
                self.input.reshape(-1))

