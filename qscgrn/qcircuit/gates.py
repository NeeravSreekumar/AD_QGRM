import numpy as np

__all__ = ["ry_gate", "der_ry_gate", "cnot_gate", "cry_gate"]


def ry_gate(theta):
    """
    Computes the matrix transformation for the `Ry` gate with an
    angle parameter of `theta`.
    `T = Ry(theta)`.
    Parameters
    ----------
    theta : float or int
        The rotation angle in radians. In q-sphere, the rotation angle
        around the y-axis.
    Returns
    -------
    T : ndarray
    Raises
    ------
    TypeError
        If the input is not a float or integer.
    """
    if not isinstance(theta, (float, int)):
        raise TypeError("Ry gate operation unsupported for {type}"
                        .format(type=type(theta)))

    half_theta = theta / 2
    sin = np.sin(half_theta)
    cos = np.cos(half_theta)
    return np.array([[cos, -sin],
                     [sin, cos]])


def der_ry_gate(theta):
    """
    Computes the matrix transformation for the derivative of `Ry` gate
    with an angle parameter of `theta`.
    `dT = dRy(theta) dtheta `.
    Parameters
    ----------
    theta : float or int
        The rotation angle in radians. In q-sphere, the rotation angle
        around the y-axis.
    Returns
    -------
    dT : ndarray
    Raises
    ------
    TypeError
        If the input is not a float or integer.
    """
    if not isinstance(theta, (float, int)):
        raise TypeError("dRy gate operation unsupported for {type}"
                        .format(type=type(theta)))

    half_theta = theta / 2
    sin = np.sin(half_theta)
    cos = np.cos(half_theta)
    return 0.5 * np.array([[-sin, -cos],
                           [cos, -sin]])


def cnot_gate(control, target):
    """
    Computes the matrix transformation for the `CNOT` gate.
    `T = CNOT`.
    Parameters
    ----------
    control : int
        The control qubit index.
    target : int
        The target qubit index.
    Returns
    -------
    T : ndarray
        The matrix representation of the CNOT gate.
    """
    T = np.zeros((4, 4))
    T[0, 0] = 1
    T[1, 1] = 1
    T[2, 3] = 1
    T[3, 2] = 1 if control == 0 and target == 1 else 0  # Apply NOT if control qubit is 1
    return T


def cry_gate(theta, nqubits, control, target):
    """
    Computes the matrix transformation for the `c-Ry` gate with an
    angle parameter of `theta`.
    `T = c-Ry(theta)`.

    Parameters
    ----------
    theta : float or int
        The rotation angle in radians. In q-sphere, the rotation angle
        around the y-axis.
    nqubits : int
        The number of qubits in the quantum circuit. Similar
        to the number of genes for modelling.
    control : int
        The control qubit, which activates or not the `Ry` gate in the
        target qubit.
    target : int
        The target qubit, where the `Ry` gate will be applied.
    Returns
    -------
    T : ndarray
        The matrix representation of the c-Ry gate.
    Raises
    ------
    TypeError
        If the input theta is not a float or integer.
    """
    if not isinstance(theta, (float, int)):
        raise TypeError("c-Ry gate operation unsupported for {type}"
                        .format(type=type(theta)))

    # Initialize the gate matrix with identity
    gate_matrix = np.identity(2**nqubits)

    # Apply the decomposition
    gate_matrix = np.dot(cnot_gate(control, target), gate_matrix)
    gate_matrix = np.dot(ry_gate(theta / 2), gate_matrix)
    gate_matrix = np.dot(cnot_gate(control, target), gate_matrix)
    gate_matrix = np.dot(ry_gate(-theta / 2), gate_matrix)

    return gate_matrix
