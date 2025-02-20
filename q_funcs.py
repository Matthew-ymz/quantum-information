import numpy as np
from functools import reduce
from itertools import product
global sigma_x, sigma_y, sigma_z, identity, pauli_basis
 # 定义泡利矩阵
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity = np.array([[1, 0], [0, 1]], dtype=complex)
pauli_basis = [
    identity,        # σ0 = I
    sigma_x,        # σ1 = X
    sigma_y,     # σ2 = Y
    sigma_z        # σ3 = Z
]

def expand_kraus(K, total_qubits, targets):
    """
    将Kraus算子扩展到n量子比特系统，并作用于指定目标比特
    :param K: 输入的Kraus算子 (2^m × 2^m矩阵，m为目标比特数)
    :param total_qubits: 总量子比特数
    :param targets: 目标比特的位置列表 (必须已排序)
    """
    m = len(targets)
    assert K.shape == (2**m, 2**m), "K维度与目标比特数不符"
    
    # 生成非目标比特的索引
    non_targets = [i for i in range(total_qubits) if i not in targets]
    
    # 构造排列后的量子比特顺序: [目标比特..., 非目标比特...]
    perm = targets + non_targets
    
    # 构造扩展后的算子 (在排列后的顺序中)
    dim_non_target = 2**(total_qubits - m)
    expanded = np.kron(K, np.eye(dim_non_target, dtype=complex))
    
    # 将算子转换为张量格式 [输入轴... + 输出轴...]
    tensor = expanded.reshape([2]*(2*total_qubits))
    
    # 计算逆排列（输入和输出轴分开处理）
    inv_perm_input = list(np.argsort(perm))  # 输入轴逆排列
    inv_perm_output = [i + total_qubits for i in inv_perm_input]  # 输出轴逆排列
    
    # 合并轴排列顺序
    new_axes = inv_perm_input + inv_perm_output
    
    # 调整轴顺序并恢复矩阵形状
    tensor = tensor.transpose(new_axes)
    return tensor.reshape((2**total_qubits, 2**total_qubits))

def quantum_EI(K_ls, total_qubits):
    """
    计算量子互信息（EI）
    :param K_ls: Kraus算子列表
    :param total_qubits: 总量子比特数
    """
    # 生成n对贝尔态的直积态
    m = total_qubits // 2
    assert total_qubits % 2 == 0, "总比特数必须为偶数"
    targets = [i for i in range(0, 2 * m, 2)]
    # 构造贝尔态直积态
    bell_state = np.array([1, 0, 0, 1], dtype=complex)/np.sqrt(2)
    state = reduce(np.kron, [bell_state]*m)
    
    # 构造密度矩阵
    rho = np.outer(state, state.conj())
    
    # 应用Kraus算子
    rho_new = sum(expand_kraus(K, total_qubits, targets) @ rho @ 
                  expand_kraus(K, total_qubits, targets).conj().T
                  for K in K_ls)
    
    # 计算部分迹
    rho_a = partial_trace(rho_new, total_qubits, keep=targets)
    rho_b = partial_trace(rho_new, total_qubits, 
                         keep=[i for i in range(total_qubits) if i not in targets])
    
    return quantum_mutual_information(rho_new, rho_a, rho_b)

def partial_trace(rho, n, keep):
    """
    计算n量子比特系统的部分迹
    :param rho: 联合密度矩阵，形状为(2^n, 2^n)
    :param n: 总量子比特数
    :param keep: 保留的量子比特的索引列表（如[0,1]表示保留前两个）
    :return: 保留子系统后的密度矩阵
    """
    dim = 2**n
    if rho.shape != (dim, dim):
        raise ValueError("输入的密度矩阵维度错误")
    keep = sorted(keep)
    trace_over = [i for i in range(n) if i not in keep]
    perm = keep + trace_over + [k + n for k in keep] + [k + n for k in trace_over]
    rho = rho.reshape([2] * 2 * n)
    rho = np.transpose(rho, perm).reshape(
        (2**len(keep), 2**len(trace_over), 2**len(keep), 2**len(trace_over))
    )
    return np.einsum('ijik->jk', rho)

def quantum_mutual_information(rho_ab, rho_a, rho_b, threshold=1e-10):
    """
    计算量子互信息
    :param rho_ab: 联合密度矩阵
    :param rho_a: 子系统A的密度矩阵
    :param rho_b: 子系统B的密度矩阵
    """
    def entropy(rho):
        eigvals = np.linalg.eigvalsh(rho + threshold * np.eye(rho.shape[0]))
        return -np.sum(eigvals * np.log2(eigvals, where=eigvals > 0))
    return entropy(rho_a) + entropy(rho_b) - entropy(rho_ab)

def tpm_ei(tpm, log_base = 2):
    # marginal distribution of y given x ~ Unifrom Dist
    puy = tpm.sum(axis=0)
    n = tpm.shape[0]
    # replace 0 to a small positive number to avoid log error
    eps = 1E-10
    tpm_e = np.where(tpm==0, eps, tpm)
    puy_e = np.where(puy==0, eps, puy)
    
    # calculate EI of specific x
    ei_x = (np.log2(n * tpm_e / puy_e) / np.log2(log_base)  * tpm).sum(axis=1)
    
    # calculate total EI
    ei_all = ei_x.mean()
    return ei_all

def kraus_to_transfer_matrix(kraus_ops):
    """
    根据Kraus算子计算经典转移矩阵 P，其中 P[j][i] = Σ |⟨j|K_k|i⟩|²
    
    参数:
        kraus_ops (list of np.ndarray): Kraus算子列表，每个元素为 d×d 的复数矩阵
    
    返回:
        P (np.ndarray): d×d 的实数矩阵，元素为经典转移概率
    """
    # 获取系统维度 d
    d = kraus_ops[0].shape[0]
    # 初始化转移矩阵
    P = np.zeros((d, d), dtype=float)
    
    # 遍历所有输入态 i 和输出态 j
    for i in range(d):
        e_i = np.eye(d)[:, i]  # 输入态 |i⟩ 的向量表示
        for j in range(d):
            e_j = np.eye(d)[:, j]  # 输出态 |j⟩ 的向量表示
            prob = 0.0
            # 对每个 Kraus 算子 K_k 计算贡献
            for K in kraus_ops:
                # 计算 K|i⟩
                K_e_i = K @ e_i
                # 计算 ⟨j|K|i⟩ 的模平方
                inner = np.vdot(e_j, K_e_i)  # 等价于 e_j^† @ K_e_i
                prob += np.abs(inner)**2#np.dot(inner, inner.T.conj())
            P[j, i] = prob
    return P

def generate_kraus_operators(dimension, num_operators):
    dims = num_operators * dimension
    A = np.random.randn(dims, dims) + 1j * np.random.randn(dims, dims)
    
    # 奇异值分解
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    K = U[:dimension,:].reshape(dimension, dimension, num_operators)
    
    return K

def generate_bell_state():
    # 定义单比特 Hadamard 门
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    I = np.eye(2, dtype=complex)
    # 构造两比特幺正操作 U = CNOT · (H⊗I)
    H_tensor_I = np.kron(H, I)
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    U = CNOT @ H_tensor_I  # 组合操作

    # Kraus算子列表（此处为单一幺正操作）
    kraus_ops = [U]
    return kraus_ops

def generate_depolarizing_kraus_operators(p):
    """
    生成去极化信道的 Kraus 算子。
    
    参数:
    - p: 去极化信道的噪声参数 (0 <= p <= 1)
    
    返回:
    - 一组 Kraus 算子列表
    """
    # 定义 Pauli 矩阵
    I = np.eye(2)  # 恒等矩阵
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # Pauli X 矩阵
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)  # Pauli Y 矩阵
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)  # Pauli Z 矩阵

    # 计算 Kraus 算子
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p / 3) * X
    K2 = np.sqrt(p / 3) * Y
    K3 = np.sqrt(p / 3) * Z

    return [K0, K1, K2, K3]

def generate_pauli_basis(n):
    """生成n量子比特的泡利基矩阵（包含单位矩阵）"""
    
    # 生成所有可能的张量积组合
    basis = []
    for indices in product(range(4), repeat=n):
        mat = 1  # 初始化为单位标量
        for idx in indices:
            mat = np.kron(mat, pauli_basis[idx])
        basis.append(mat)
    return basis

def compute_M_matrix_n_qubit(K_list):
    """计算n量子比特信道的泡利基线性变换矩阵M"""
    n = int(np.log2(K_list[0].shape[0]))  # 从Kraus算子维度推断量子比特数
    pauli_basis_n = generate_pauli_basis(n)
    dim = 4**n
    M = np.zeros((dim, dim), dtype=complex)
    
    for i in range(dim):
        sigma_i = pauli_basis_n[i]
        for j in range(dim):
            sigma_j = pauli_basis_n[j]
            
            # 计算 ∑ K_k σ_i K_k^†
            sum_term = np.zeros((2**n, 2**n), dtype=complex)
            for K in K_list:
                term = K @ sigma_i @ K.conj().T
                sum_term += term
            
            # 计算迹并存储结果
            M[j, i] = (1/(2**n)) * np.trace(sigma_j @ sum_term)
    
    return np.real(M)

# def compute_M_matrix(K_list):
#     """计算量子信道的泡利基线性变换矩阵M"""
#     M = np.zeros((4, 4), dtype=complex)
    
#     for i in range(4):  # σ_i索引
#         sigma_i = pauli_basis[i]
#         for j in range(4):  # σ_j索引
#             sigma_j = pauli_basis[j]
            
#             # 计算 ∑ K_k σ_i K_k^†
#             sum_term = np.zeros((2, 2), dtype=complex)
#             for K in K_list:
#                 term = K @ sigma_i @ K.conj().T
#                 sum_term += term
                
#             # 计算迹并存储结果
#             M[j, i] = 0.5 * np.trace(sigma_j @ sum_term)
    
#     # 确保结果为实数（根据量子信道性质）
#     return np.real(M)