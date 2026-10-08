import numpy as np


def resolve_lu(A, b):
  """Resolve o sistema linear Ax = b utilizando a decomposição LU (A = LU).

  L é triangular inferior com 1s na diagonal principal.
  U é triangular superior.
  """
  # Garantir que A e b são arrays do numpy
  A = np.array(A, dtype=float)
  b = np.array(b, dtype=float)

  n = A.shape[0]

  # Inicializa L como matriz identidade e U como cópia de A
  L = np.eye(n)
  U = A.copy()

  # 1. Decomposição LU (sem pivoteamento)
  for i in range(n):
    # Verifica se o pivô é nulo (ou muito próximo de zero)
    if np.isclose(U[i, i], 0.0):
      raise Exception(
          "Pivô nulo encontrado. Utilize uma função alternativa para a solução"
          " do sistema."
      )

    for j in range(i + 1, n):
      # Calcula o multiplicador
      fator = U[j, i] / U[i, i]
      L[j, i] = fator
      # Atualiza a linha j de U
      U[j, i:] = U[j, i:] - fator * U[i, i:]

  # 2. Resolução de Ly = b por Substituição Progressiva
  y = np.zeros(n)
  for i in range(n):
    soma = sum(L[i, j] * y[j] for j in range(i))
    y[i] = (b[i] - soma) / L[i, i]

  # 3. Resolução de Ux = y por Substituição Regressiva
  x = np.zeros(n)
  for i in range(n - 1, -1, -1):
    soma = sum(U[i, j] * x[j] for j in range(i + 1, n))
    x[i] = (y[i] - soma) / U[i, i]

  # 4. Retorna as matrizes L, U e o vetor solução x
  return L, U, x
