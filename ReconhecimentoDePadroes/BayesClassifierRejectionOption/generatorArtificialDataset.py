import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gerar dados para a classe 0
np.random.seed(42)  # Para reprodutibilidade
class_0_a = np.random.normal(loc=[0, 0], scale=0.1, size=(17, 2))
class_0_b = np.random.normal(loc=[1, 0], scale=0.1, size=(17, 2))
class_0_c = np.random.normal(loc=[0, 1], scale=0.1, size=(16, 2))

# Concatenar os três grupos para a classe 0
class_0 = np.vstack([class_0_a, class_0_b, class_0_c])

# Gerar dados para a classe 1
class_1 = np.random.normal(loc=[1, 1], scale=0.1, size=(50, 2))

# Criar rótulos
labels_class_0 = np.zeros(50)
labels_class_1 = np.ones(50)

# Concatenar os dados e os rótulos
X = np.vstack([class_0, class_1])
y = np.hstack([labels_class_0, labels_class_1])

# Criar DataFrame
df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
df['class'] = y

df.to_csv("artificial.data", index=False, header=False)
df.to_excel("artificial.xlsx", index=False)

# Mostrar o DataFrame gerado
print(df.head(10))

# Plotar os dados
plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Classe 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Classe 1')
plt.title('Dados Gerados')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
