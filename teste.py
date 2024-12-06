import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import random

file_path = '../DepressionStudentDataset.csv' 
data = pd.read_csv(file_path)

# Armazena os LabelEncoders de cada coluna
label_encoders = {}

# Transformar as colunas usando um LabelEncoder para cada uma
for col in ['Gender', 'Sleep Duration', 'Dietary Habits', 
            'Have you ever had suicidal thoughts ?', 
            'Family History of Mental Illness', 'Depression']:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    label_encoders[col] = encoder

X = data.drop(columns=['Depression'])
y = data['Depression']

# Normalizar características numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir treino e teste 80 / 20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)  # Pesos entre entrada e camada oculta
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)  # Pesos entre camada oculta e saída
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):   
        self.hidden = self.sigmoid(np.dot(X, self.weights_input_hidden))
        self.output = self.sigmoid(np.dot(self.hidden, self.weights_hidden_output))
        return self.output

    def backward(self, X, y, output):
        output_error = y - output  # Erro na saída
        output_delta = output_error * self.sigmoid_derivative(output)  # Ajuste da saída

        hidden_error = output_delta.dot(self.weights_hidden_output.T)  # Erro na camada oculta
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)  # Ajuste da camada oculta

        # Atualizar pesos
        self.weights_hidden_output += self.hidden.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate

    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)  # Retornar o índice da classe com maior probabilidade

# Busca Aleatória de Hiperparâmetros
results = []
hidden_sizes = [5, 10, 20, 50]  # Tamanhos possíveis para a camada oculta
learning_rates = [0.001, 0.01, 0.1, 0.05]  # Taxas de aprendizado possíveis

for _ in range(10):  # Realiza 10 iterações de busca aleatória
    hidden_size = random.choice(hidden_sizes)
    learning_rate = random.choice(learning_rates)

    print(f"\nTestando hidden_size={hidden_size}, learning_rate={learning_rate}")
    
    # Criar e treinar a rede neural
    nn = NeuralNetwork(X_train.shape[1], hidden_size, output_size=2, learning_rate=learning_rate)
    nn.train(X_train, y_train_encoded, epochs=1000)
    
    # Fazer previsões e calcular a acurácia
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == y_test.values)
    results.append((hidden_size, learning_rate, accuracy))

# Exibir os resultados
print("\nResultados da Busca Aleatória de Hiperparâmetros:")
for hidden_size, learning_rate, accuracy in results:
    print(f"hidden_size={hidden_size}, learning_rate={learning_rate}, Acurácia={accuracy:.4f}")

# Organizar os resultados para visualização
results_df = pd.DataFrame(results, columns=['hidden_size', 'learning_rate', 'accuracy'])


plt.figure(figsize=(10, 6))

# Acurácia por tamanho da camada oculta e taxa de aprendizado
plt.subplot(1, 2, 1)
for hidden_size in hidden_sizes:
    subset = results_df[results_df['hidden_size'] == hidden_size]
    plt.plot(subset['learning_rate'], subset['accuracy'], label=f'Hidden Size {hidden_size}', marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Acurácia')
plt.title('Acurácia por Tamanho da Camada Oculta e Taxa de Aprendizado')
plt.legend()

plt.tight_layout()
plt.show()

# melhor combinação de hiperparâmetros
best_result = max(results, key=lambda x: x[2])
print(f"\nMelhor resultado: hidden_size={best_result[0]}, learning_rate={best_result[1]}, Acurácia={best_result[2]:.4f}")
