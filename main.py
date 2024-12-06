import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Carregar o dataset
file_path = '../DepressionStudentDataset.csv'
data = pd.read_csv(file_path)

# Transformar colunas categóricas para numéricas
label_encoders = {}
for col in ['Gender', 'Sleep Duration', 'Dietary Habits', 
            'Have you ever had suicidal thoughts ?', 
            'Family History of Mental Illness', 'Depression']:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])
    label_encoders[col] = encoder

# Separar X (features) e y (target)
X = data.drop(columns=['Depression'])
y = data['Depression']

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Rede Neural
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.01):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
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
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)

        self.weights_hidden_output += np.dot(self.hidden.T, output_delta) * self.learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * self.learning_rate

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:  # progresso
                loss = np.mean((y - output) ** 2)
                print(f"Época {epoch}, Loss: {loss:.4f}")

    def predict_probability(self, X):
        output = self.forward(X)
        return output  # Probabilidade de ter depressão

# Configuração da rede neural
input_size = X_train.shape[1]
hidden_size = 20
output_size = 1
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.01)

# Ajustar formato do target
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

# Treinamento
nn.train(X_train, y_train, epochs=1000)

# Fazer previsões no conjunto de teste
y_test_pred = nn.predict_probability(X_test)
y_test_pred_binary = (y_test_pred > 0.5).astype(int)  # Converter para binário

# Calcular as métricas
precision = precision_score(y_test, y_test_pred_binary)
recall = recall_score(y_test, y_test_pred_binary)
f1 = f1_score(y_test, y_test_pred_binary)
accuracy = accuracy_score(y_test, y_test_pred_binary)

# Exibir os resultados
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Acurácia: {accuracy:.2f}")


# Entrada do usuário
def get_user_input():
    user_data = {}
    for col in ['Gender', 'Age', 'Academic Pressure', 'Study Satisfaction', 
                'Sleep Duration', 'Dietary Habits', 
                'Have you ever had suicidal thoughts ?', 
                'Study Hours', 'Financial Stress', 'Family History of Mental Illness']:
        
        if col in label_encoders:  # Para colunas categóricas
            # Obter as classes e exibir com opções numéricas
            classes = list(label_encoders[col].classes_)
            print(f"\nSelecione o valor para '{col}':")
            for i, option in enumerate(classes):
                print(f"{i}: {option}")
            
            while True:
                try:
                    value = int(input("Digite o número correspondente: "))
                    if 0 <= value < len(classes):  # Validar a escolha
                        user_data[col] = value
                        break
                    else:
                        print("Opção inválida. Escolha novamente.")
                except ValueError:
                    print("Entrada inválida. Digite um número.")
        else: 
            while True:
                try:
                    value = int(input(f"\nInsira o valor para '{col}': "))
                    user_data[col] = value
                    break
                except ValueError:
                    print("Entrada inválida. Digite um número.")
                    
    return user_data

# input
print("Forneça as informações para prever a probabilidade de depressão.")
user_input = get_user_input()
# Converter para DataFrame
user_df = pd.DataFrame([user_input])
# Normalizar a entrada do usuário
user_scaled = scaler.transform(user_df)

probability = nn.predict_probability(user_scaled)
print(f"Probabilidade de depressão: {probability[0][0] * 100:.1f}%")


