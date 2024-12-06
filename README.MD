## Análise de Probabilidade de Depressão com Rede Neural

Este projeto implementa uma rede neural simples para prever a probabilidade de depressão com base em dados coletados de um dataset sobre depressão estudantil. O objetivo é criar um modelo funcional e simular uma interação de análise com o usuário.

O projeto utiliza uma rede neural para treinar um modelo capaz de classificar a probabilidade de depressão com base em características como:
- Gender
- Age
- Academic Pressure
- Study Satisfaction
- Sleep Duration
- Dietary Habits
- Have you ever had suicidal thoughts?
- Study Hours
- Financial Stress
- Family History of Mental Illness

O modelo de Rede neural escolhido foi o modelo feedforward, pela simplicidade e versatilidade do modelo, que não exige uma estrutura altamente complexa. Ótimo para modelos simples.

O dataset foi selecionado na plataforma [Kaggle](https://www.kaggle.com/datasets), uma empresa de ciência de dados que disponibiliza datasets pré-treinados. O dataset em questão é o [DepressionStudent](https://www.kaggle.com/datasets/ikynahidwin/depression-student-dataset/data), que conta com mais de 500 registros, com várias características que são usadas posteriormente para a classificação da ausência ou presença de depressão.

### O projeto inclui:

- Treinamento do modelo com um dataset pré-processado.
- Normalização e codificação de dados categóricos.
- Simulação de um "diagnóstico".
- `teste.py` utilizado para testes.
- `main.py` arquivo principal com o treinamento e o diagnóstico.

### Tecnologias Utilizadas:

Para a rede neural, foi utilizada a biblioteca **scikit-learn**, uma biblioteca open-source para aprendizado de máquina em Python. Ela fornece uma ampla gama de ferramentas eficientes para análise de dados, modelagem preditiva, entre outros.

Foi utilizado o **LabelEncoder** do `sklearn.preprocessing` para transformar colunas categóricas em numéricas. No projeto, o `teste.py` foi utilizado o **one-hot encoding** para transformar variáveis categóricas em uma representação binária, pois no treinamento só era retornado 0 ou 1.

### Etapas do Projeto:

1. **Treinamento e Validação**: A primeira etapa foi o treinamento e validação (80/20) com os dados do dataset. O modelo retornava apenas a classe (1 para depressão e 0 para sem depressão).
2. **Testes com Hiperparâmetros Aleatórios**: Foi realizado um teste com combinações aleatórias de hiperparâmetros para descobrir quais performavam melhor. Nos testes aleatórios, o melhor resultado com 1000 epochs foi `hidden_size=20`, `learning_rate=0.1`, e uma **acurácia de 95.05%**.

### Resultados Obtidos:

Com os seguintes parâmetros:
- `hidden_size = 10`
- `output_size = 1`
- `learning_rate = 0.01`
- `epochs = 1000`

Esses foram os valores médios:
- **Precisão**: 0.91
- **Recall**: 0.96
- **F1-Score**: 0.93
- **Acurácia**: 0.93

### Melhorias e Observações:

Como melhorias, eu poderia ter testado mais parâmetros diferentes. A maior dificuldade foi a normalização dos dados para o treinamento e para a realização do diagnóstico no projeto `main.py`.

Sempre que os resultados do diagnóstico eram retornados, eles apresentavam números nas extremidades, ou muito altos ou muito baixos. Não sei se isso pode ser um erro ou se faz parte do comportamento do modelo.
