# ===============================================
# Global Solution 
# ===============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# === 1) Carregamento e preparação dos dados ===
df = pd.read_csv("flood_risk_dataset_india.csv")
df.rename(columns={
    "Rainfall (mm)": "Precipitacao",
    "Temperature (°C)": "Temperatura"
}, inplace=True)

# === 2) Tabelas de frequência ===
variavel_discreta = df['Flood Occurred']
freq_discreta = variavel_discreta.value_counts().sort_index()

variavel_continua = df['Temperatura']
classes = pd.cut(variavel_continua, bins=5)
freq_continua = classes.value_counts().sort_index()

# === Tabelas de Frequência ===
print("\nTABELA DE FREQUÊNCIA - VARIÁVEL DISCRETA (Flood Occurred)")
print("-" * 50)
print(freq_discreta.to_string())
print("-" * 50)

print("\nTABELA DE FREQUÊNCIA - VARIÁVEL CONTÍNUA (Temperatura)")
print("-" * 50)
print(freq_continua.to_string())
print("-" * 50)

# === 3) Gráfico de Barras (Discreta) ===
plt.figure(figsize=(8, 4))
bars = plt.bar(['Não', 'Sim'], freq_discreta, color='cornflowerblue')
plt.title('Distribuição de Ocorrência de Enchentes')
plt.xlabel('Enchente Ocorrida')
plt.ylabel('Frequência')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Rótulos nas barras
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 100, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.savefig("grafico_discreto.png")
plt.show()

# === 4) Histograma (Contínua) ===
plt.figure(figsize=(8, 4))
counts, bins, patches = plt.hist(variavel_continua, bins=10, color='#FF6666', edgecolor='black')
plt.title('Distribuição da Temperatura')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Frequência')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Rótulos nas colunas
for i in range(len(patches)):
    plt.text(patches[i].get_x() + patches[i].get_width()/2, counts[i] + 50, int(counts[i]), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("grafico_continuo.png")
plt.show()

# === 5) Estatística Descritiva ===
media = variavel_continua.mean()
mediana = variavel_continua.median()
moda = variavel_continua.mode().iloc[0]
max_val = variavel_continua.max()
min_val = variavel_continua.min()
amplitude = max_val - min_val
variancia = variavel_continua.var()
desvio_padrao = variavel_continua.std()
coef_var = (desvio_padrao / media) * 100
quartis = variavel_continua.quantile([0.25, 0.5, 0.75])

# === Estatística Descritiva ===
print("\nESTATÍSTICA DESCRITIVA - TEMPERATURA")
print("-" * 50)
print(f"Média:                 {media:.2f} °C")
print(f"Mediana:               {mediana:.2f} °C")
print(f"Moda:                  {moda:.2f} °C")
print(f"Máximo:                {max_val:.2f} °C")
print(f"Mínimo:                {min_val:.2f} °C")
print(f"Amplitude:             {amplitude:.2f} °C")
print(f"Variância:             {variancia:.2f}")
print(f"Desvio Padrão:         {desvio_padrao:.2f}")
print(f"Coef. Variação:        {coef_var:.2f} %")
print(f"Quartis:")
print(f"    Q1 (25%): {quartis[0.25]:.2f} °C")
print(f"    Q2 (50%): {quartis[0.50]:.2f} °C")
print(f"    Q3 (75%): {quartis[0.75]:.2f} °C")
print("-" * 50)

# === 6) Regressão Linear Simples ===
X = df[['Temperatura']]
y = df['Precipitacao']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

r2 = modelo.score(X_test, y_test)
correlacao = df['Temperatura'].corr(df['Precipitacao'])

# === Regressão Linear ===
print("\nREGRESSÃO LINEAR - TEMPERATURA x PRECIPITAÇÃO")
print("-" * 50)
print(f"Coeficiente Angular (a):     {modelo.coef_[0]:.4f}")
print(f"Intercepto (b):              {modelo.intercept_:.4f}")
print(f"Score R² (ajuste):           {r2:.4f}")
print(f"Correlação (r):              {correlacao:.4f}")
print("-" * 50)

# === Gráfico de Regressão ===
df_sample = df.sample(n=1000, random_state=42)
plt.figure(figsize=(8, 4))
plt.scatter(df_sample['Temperatura'], df_sample['Precipitacao'], color='gray', label='Dados reais', s=10, alpha=0.6)
plt.plot(X, modelo.predict(X), color='blue', linewidth=2, label='Regressão Linear')
plt.title('Regressão Linear: Temperatura x Precipitação')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Precipitação (mm)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Texto de R² e r
plt.text(0.95, 0.05, f"R²: {r2:.4f}\nr: {correlacao:.4f}",
         transform=plt.gca().transAxes, fontsize=9, verticalalignment='bottom',
         horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.6))

plt.tight_layout()
plt.savefig("grafico_regressao.png")
plt.show()
