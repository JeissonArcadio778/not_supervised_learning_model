import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Datos estáticos para el ejemplo
data = {
    "Hora del Día": ["8:00", "9:00", "17:00", "18:00", "21:00", "22:00", "6:00", "7:00", "12:00", "13:00"],
    "Día de la Semana": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo", "Lunes", "Martes", "Miércoles"],
    "Temporada del Año": ["Verano", "Verano", "Otoño", "Otoño", "Invierno", "Invierno", "Primavera", "Primavera", "Verano", "Verano"],
    "Estación de Transporte": ["Estación 1", "Estación 2", "Estación 3", "Estación 4", "Estación 5", "Estación 1", "Estación 2", "Estación 3", "Estación 4", "Estación 5"],
    "Número de Pasajeros": [200, 150, 300, 250, 100, 80, 190, 230, 160, 120],
    "Duración Promedio de Viaje": [15, 20, 35, 30, 25, 10, 14, 18, 22, 28],
    "Incidentes Reportados": [0, 1, 0, 2, 3, 0, 1, 0, 0, 1]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Convertir datos categóricos en numéricos usando one-hot encoding
df_encoded = pd.get_dummies(df, columns=["Hora del Día", "Día de la Semana", "Temporada del Año", "Estación de Transporte"])

# Escalar los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_scaled)

# Añadir las etiquetas de cluster al DataFrame original
df['Cluster'] = kmeans.labels_

# Visualizar los resultados del clustering
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
plt.figure(figsize=(8, 8))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('Clustering de Estaciones de Transporte')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar()
plt.show()

# Mostrar los resultados
print(df)
