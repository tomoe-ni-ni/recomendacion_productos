from flask import Flask, render_template, request
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)

# Cargar el archivo CSV
data = pd.read_csv('/home/ninidani/productos/producto.csv')

# Dividir las transacciones en productos separados por comas
data['Producto'] = data['Producto'].apply(lambda x: x.split(', '))

# Expandir la lista de productos en una tabla donde cada columna sea un producto
basket = data['Producto'].str.join('|').str.get_dummies()

# Aplicar Apriori para encontrar conjuntos frecuentes de productos
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)  # Reduce el soporte mínimo

# Generar reglas de asociación a partir de los conjuntos frecuentes
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Imprimir las reglas generadas para diagnóstico
print(rules)  # Esto te ayudará a ver qué reglas se han generado

@app.route('/')
def index():
    productos_disponibles = basket.columns.tolist()
    return render_template('index.html', productos=productos_disponibles)

@app.route('/recomendar', methods=['POST'])
def recomendar():
    producto_usuario = request.form['producto']

    if producto_usuario not in basket.columns:
        return render_template('index.html', productos=basket.columns.tolist(), error=f"El producto '{producto_usuario}' no está disponible.", producto_seleccionado=producto_usuario)

    # Filtrar las reglas donde el producto ingresado está en los antecedentes
    recomendaciones = rules[rules['antecedents'].apply(lambda x: producto_usuario in x)]

    if recomendaciones.empty:
        return render_template('index.html', productos=basket.columns.tolist(), error=f"No hay recomendaciones disponibles para '{producto_usuario}'.", producto_seleccionado=producto_usuario)

    recomendaciones_set = set()  # Usamos un conjunto para evitar duplicados

    for index, row in recomendaciones.iterrows():
        # Agregar los consecuentes al conjunto
        for consequent in row['consequents']:
            recomendaciones_set.add(consequent)

    # Convertir el conjunto a una lista de diccionarios
    recomendaciones_list = [{'consequents': [producto]} for producto in recomendaciones_set]

    return render_template('index.html', productos=basket.columns.tolist(), recomendaciones=recomendaciones_list, producto_usuario=producto_usuario, producto_seleccionado=producto_usuario)

if __name__ == '__main__':
    app.run(debug=True)
