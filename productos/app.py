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
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)

# Generar reglas de asociación a partir de los conjuntos frecuentes
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)


@app.route('/')
def index():
    productos_disponibles = basket.columns.tolist()
    return render_template('index.html', productos=productos_disponibles)


@app.route('/recomendar', methods=['POST'])
def recomendar():
    producto_usuario = request.form['producto']

    if producto_usuario not in basket.columns:
        return render_template('index.html', productos=basket.columns.tolist(), error=f"El producto '{producto_usuario}' no está disponible.")

    # Filtrar las reglas donde el producto ingresado está en los antecedentes
    recomendaciones = rules[rules['antecedents'].apply(lambda x: producto_usuario in x)]

    if recomendaciones.empty:
        return render_template('index.html', productos=basket.columns.tolist(), error=f"No hay recomendaciones disponibles para '{producto_usuario}'.")

    recomendaciones_list = []
    for index, row in recomendaciones.iterrows():
        recomendaciones_list.append({
            'consequents': list(row['consequents']),  # Solo obtenemos los consecuintes
        })

    return render_template('index.html', productos=basket.columns.tolist(), recomendaciones=recomendaciones_list, producto_usuario=producto_usuario)


if __name__ == '__main__':
    app.run(debug=True)
