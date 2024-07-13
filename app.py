from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from sklearn.cluster import KMeans
import pandas as pd
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Menggunakan backend non-interaktif
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Contoh data awal
data = [
    {"Jenis Tanah": "Tanah Lempung", "Kelembapan Tanah": 765, "pH Tanah": 6.09, "Suhu Udara": 32.2284},
    {"Jenis Tanah": "Tanah Humus", "Kelembapan Tanah": 808, "pH Tanah": 6.45, "Suhu Udara": 34.2221},
    {"Jenis Tanah": "Tanah Latosol", "Kelembapan Tanah": 711, "pH Tanah": 5.08, "Suhu Udara": 33.1357},
    {"Jenis Tanah": "Tanah Entisol", "Kelembapan Tanah": 925, "pH Tanah": 7.39, "Suhu Udara": 32.181},
    {"Jenis Tanah": "Tanah Laterit", "Kelembapan Tanah": 689, "pH Tanah": 5.31, "Suhu Udara": 30.01}
]

# Titik pusat awal untuk centroid
initial_centroids = np.array([
    [765, 6.09, 32.2284],  # Centroid 1
    [808, 6.45, 34.2221]   # Centroid 2
])

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def perform_kmeans_with_initial_centroids(df):
    X = df[["Kelembapan Tanah", "pH Tanah", "Suhu Udara"]].values
    kmeans = KMeans(n_clusters=len(initial_centroids), init=initial_centroids, n_init=1)
    kmeans.fit(X)
    df['C1'] = [euclidean_distance(x, initial_centroids[0]) for x in X]
    df['C2'] = [euclidean_distance(x, initial_centroids[1]) for x in X]
    df['K'] = ['Centroid 1' if c1 < c2 else 'Centroid 2' for c1, c2 in zip(df['C1'], df['C2'])]
    return df, kmeans.cluster_centers_

def plot_clusters(df, centroids):
    X = df[["Kelembapan Tanah", "pH Tanah", "Suhu Udara"]].values
    labels = df['K'].apply(lambda x: int(x.split()[-1]) - 1).values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis')

    for i, centroid in enumerate(centroids):
        ax.scatter(centroid[0], centroid[1], centroid[2], color='red', marker='x', s=100, label=f'Centroid {i+1}')

    ax.set_xlabel('Kelembapan Tanah')
    ax.set_ylabel('pH Tanah')
    ax.set_zlabel('Suhu Udara')
    plt.legend()
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'clusters.png'))
    plt.close()

@app.route('/')
def index():
    global data
    df = pd.DataFrame(data)
    df, centroids = perform_kmeans_with_initial_centroids(df)
    plot_clusters(df, centroids)
    data = df.to_dict(orient='records')
    return render_template('index.html', data=data)

@app.route('/import', methods=['POST'])
def import_file():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    df, centroids = perform_kmeans_with_initial_centroids(df)
    plot_clusters(df, centroids)

    global data
    data = df.to_dict(orient='records')

    return redirect(url_for('index'))

@app.route('/add', methods=['POST'])
def add_data():
    try:
        jenis_tanah = request.form['jenis_tanah']
        kelembapan_tanah = float(request.form['kelembapan_tanah'])
        ph_tanah = float(request.form['ph_tanah'])
        suhu_udara = float(request.form['suhu_udara'])

        new_entry = {
            "Jenis Tanah": jenis_tanah,
            "Kelembapan Tanah": kelembapan_tanah,
            "pH Tanah": ph_tanah,
            "Suhu Udara": suhu_udara
        }

        global data
        data.append(new_entry)

        df = pd.DataFrame(data)
        df, centroids = perform_kmeans_with_initial_centroids(df)
        plot_clusters(df, centroids)
        data = df.to_dict(orient='records')

    except Exception as e:
        print(f"Error adding data: {e}")

    return redirect(url_for('index'))

@app.route('/delete/<int:index>', methods=['POST'])
def delete_data(index):
    global data
    data.pop(index)

    df = pd.DataFrame(data)
    df, centroids = perform_kmeans_with_initial_centroids(df)
    plot_clusters(df, centroids)
    data = df.to_dict(orient='records')

    return redirect(url_for('index'))

@app.route('/clusters', methods=['GET'])
def clusters():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'clusters.png'), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
