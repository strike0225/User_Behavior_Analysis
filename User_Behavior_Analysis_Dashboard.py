import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv('online_shoppers_intention.csv')
pca_kmeans = pd.read_csv('pca_kmeans.csv')
tsne_kmeans = pd.read_csv('tsne_kmeans.csv')
pca_dbscan = pd.read_csv('pca_dbscan.csv')
tsne_dbscan = pd.read_csv('tsne_dbscan.csv')
pca_spectral = pd.read_csv('pca_spectral.csv')
tsne_spectral = pd.read_csv('tsne_spectral.csv')
pca_gmm = pd.read_csv('pca_gmm.csv')
tsne_gmm = pd.read_csv('tsne_gmm.csv')
pca_fuzzycmeans = pd.read_csv('pca_fuzzycmeans.csv')
tsne_fuzzycmeans = pd.read_csv('tsne_fuzzycmeans.csv')

# Set page configuration
st.set_page_config(page_title="User Behavior Analysis", page_icon=":bar_chart:")

# Web title
st.title("User Behavior Analysis :earth_asia: :globe_with_meridians: :chart: :bar_chart:")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["K-Means", "DBSCAN", "Spectral Clustering", "Gaussian Mixture Model (GMM)", "Fuzzy C-Means"])

# Function to display the PCA plot in Streamlit
def display_pca_plot(pca_data):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_data['PCA1'], pca_data['PCA2'], c=pca_data['Cluster'], cmap='viridis', alpha=0.7, edgecolors='k')
    plt.colorbar(scatter)
    plt.title('PCA Clusters Plot for K-Means')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    
    # Display the plot in Streamlit
    st.pyplot(plt)

# Display the data in Streamlit
st.write("PCA Results for K-Means Clustering")
st.dataframe(pca_kmeans)

# Call the function to display the PCA plot
display_pca_plot(pca_kmeans)