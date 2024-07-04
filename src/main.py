from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from numpy import savetxt
import tensorflow as tf
import tensorflow_hub as hub
import umap
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
openai.api_key = "OPENAI API KEY"
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

class FaithfullnessCalculator:
    def __init__(self):
        self.sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
        self.use_model = hub.load(self.module_url)
        print("module %s loaded" % self.module_url)
    
    def create_cluster(self,embeddings,file_name):
        """
        Creates a cluster visualization of the embeddings using dimension reduction (UMAP) and clustering (KMeans).

        Parameters:
            embeddings (numpy.ndarray): The embeddings to visualize.
            file_name (str): The name of the output file for the cluster visualization.

        Returns:
            None
        """
        # create array with integer labels from 0 to length of embeddings
        labels = np.arange(len(embeddings))

        #tsne = TSNE(n_components=2,perplexity=25)
        #tsne_result = tsne.fit_transform(embeddings)
        
        # Perform dimension reduction using UMAP
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.2, metric='correlation')
        embedding = reducer.fit_transform(embeddings)

        # Perform clustering using KMeans
        kmeans = KMeans(n_clusters=4
                       )
        kmeans.fit(embedding)
        
        # Plot the cluster visualization
        fig, ax = plt.subplots(figsize=(10,10))
        cmap = matplotlib.cm.get_cmap('plasma', 5)
        sc = plt.scatter(embedding[:,0], embedding[:,1], c=kmeans.labels_,cmap=cmap)
        
        # Annotate the points with their corresponding labels
        for i, label in enumerate(labels):
            plt.annotate(label + 1, (embedding[i,0], embedding[i,1]))

        # Create a legend based on the unique labels
        unique_labels = set(kmeans.labels_)
        leg = []
        for label in unique_labels:
            leg.append(str(label))

        print("Legend: ", leg)
        plt.legend(*sc.legend_elements(), title='clusters')
        
        # Save the cluster visualization to a file
        plt.savefig(f'{file_name}.png')
        plt.show()
    
    def sbert(self, dataset):
        """
        Calculates the cosine similarity between each abstract and its corresponding summary using SBERT.

        Parameters:
            dataset (list): The dataset containing abstracts and summaries.

        Returns:
            pandas.DataFrame: A DataFrame containing the similarity scores between the abstracts and summaries.
        """
        # Obtain embeddings for the entire dataset using the SBERT model
        dataset_embeddings = self.sbert_model.encode(dataset)
        
        # Extract the summary embeddings from the dataset embeddings
        summary_embeddings = dataset_embeddings[:10]
        
        # Extract the abstract embeddings from the dataset embeddings
        abstract_embeddings = dataset_embeddings[10:20]
        
        # Create a cluster using the dataset embeddings and save it to a file
        self.create_cluster(dataset_embeddings,file_name='sbert_cluster')
        
        # calculate cosine similarity between each abstract and correcponsing summary 
        l = []
        for i in range(len(summary_embeddings)):
            # Reshape the summary and abstract embeddings to match the expected input shape of cosine_similarity
            sum_emb = summary_embeddings[i].reshape(1, -1)
            abs_emb = abstract_embeddings[i].reshape(1, -1)
            
            # Calculate cosine similarity using cosine_similarity
            sim_sum_cos = cosine_similarity(sum_emb, abs_emb)
            
            # Append the similarity score to the list
            l.append(sim_sum_cos)
        
        # Convert the 3D list (similarity_scores) to a DataFrame
        f = []
        for i in l:
            for s in i:
                for si in s:
                    f.append(s)
        d = pd.DataFrame(f, columns=['sim scores'])
        
        # Return the results DataFrame
        return d
        
    def get_embedding(self,text_or_tokens, model=EMBEDDING_MODEL):
      """
      Retrieves the embedding for the given text or tokens using the OPENAI model.

      Parameters:
          text_or_tokens: The input text or tokens for which to retrieve the embedding.
          model (str): The name of the embedding model to use. Defaults to EMBEDDING_MODEL.

      Returns:
          The embedding array for the input text or tokens.
      """
      return openai.Embedding.create(input=text_or_tokens, model=model)["data"][0]["embedding"]

    def cosine(self, u, v):
      """
      Calculates the cosine similarity between two vectors u and v.

      Parameters:
          u (numpy.ndarray): The first vector.
          v (numpy.ndarray): The second vector.

      Returns:
          float: The cosine similarity between u and v.
      """
      return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    def USE(self, dataset):
      """
      Calculates the cosine similarity between each abstract and its corresponding summary using the Universal Sentence Encoder (USE).

      Parameters:
          dataset (list): The dataset containing abstracts and summaries.

      Returns:
          numpy.ndarray: An array of similarity scores between the abstracts and summaries.
      """
      # Obtain embeddings for the entire dataset using the USE model
      dataset_embeddings_USE = self.use_model(dataset)
      
      # Extract the summary embeddings from the dataset embeddings
      summary_embeddings_USE = dataset_embeddings_USE[:10]
      
      # Extract the abstract embeddings from the dataset embeddings
      abstract_embeddings_USE = dataset_embeddings_USE[10:20]
      
      # Create a cluster using the dataset embeddings and save it to a file
      self.create_cluster(dataset_embeddings_USE,file_name='use_cluster')
      
      # calculate cosine similarity between each abstract and correcponsing summary 
      similarity_scores = []
      for i in range(len(summary_embeddings_USE)):
          # Calculate cosine similarity using the self.cosine function
          sim_sum_cos = self.cosine(summary_embeddings_USE[i], abstract_embeddings_USE[i])
          
          # Append the similarity score to the list
          similarity_scores.append(sim_sum_cos)
      
      # Return the similarity scores as a numpy array
      return np.array(similarity_scores)

    def openai_cosine(self,dataset):
      """
      Calculates the cosine similarity between each abstract and its corresponding summary using OpenAI's embedding model.

      Parameters:
          dataset (list): The dataset containing abstracts and summaries.

      Returns:
          pandas.Series: A Series containing the cosine similarity scores between the abstracts and summaries.
      """
      # Create a DataFrame from the dataset
      df = pd.DataFrame(dataset)
      
      # Create a DataFrame with a single column named 'dataset'
      dfw = pd.DataFrame(dataset)
      dfw.columns = ["dataset"]
      
      # Extract the summary and abstract data from the DataFrame
      summary_df = df[0:10]
      abstract_df = df[10:20]
      abstract_df.columns = ["Paper"]
      summary_df.columns = ["Paper"]
      
      # Create a new DataFrame to store the inputs for cosine similarity calculation
      df_inputs = pd.DataFrame()
      df_inputs['Abstract'] = abstract_df['Paper']
      df_inputs.index = summary_df.index
      df_inputs['Summary'] = summary_df['Paper']
      
      # Create a DataFrame to store the embeddings and calculate the OpenAI cluster
      df1 = pd.DataFrame()
      df1['text'] = dfw["dataset"]
      df1['text_embedding'] = df1.text.apply(lambda x: self.get_embedding(x))
      self.create_cluster(list(df1['text_embedding']),file_name='openai_cluster')
      
      # Retrieve embeddings for the abstracts and summaries
      df_inputs['Abstract_Embedding'] = df_inputs.Abstract.apply(lambda x: self.get_embedding(x))     
      df_inputs['Summary_Embedding'] = df_inputs.Summary.apply(lambda x: self.get_embedding(x))     
      # Calculate cosine similarity scores
      df_inputs['Cosine_Similarity'] = df_inputs.apply(lambda row: cosine_similarity([row['Abstract_Embedding']], [row['Summary_Embedding']])[0][0], axis=1)
      
      # Retrieve the cosine similarity scores
      sim_sum = df_inputs['Cosine_Similarity']
      
      # Return the cosine similarity scores 
      return sim_sum

def main():
    # get data
    with open('dataset.txt') as f:
        dataset = f.read().splitlines()
    
    faithfullness_calculator = FaithfullnessCalculator()
    # use sbert embedding to find faithfullness
    results_sbert = faithfullness_calculator.sbert(dataset)
    # save results
    results_sbert.to_csv("sbert_sim.csv", sep='\t')
    
    # use USE embedding to find faithfullness      
    results_use = faithfullness_calculator.USE(dataset)
    # save results
    savetxt('use_sim.csv', results_use, delimiter=',')
    
    # use openai embedding to find faithfullness 
    results_openai = faithfullness_calculator.openai_cosine(dataset)
    # save results
    results_openai.to_csv("embeddings_with_cosine_similarity.csv", index=False)
if __name__ == '__main__':
    main()
