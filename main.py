from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import numpy as np
import umap
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib
import os

def create_cluster(model, sentences, labels,file_name):
    embeddings = model.encode(sentences)
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.2, metric='correlation')
    embedding = reducer.fit_transform(embeddings)

    cmap = matplotlib.cm.get_cmap('plasma', 5)
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(embedding)
    fig, ax = plt.subplots(figsize=(10, 10))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=kmeans.labels_, cmap=cmap)
    for i, label in enumerate(labels):
        plt.annotate(label + 1, (embedding[i, 0], embedding[i, 1]))

    unique_labels = set(kmeans.labels_)
    leg = [str(label) for label in unique_labels]

    print("Legend: ", leg)
    plt.legend(*sc.legend_elements(), title='clusters')
    plt.show()
    plt.savefig(file_name)
    plt.close(fig)


def find_top_three_largest(numbers):
    if len(numbers) < 3:
        raise ValueError("The list must contain at least 3 numbers.")
    top_three = sorted(numbers, reverse=True)[:3]
    return top_three


def save_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(f"{item}\n")


def compute_similarity(reference_embedding, embeddings, ranges):
    similarities = []
    for start, end in ranges:
        sim = cosine_similarity(np.atleast_2d(reference_embedding), embeddings[start:end])
        similarities.append(sim)
    return similarities

def main():
    print("PlanKG tool starting.")

    # Load SBERT model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    print("SBERT model loaded.")
    
    delimiter1 = "\n\n---\n\n"
    delimiter2 = "\n\n---stopword---\n\n"

    # Load the list of KGs from the files
    with open('/Users/cygnus/Documents/DFKI/healthai/Plankg/Health-AI/ER_23/data/KG_ActivitySequences.txt', 'r') as file:
            content = file.read()
            kg = content.split(delimiter2)
    print(len(kg))
    print(type(kg))
    # load wiki data
    with open('/Users/cygnus/Documents/DFKI/healthai/Plankg/Health-AI/ER_23/data/WikiDescriptions.txt', 'r') as file:
        plots = file.read().split(delimiter2)
    # load llm summaries
    with open('/Users/cygnus/Documents/DFKI/healthai/Plankg/Health-AI/ER_23/data/LLMsummaries.txt', 'r') as file:
        summaries = file.read().split(delimiter1)

    print("Movie-related data loaded.")
    results_dir = '/Users/cygnus/Documents/DFKI/healthai/Plankg/Health-AI/ER_23/results'
    os.makedirs(results_dir, exist_ok=True)
    # Perform clustering
    print("Starting clustering.")
    create_cluster(model, kg, np.arange(len(kg)),os.path.join(results_dir, 'kg_cluster.png'))
    create_cluster(model, plots, np.arange(len(plots)),os.path.join(results_dir, 'plots_cluster.png'))
    create_cluster(model, summaries, np.arange(len(summaries)),os.path.join(results_dir, 'summaries_cluster.png'))
    create_cluster(model, kg + plots + summaries, np.arange(len(kg) + len(plots) + len(summaries)), os.path.join(results_dir, 'combined_cluster.png'))
    print("All Clustering done.")

    # Embedding all plots
    plotEmbeddings = model.encode(plots)
    print(len(plotEmbeddings))

    with open('/Users/cygnus/Documents/DFKI/healthai/Plankg/Health-AI/ER_23/data/testWikiAction.txt', 'r') as file:
        refPlotAction = file.read().split(delimiter2)    
    refPlotAction = refPlotAction[0]

    
    with open('/Users/cygnus/Documents/DFKI/healthai/Plankg/Health-AI/ER_23/data/testWikiComedy.txt', 'r') as file:
        refPlotComedy = file.read().split(delimiter2)    
    refPlotComedy = refPlotComedy[0]

    
    refPlotEmbeddingAction = model.encode(refPlotAction)
    refPlotEmbeddingComedy = model.encode(refPlotComedy)

    ranges = [(0, 21), (21, 41), (41, 61), (61, 80)]

    # Compute similarities for action movie reference
    sim_action = compute_similarity(refPlotEmbeddingAction, plotEmbeddings,ranges)
    print("Similarity of 'The Equalizer' to various genres: ", sim_action)
    print(len(sim_action))
    # Compute similarities for comedy movie reference
    sim_comedy = compute_similarity(refPlotEmbeddingComedy, plotEmbeddings,ranges)
    print("Similarity of 'Dumb and Dumber' to various genres: ", sim_comedy)
    #print(sim_comedy.shape)
    # Find top three similarities
   

#LLM SUMMARIES

    sumEmbeddings = model.encode(summaries)

        # 2.3.1. Reference movie: The Equalizer (action movie)

    refSumAction = [
            "Robert McCall, a quiet man working at a hardware store, befriends a teenage prostitute named Alina who dreams of becoming a singer. When Alina is brutally beaten by her pimp, Robert seeks revenge and discovers a larger criminal syndicate behind it. With his skills as a vigilante, Robert takes down the criminals one by one, ultimately confronting the syndicate's leader and starting a new life as a helper of the oppressed."  
        ]

    refSumEmbeddingAction = model.encode(refSumAction)
# Compute similarities for action movie reference
    sim_action_act_seq = compute_similarity(refSumEmbeddingAction, sumEmbeddings,ranges)
    print("Similarity of LLM summary of 'The Equalizer' to LLM SUMMARIES - various genres: ", sim_action_act_seq)
    print(len(sim_action_act_seq))
    

# ACTIVITY SEQUENCES

    seqEmbeddings = model.encode(kg)
    refSeqAction = [
        "action thriller, QuietLife, Hopeful, Desire, Aggression, Injured, Deception, Determination, Vengeance, Justice, Guilt, Fear, Protective, Betrayal, Frustration, Concern, Dangerous, Fearful, Captured, Mysterious, Defiant, Threatening, Cunning, Triumphant, Vengeful, Inspired, GratitudeActions/Events continues, encounters, hasEmotion, VigilanteJustice, Criminals, returnsRacketeeringMoneyTo, blackmailedBy, beatsWith, after, worksAt, GunmanRobbery, robs, HardwareStore, travelsTo, helpsIdentify, formerColleagueOf, worksFor, DefenseIntelligenceAgency, posesAs, visits, hasEmotion, flashesPictureOf, offersAsWarning, walksAway, failsToAbduct, skipsMeetingWith, kills, guards, surprises, with, destroys, OilTankers, abducts, forcesToMeet, tracksDown, helpsTakeDown, takenIntoCustody, when, leavesNote, withMessage, threatens, formerSpetsnazOperative, RussianSecretPoliceAgent, belongsTo, guardedBy, tricksInto, electrocutingHimself, finds, inspires, postsOnlineAdsAs, thanks, describes"
    ]
    refSeqActionEmbedding = model.encode(refSeqAction)
    # Compute similarities for action movie reference
    sim_action_act_seq = compute_similarity(refSeqActionEmbedding, seqEmbeddings,ranges)
    print("Similarity of KG of 'The Equalizer' to ACTIVITY SEQUENCE various genres: ", sim_action_act_seq)
    print(len(sim_action_act_seq))
    # Compute similarities for comedy movie reference
    sim_comedy_act_seq = compute_similarity(refSeqActionEmbedding, seqEmbeddings,ranges)
    print("Similarity of Kg of 'Dumb and Dumber' ACTIVITY SEQUENCE to various genres: ", sim_comedy_act_seq)
    #print(sim_comedy.shape)


    # EXAMPLE: Find top three similar sequences of action movie extraction in the data

    concatenated_list1 = [item for array in sim_action_act_seq for sublist in array for item in sublist]
    print(len(concatenated_list1))
    plotEmbedding_list = kg
    similarity_list =  concatenated_list1
        # Convert similarity_list to numpy array for easier manipulation
    similarity_array = np.array(similarity_list)

    # Get indices of top 3 largest values in similarity_list
    top_indices = similarity_array.argsort()[-3:][::-1]

    # Map indices to plotEmbedding_list
    top_plotEmbeddings = [plotEmbedding_list[idx] for idx in top_indices]

    print("Top 3 Similarity Values:", similarity_array[top_indices])
    print("Mapped plotEmbedding Values:", top_plotEmbeddings)
    print("Indices in plotEmbedding list:", top_indices)
    # Save top_plotEmbeddings to a file
    output_file = "results/top_3_act_seq.txt"
    with open(output_file, 'w') as file:
        for value in top_plotEmbeddings:
            file.write(str(value) + '\n')

    print(f"Top seq values saved to {output_file}")



if __name__ == '__main__':
    main()
