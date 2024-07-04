# PlanKG

This is the GitHub repository for the PlanKG paper at ECIS 2024.

---

Abstract: 
Planning for complex tasks is a key task for knowledge workers that is often time-consuming and depends on the manual extraction of knowledge from documents. In this research, we propose an end-to-end method, called PlanKG, that: (1) extracts knowledge graphs from full-text plan descriptions(FTPD); and (2) generates novel FTPD according to plan requirements and context informationprovided by users. From the knowledge graphs, activity sequences are obtained and projected into embedding spaces. We show that compressed activity sequences are sufficient for the search and generation of plan descriptions. The PlanKG method uses a pipeline consisting of decoder-only transformer models and encoder-only transformer models. To evaluate the PlanKG method, we conducted an experimental study for movie plot descriptions and compared our method with original FTPDs and FTPD summarizations. The results of this research has significant potential for enhancing efficiency and precision when searching and generating plans.



# Repository Structure

.
├── README.md
├── Resources
│   ├── JupyterNotebooks
│   │   ├── Benchmark2.ipynb
│   │   ├── PlanKG_sim.ipynb
│   │   └── cluster.ipynb
│   ├── Prompts
│   │   └── singlePrompts
│   │       ├── ADarkSong.txt
│   │       ├── Annabelle.txt
│   │       ├── Atonement.txt
│   │       ├── Australia.txt
│   │       ├── Belle.txt
│   │       ├── BrightStar.txt


│   ├── RDFs
│   │   ├── AllRDFs.docx
│   │   ├── actionRDF.txt
│   │   ├── comedyRDF.txt
│   │   └── horrorRDF.txt
│   ├── Summarizations
│   │   └── singleSummarizations
│   │       ├── ADarkSong.txt
│   │       ├── Annabelle.txt
│   │       ├── Atonement.txt
│   │       ├── Australia.txt
│   │       ├── Belle.txt
│   │       ├── BrightStar.txt



│   └── Visualizations
│       ├── AtomicBlondeRDF.png
│       ├── BudapestRDF-1.png
│       ├── RDF_Titanic.png
│       └── TitanicKnowledgeGraphRDFtext.txt
├── benchmark.py
├── cluster_results
├── data
│   ├── KG_ActivitySequences.txt
│   ├── LLMsummaries.txt
│   ├── WikiDescriptions.txt
│   ├── testPlotComedy.txt
│   ├── testWikiAction.txt
│   └── testWikiComedy.txt
├── main.py
├── plan_generation.py
├── results
│   ├── combined_cluster.png
│   ├── kg_cluster.png
│   ├── plots_cluster.png
│   ├── summaries_cluster.png
│   └── top_3_act_seq.txt
└── tree_structure.txt

12 directories, 108 files



## Code
For the code related to the paper, see the file "main.py". All benchmark-related code (clustering, similarity analysis) can be found there. main.py saves the top three similar set of activity sequences to a text file. This can then be given as input to plan_generation.py to generate novel movie plots. 

## Resources
You can find the prompts used with ChatGPT in the top directory. Other resources, like the generated RDFs about movie plots, can be found in the "Resources" folder.
