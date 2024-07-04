## SNOOP Method: Faithfulness of Text Summarizations for Single Nucleotide Polymorphisms
This repository contains the implementation of the **SNOOP method**

File structure
```
▒   README.md
▒
+---data
▒       abstract.txt
▒       dataset.txt
▒       summary.txt
▒       text_with_embeddings.csv
▒
+---result
▒       cluster.png.jpg
▒       sbert_sim.csv
▒
+---src
        get_openai_embeddings.py
        main.py
        openai.py
```

* main.py    &emsp;       &emsp;&emsp;&emsp;          > Implementation of Snoop method 

* dataset.txt       &emsp;&emsp;          > the dataset consisting documents, references and out of domain texts
* text_with_embeddings.csv  &emsp;&emsp;  > contains the data (documents and references) and their embeddings obtained using openai embedding model
* get_openai_embeddings.py  &emsp;  > python file to create openai embeddings
* openai.py                &emsp;&emsp;&emsp;&emsp;   > To implement Snoop using openai embeddings, use this file.

