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

---

### Read our paper

(Paper)(https://www.iss.uni-saarland.de/wp-content/uploads/2023/07/Snoop_AAAI_SS23_MedicalAI_submission.pdf)

### Citation
```
Maass, W., Agnes, C. K., Rahman, MR., Almeida, J. S. (2023). SNOOP Method: Faithfulness of Text Summarizations for Single Nucleotide Polymorphisms. 2nd Symposium on Human Partnership with Medical AI: Design, Operationalization, and Ethics at the Association for the Advancement of Artificial Intelligence (AAAI)Summer Symposium 2023, Singapore.

```
