# Biomedical_Relation_Extraction
This project explores the use of large language models (LLMs) for extracting relations between biomedical entities such as chemicals, diseases, and genes using BC5CDR and CHEM_DIS_GENE datasets.

### Tools and Environment Setup

1. **Version Control:**
     - Repository name: `Biomedical_NER_RE`
     - Repository structure:
       ```
       ├── data/                  # Contains datasets.
       ├── src/                   # Source code for preprocessing, training, and evaluation.
       ├── models/                # Saved model weights.
       ├── results/               # Evaluation results and visualizations.
       ├── README.md              # Overview and instructions
       └── requirements.txt       # Dependencies
       ```

2. **Environment:**
   - **Cheaha**: Ensure you've have access to the computing resources for large-scale training.
 
     
3. **Install necessary dependencies:**

Key Features

	1.	Multi-Task Learning:
	•	Shared SciBERT encoder for both NER and RE tasks.
	•	Separate classification heads for NER (BIO tagging) and RE (relation classification).
	2.	Biomedical Domain-Specific Pretraining:
	•	Leveraged SciBERT (allenai/scibert_scivocab_cased), specifically designed for scientific and biomedical texts.
	3.	Joint Training:
	•	NER identifies entities (e.g., chemicals, diseases).
	•	RE extracts relationships between entities (e.g., chemical-disease therapeutic effects).

Dataset

	1.	BC5CDR:
	•	Entities: Chemicals, Diseases.
	•	Relations: Chemical-disease associations (e.g., CID relationships).
	2.	CHEM_DIS_GENE:
	•	Entities: Chemicals, Diseases, Genes.
	•	Relations: Multi-entity interactions (e.g., chem_gene:affects^expression, chem_disease:marker/mechanism).

Model Architecture

	•	Encoder:
	•	Shared SciBERT encoder (allenai/scibert_scivocab_cased).
	•	NER Head:
	•	Fully connected layer for token classification (BIO tagging).
	•	RE Head:
	•	Fully connected layer for sequence classification (relation types).

Requirements

Python Environment

	•	Python 3.8 or later.
	•	Install required packages:

pip install -r requirements.txt



Dependencies

	•	transformers
	•	torch
	•	scikit-learn


Usage:
* NER.ipynb
* RE.ipynb

1. Prepare Datasets

Place the datasets in the datasets/ directory. Ensure that the data is split into train/test and formatted with:
	•	NER: bio_tags with tokens and BIO labels.
	•	RE: relation_tuples with (entity1, relation, entity2).

2. Train the Model

Run the training script to fine-tune SciBERT for multi-task learning:

python train.py

3. Evaluate the Model

Evaluate the trained model on the test set:

python evaluate.py

4. Save/Load the Model

	•	Save the trained model:

torch.save(model.state_dict(), "multi_task_model_sci.pth")


	•	Load the model:

model.load_state_dict(torch.load("multi_task_model_sci.pth"))

Results

* NER_RE_Predictions.ipynb

Contributions
| **Team Member**   | **Responsibilities**                                                                 |
|-------------------|--------------------------------------------------------------------------------------|
| **Aziz**          |  Data preprocessing, Fine-tuning BioBert for RE. (RE_New.ipynb)                      |
| **Pratul C Perla**|  Data preprocessing, Baseline model setup, NER tasks, Evaluation. (NER.ipynb)        |

Future Work:

1. Implement Data Augmentation
2. Include Gene entity In NER Tasks
3. Train on Large Language Models
4. ⁠⁠Replicate a existing paper
5. ⁠⁠Train with different models like Mamba/Jamba

Acknowledgments
Special thanks to:
	•	Professor John D. Osborne for guidance on dataset selection and project structure.
