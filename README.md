<table border="0">
  <tr>
    <td valign="middle" width="200">
  <img src="https://github.com/user-attachments/assets/7f6f1a68-a566-41b9-a074-605342f220c6" alt="logo" style="width:200px;"/>
    </td>
    <td align="right" valign="middle">
      <h1>Multimodal Predictive Process Monitoring and its Application to Explainable Clinical Pathways</h1>
    </td>
  </tr>
</table>

**The repository contains code referred to the work:**

*Vincenzo Pasquadibisceglie, Ivan Donadello, Annalisa Appice, Oswald Lanz, Fabrizio Maria Maggi, Giuseppe Fiameni, Donato Malerba*


[*Multimodal Predictive Process Monitoring and its Application to Explainable Clinical Pathways*]

# How to Use

To run the scripts in this repository, you first need to download the **CDLS dataset** from the following link:  
👉 https://physionet.org/content/covid-data-shared-learning/1.0.0/  

Please make sure to follow all requirements specified by PhysioNet.

---

## Step 1: Prepare the Dataset
After downloading, copy the following files into your dataset folder:
atc5.csv, diagnosis_hosp_03.csv, patient_01.csv,
atc7.csv vital_signs_04.csv,
CDSL-1.0.0-dicom-metadata.csv, icd10_codes_dict.txt,
diagnosis_er_02.csv lab_06.csv,
medication_05.csv


## Step 2: Generate the Multimodal Event Log
```
./generation.sh
```
## Step 3: Extract Labeled Image–Text Pairs
Once the multimodal event log has been generated, specify the image folder path inside **create_mm_examples.py**.  
For example: path_folder = '/home/covid_log/IMAGES/' <br>
Then run:
```
python -m mm_log_generation.create_mm_examples
```

## Step 4: Fine-Tune Pre-Trained Models

Once the labelled examples have been generated, you can fine-tune the pre-trained models.

### Unimodal Text Model
```
python -m unimodal.finetune_txt_model roberta
```
### Unimodal Vision Model
```
python -m unimodal.finetune_vis_model vit
```
### Multimodal Model
```
python -m multimodal.finetune_concat
python -m multimodal.finetune_flava
```
## Multimodal ML Model
- Embeddings generations:
```
  python -m multimodal_ml.embeddings_generation
```
- Training:
```
  python -m multimodal_ml.train_ml_model lr
```
