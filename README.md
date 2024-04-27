# DGCL

This repository provides the code and data for the paper

>Prediction of multi-relational drug-gene interaction via dynamic hypergraph contrastive learning

## Citation
If you want to use our code and datasets in your research, please cite:

```
@article{tao2023prediction,
  title={Prediction of multi-relational drug--gene interaction via Dynamic hyperGraph Contrastive Learning},
  author={Tao, Wen and Liu, Yuansheng and Lin, Xuan and Song, Bosheng and Zeng, Xiangxiang},
  journal={Briefings in Bioinformatics},
  volume={24},
  number={6},
  pages={bbad371},
  year={2023},
  publisher={Oxford University Press}
}
```

## 1. Requirements

To run the code, you need the following dependencies:

```
python                        3.10.4
pytorch                       1.11.0
numpy                         1.22.3
pandas                        1.5.2
scipy                         1.7.3
```

## 2. Datasets

We utilize three datasets to evaluate DGCL: DrugBank, DGIdb, and LINCS L1000. The DrugBank and DGIdb datasets are divided into training and testing sets with an 8:2 ratio. LINCS L1000 is used as an external testing set.

## 3. Training and Evaluation

Please use the following command:

* DrugBank
```bash
python Main.py --data DrugBank --iteration 5
```
* DGIdb
```bash
python Main.py --data DGIdb --iteration 5
```
* LINCS L1000
```bash
python Main.py --data LINCS --iteration 1 --load_model ckl/1_model_20230607_DrugBank
```
The "ckl/1_model_20230607_DrugBank" represents the model trained on the DrugBank dataset.

## 4. Instructions for Using Alternate Dataset

This project supports the use of an alternate dataset for training or testing the model. The alternate dataset should follow the format: DrugID, GeneID, Label.

### Dataset Format

The alternate dataset should contain three columns:

1. DrugID: This column should contain a unique identifier or name related to the drug.

2. GeneID: This column should contain a unique identifier or name related to the gene.

3. Label: This column should contain a label or identifier for the drug-gene interaction.

|  DrugID  | GeneID | Label |
| -------------|-------------|-----|
|  DB03756     | 6257        | 9   |
|  ...         | ...         | ... |
|  DB00270     | 775         | 1   |

Please ensure that the dataset is stored in the correct format.

### Using the Alternate Dataset
To use the alternate dataset, follow these steps:

1. Prepare the Dataset File: Ensure that you have prepared the alternate dataset file, and it adheres to the format requirements mentioned above.

2. Splitting Dataset: Split the dataset into training and testing sets. Here, the DrugBank and DGIdb datasets are split in an 8:2 ratio.

3. Re-Train or Test the Model: Depending on your requirements, re-run the model's training or testing process. Ensure that the model is using the alternate dataset you provided.

4. Evaluate the Results: Based on the results obtained from training or testing with the alternate dataset, evaluate the model's performance and record any necessary metrics or outputs.

### Considerations
* Ensure the quality and integrity of the alternate dataset to ensure the reliability of the model's training or testing results.

* When replacing the default dataset, make sure that the paths and data loading processes in the code are correct.

* Depending on the characteristics of the alternate dataset, you may need to adjust the model's parameters or hyperparameters to achieve optimal performance.

Please use the alternate dataset according to your specific needs and project configuration to achieve the best results. If you have any further questions or require additional assistance, feel free to contact [Wen Tao](taowen@hnu.edu.cn).


For any clarification, comments, or suggestions please create an issue or contact [Wen Tao](taowen@hnu.edu.cn).
