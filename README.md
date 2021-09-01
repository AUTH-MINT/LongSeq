# LongSeq: Improved Biomedical Entity Recognition via longer context modeling

This repository hosts the implementation described in our paper [Improved Biomedical Entity Recognition via longer context modeling](https://www.springerprofessional.de/en/improved-biomedical-entity-recognition-via-longer-context-modeli/19280400).

# Introduction
LongSeq is a state-of-the-art Evidence-Based Medicine (EBM) and Biomedical Named Entity Recognition (BioNER) system. Our model is end-to-end and capable of efficiently model-ing significantly longer sequences than previous models, benefiting frominter-sentence dependencies .


# LongSeq models architecture

<img src="https://github.com/AUTH-MINT/LongSeq/blob/main/Mobel_Architecture.png" width="800">

# Requirements
With Python 3.6 or higher, use the requirements file provided as follows: 

```
pip install -r requirements.txt
```
 
You will also need to download, and format accordingly, the [EBM-NLP corpus](https://github.com/bepnye/EBM-NLP) using the
 scripts ``build_data.py`` and ``build_data_abstracts.py``. The preprocessed EBM data are provided in ``data/``.

# Results 
Detailed performance scores, per entity class, in terms of Precision, Recall, and F1 on all datasets.

<table>
 <tr><th> Results per Dataset using Sentences</th><th> Results per Dataset using Abstracts</th></tr>
 <tr><td>

| Dataset  |  P  |  R  |  F1 |           
|---------------------|:---:|:---:|:---:|
| EBM-NLP             | 78% | 79% | 78% |
| NCBI-Disease        | 95% | 96% | 95% |
| JNLPBA              | 63% | 77% | 59% |
| SCAI-Chemical       | 85% | 72% | 76% |
| SCAI-Disease        | 90% | 87% | 88% |

</td><td>
 
| Dataset  |  P  |  R  |  F1 |
|---------------------|:---:|:---:|:---:|
| EBM-NLP             | 79% | 81% | 79% |
| NCBI-Disease        | 96% | 96% | 96% |
| JNLPBA              | 67% | 80% | 65% |
| SCAI-Chemical       | 91% | 87% | 88% |
| SCAI-Disease        | 96% | 96% | 96% |
</td></tr> </table>


Detailed Results per datasets, including Precision, Recall and F1-score for each class, can be found in the [Results.md](https://github.com/AUTH-MINT/LongSeq/blob/main/Results.md). 

# Citation
The accompanying paper has been accepted and is set to appear in IFIP International Conference on Artificial Intelligence Applications and Innovations 2021.
Detailed citation information will become available once the proceedings are published. 
