"""RAG 系统全面评估数据集。

包含 105 条手工标注的 QA 对，覆盖所有 7 篇医学文献，6 种问题类型 × 3 种难度。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

QuestionType = Literal[
    "factoid", "numerical", "comparative", "multi_hop", "summary", "terminology"
]
Difficulty = Literal["easy", "medium", "hard"]


@dataclass
class EvalItem:
    """单条评估用例。"""
    id: str
    question: str
    reference_answer: str  # 参考回答，用于生成评估
    paper_source: str  # 来源 paper 文件名
    question_type: QuestionType
    difficulty: Difficulty
    # 检索评估相关
    relevant_phrases: list[str] = field(default_factory=list)
    relevant_sections: list[str] = field(default_factory=list)
    # 关键词用于快速判定检索质量
    must_contain_keywords: list[str] = field(default_factory=list)


DATASET: list[EvalItem] = []


# ═══════════════════════════════════════════════════════════════
# Paper 1: Kather et al. 2019 — Deep Learning CRC Survival Prediction (15 Qs)
# ═══════════════════════════════════════════════════════════════

PAPER1 = "paper1_survival_prediction.pdf"

_p1_questions = [
    # ── Factoid (3) ──
    EvalItem(
        id="p1_q01",
        question="What is the 'deep stroma score' and how is it calculated?",
        reference_answer="The deep stroma score is a prognostic biomarker calculated from the output neuron activations of a CNN trained to classify 9 tissue types in H&E-stained CRC histology images. It quantifies the abundance of stromal tissue in the tumor microenvironment and was shown to be an independent prognostic factor for overall survival.",
        paper_source=PAPER1,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["deep stroma score", "output neuron activations", "stromal"],
        relevant_sections=["Abstract", "Deep stroma score"],
        must_contain_keywords=["deep stroma score", "output neuron", "stroma"],
    ),
    EvalItem(
        id="p1_q02",
        question="How many tissue classes can the CNN identify and what are some examples?",
        reference_answer="The CNN identifies nine tissue classes including: adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, and colorectal adenocarcinoma epithelium. It achieved >94% accuracy in an independent dataset of 7,180 images from 25 CRC patients.",
        paper_source=PAPER1,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["nine tissue classes", "nine-class accuracy", "adipose", "lymphocytes", "debris",
                          "smooth muscle", "normal colon mucosa", "cancer-associated stroma", "adenocarcinoma"],
        relevant_sections=["Training and testing"],
        must_contain_keywords=["nine", "tissue", "accuracy", "94"],
    ),
    EvalItem(
        id="p1_q03",
        question="What are the four consensus molecular subtypes (CMS) of colorectal cancer mentioned in the study?",
        reference_answer="The four consensus molecular subtypes are CMS1 (MSI immune, 14%), CMS2 (canonical, WNT/MYC, 37%), CMS3 (metabolic dysregulation, 13%), and CMS4 (mesenchymal, TGF-β, 23%). About 13% of samples are unclassifiable due to transitioning phenotypes or intratumoural heterogeneity.",
        paper_source=PAPER1,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["CMS", "consensus molecular subtype", "CMS1", "CMS2", "CMS3", "CMS4"],
        relevant_sections=["Discussion"],
        must_contain_keywords=["CMS", "consensus molecular subtype"],
    ),

    # ── Numerical (3) ──
    EvalItem(
        id="p1_q04",
        question="What was the hazard ratio of the deep stroma score in the TCGA cohort and was it statistically significant?",
        reference_answer="In the TCGA cohort, the deep stroma score had a hazard ratio of 1.99 (95% CI: 1.27-3.12, p=0.0028) for overall survival in a multivariable Cox proportional hazard model, making it a statistically significant independent prognostic factor.",
        paper_source=PAPER1,
        question_type="numerical", difficulty="medium",
        relevant_phrases=["hazard ratio", "1.99", "1.27-3.12", "p = 0.0028", "multivariable Cox"],
        relevant_sections=["Abstract", "Neural network assessment"],
        must_contain_keywords=["1.99", "hazard ratio", "Cox"],
    ),
    EvalItem(
        id="p1_q05",
        question="How many patients and HE slides were used from the TCGA and DACHS cohorts?",
        reference_answer="The TCGA cohort included 862 HE slides from 500 stage I-IV CRC patients. The DACHS validation cohort included 409 stage I-IV CRC patients recruited between 2003 and 2007 in multiple German institutions.",
        paper_source=PAPER1,
        question_type="numerical", difficulty="easy",
        relevant_phrases=["862 HE slides", "500 stage", "DACHS", "409 stage", "2003 and 2007"],
        relevant_sections=["Abstract", "Patient cohorts"],
        must_contain_keywords=["862", "500", "DACHS", "409"],
    ),
    EvalItem(
        id="p1_q06",
        question="What were the hazard ratios in the DACHS validation cohort for overall survival, CRC-specific OS, and relapse-free survival?",
        reference_answer="In the DACHS validation cohort: OS HR=1.63 (95% CI 1.14-2.33, p=0.008), CRC-specific OS HR=2.29 (95% CI 1.5-3.48, p=0.0004), and RFS HR=1.92 (95% CI 1.34-2.76, p=0.0004).",
        paper_source=PAPER1,
        question_type="numerical", difficulty="hard",
        relevant_phrases=["DACHS", "HR 1.63", "HR 2.29", "HR 1.92", "relapse-free"],
        relevant_sections=["Abstract"],
        must_contain_keywords=["1.63", "2.29", "1.92", "DACHS"],
    ),

    # ── Comparative (2) ──
    EvalItem(
        id="p1_q07",
        question="How does the deep stroma score compare to the CAF gene expression signature as a prognostic marker?",
        reference_answer="The deep stroma score was an independent prognostic factor across all disease stages in the TCGA cohort, while the CAF gene expression signature was only prognostic in specific tumor stages. The neural network-based score outperformed the manual quantification of stromal areas in predicting overall survival.",
        paper_source=PAPER1,
        question_type="comparative", difficulty="hard",
        relevant_phrases=["CAF", "cancer-associated fibroblast", "deep stroma score", "prognostic in specific",
                          "compared"],
        relevant_sections=["Neural network assessment of the stromal"],
        must_contain_keywords=["deep stroma score", "CAF", "prognostic"],
    ),
    EvalItem(
        id="p1_q08",
        question="What are the differences between manual stromal quantification and the deep stroma score in prognostic power?",
        reference_answer="Manual quantification of stromal areas was not significantly prognostic in the TCGA cohort (alongside stage and grade), whereas the deep stroma score derived from CNN output neuron activations was an independent prognostic factor. This demonstrates that deep learning can extract more subtle and prognostically relevant information from histology images than manual assessment.",
        paper_source=PAPER1,
        question_type="comparative", difficulty="medium",
        relevant_phrases=["manual quantification", "not significant", "deep stroma score", "independent prognostic"],
        relevant_sections=["Abstract", "Neural network assessment"],
        must_contain_keywords=["manual", "stroma", "deep stroma score", "prognostic"],
    ),

    # ── Multi-hop (3) ──
    EvalItem(
        id="p1_q09",
        question="What evidence supports the claim that the deep stroma score is an independent prognostic factor beyond TNM staging?",
        reference_answer="Two lines of evidence: 1) In the TCGA cohort, the deep stroma score remained significant in multivariable Cox regression that included UICC stage (HR 1.99, p=0.0028). 2) In the independent DACHS cohort, it was again significant in multivariable analysis for OS (HR 1.63, p=0.008), CRC-specific OS (HR 2.29, p=0.0004), and RFS (HR 1.92, p=0.0004), confirming independence from standard staging.",
        paper_source=PAPER1,
        question_type="multi_hop", difficulty="hard",
        relevant_phrases=["multivariable Cox", "UICC", "independent prognostic", "DACHS", "TCGA"],
        relevant_sections=["Abstract", "Neural network assessment", "Deep stroma score generalizes"],
        must_contain_keywords=["multivariable", "independent", "UICC", "DACHS"],
    ),
    EvalItem(
        id="p1_q10",
        question="How were the optimal cutoffs for the deep stroma score determined and validated?",
        reference_answer="Optimal cutoffs were determined using the Youden index on the TCGA cohort, then validated externally on the DACHS cohort. The score was dichotomized into high vs low stromal score groups, and Kaplan-Meier analysis with log-rank test was used to confirm the prognostic separation.",
        paper_source=PAPER1,
        question_type="multi_hop", difficulty="medium",
        relevant_phrases=["cutoff", "Youden", "optimal", "threshold", "Kaplan-Meier", "log-rank"],
        relevant_sections=["Deep stroma score"],
        must_contain_keywords=["Youden", "cutoff", "Kaplan-Meier"],
    ),
    EvalItem(
        id="p1_q11",
        question="What are the main limitations of this study and what additional validation is needed?",
        reference_answer="Limitations include: retrospective design (needs prospective validation), proof-of-concept nature, training on manually extracted tissue regions rather than whole-slide images, and the need for validation in more diverse populations. The authors explicitly state that 'a prospective validation is required before this biomarker can be implemented in clinical workflows.'",
        paper_source=PAPER1,
        question_type="multi_hop", difficulty="medium",
        relevant_phrases=["limitation", "prospective validation", "proof of concept", "retrospective", "manually extracted"],
        relevant_sections=["Discussion"],
        must_contain_keywords=["prospective", "validation", "limitation"],
    ),

    # ── Summary (2) ──
    EvalItem(
        id="p1_q12",
        question="Summarize the main workflow of this study from tissue images to prognostic prediction.",
        reference_answer="The workflow: 1) Manual delineation of 9 tissue types in 86 CRC slides producing >100,000 image patches, 2) Training VGG19 CNN by transfer learning from ImageNet, 3) Automated tissue decomposition of 862 HE slides from TCGA, 4) Calculating deep stroma score from CNN output activations, 5) Validation in independent DACHS cohort (409 patients).",
        paper_source=PAPER1,
        question_type="summary", difficulty="medium",
        relevant_phrases=["86 CRC", "100,000", "VGG19", "transfer learning", "tissue decomposition", "DACHS"],
        relevant_sections=["Abstract", "Methods"],
        must_contain_keywords=["VGG19", "transfer learning", "tissue decomposition"],
    ),
    EvalItem(
        id="p1_q13",
        question="What is the clinical significance of this research according to the authors?",
        reference_answer="The authors conclude that deep learning can inexpensively predict CRC patient outcomes from ubiquitously available histological images. They emphasize that for every CRC patient, H&E slides are routinely available, making this approach widely applicable. However, they note that prospective validation studies are needed before clinical implementation.",
        paper_source=PAPER1,
        question_type="summary", difficulty="easy",
        relevant_phrases=["inexpensive", "ubiquitously available", "prospective validation", "clinical"],
        relevant_sections=["Author summary", "Conclusions"],
        must_contain_keywords=["deep learning", "prognosis", "histological"],
    ),

    # ── Terminology (2) ──
    EvalItem(
        id="p1_q14",
        question="What does 'tissue decomposition' mean in the context of this study?",
        reference_answer="Tissue decomposition refers to the process where the trained CNN analyzes a multi-tissue H&E image and classifies each region into one of nine tissue classes (adipose, background, debris, lymphocytes, mucus, smooth muscle, normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma epithelium), effectively 'decomposing' the complex tissue image into its constituent parts.",
        paper_source=PAPER1,
        question_type="terminology", difficulty="easy",
        relevant_phrases=["tissue decomposition", "nine tissue", "constituent"],
        relevant_sections=["Abstract", "Training and testing"],
        must_contain_keywords=["tissue decomposition", "nine", "class"],
    ),
    EvalItem(
        id="p1_q15",
        question="What transfer learning approach was used and why?",
        reference_answer="VGG19 pretrained on ImageNet was used via transfer learning. The CNN was fine-tuned on >100,000 manually annotated H&E image patches of 9 tissue classes. Transfer learning allowed leveraging features learned from natural images to overcome limited medical training data, achieving >94% accuracy on an independent test set.",
        paper_source=PAPER1,
        question_type="terminology", difficulty="easy",
        relevant_phrases=["VGG19", "transfer learning", "pretrained", "ImageNet", "fine-tuned"],
        relevant_sections=["Training and testing"],
        must_contain_keywords=["VGG19", "transfer learning", "ImageNet"],
    ),
]

# ═══════════════════════════════════════════════════════════════
# Paper 2: Bokhorst et al. 2023 — Automated Risk Classification Colon Biopsies (15 Qs)
# ═══════════════════════════════════════════════════════════════

PAPER2 = "paper2_Automated_Risk_Classification_Colon_Biopsies.pdf"

_p2_questions = [
    # ── Factoid ──
    EvalItem(
        id="p2_q01",
        question="How many tissue types does the segmentation model in the Bokhorst 2023 Medical Image Analysis paper classify and what are they?",
        reference_answer="The model classifies 14 tissue types: normal epithelium, low-grade dysplastic epithelium, high-grade dysplastic/cancerous epithelium, stroma lamina propria, submucosal stroma, desmoplastic stroma, muscle, nerve, adipose, mucus, necrosis & debris, background, erythrocytes, and lymphocytes.",
        paper_source=PAPER2,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["14 components", "14 different tissue", "normal epithelium", "low-grade dysplastic",
                          "high-grade dysplastic", "lamina propria", "desmoplastic"],
        must_contain_keywords=["14", "tissue", "segmentation", "epithelium"],
    ),
    EvalItem(
        id="p2_q02",
        question="What four loss functions were compared for semantic segmentation?",
        reference_answer="Four loss functions were compared: 1) Categorical Cross-entropy loss, 2) Focal loss, 3) Bi-tempered loss, and 4) Lovasz-softmax loss. All were evaluated using the U-Net encoder-decoder architecture.",
        paper_source=PAPER2,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["Categorical Cross-entropy", "Focal loss", "Bi-tempered loss", "Lovasz-softmax", "four representative"],
        must_contain_keywords=["cross-entropy", "focal", "bi-tempered", "lovasz"],
    ),
    EvalItem(
        id="p2_q03",
        question="What four diagnostic categories does the CAD system classify colon biopsies into?",
        reference_answer="The four categories are: 1) high-risk (tumor and high-grade dysplasia), 2) low-grade dysplasia, 3) hyperplasia, and 4) benign conditions (normal tissue).",
        paper_source=PAPER2,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["four main categories", "high-risk", "low-grade dysplasia", "hyperplasia", "benign"],
        must_contain_keywords=["four", "categories", "high-risk", "low-grade dysplasia"],
    ),

    # ── Numerical ──
    EvalItem(
        id="p2_q04",
        question="How many patients were in the independent validation cohort for the CAD classification system?",
        reference_answer="The CAD classification system was validated on an independent cohort of more than 1,000 patients.",
        paper_source=PAPER2,
        question_type="numerical", difficulty="easy",
        relevant_phrases=["more than 1,000 patients", "independent cohort", "> 1000"],
        must_contain_keywords=["1,000", "1000", "patients"],
    ),
    EvalItem(
        id="p2_q05",
        question="How many medical centers contributed to the multi-centric validation dataset and what was the sample size?",
        reference_answer="The multi-centric dataset contained n=27 whole-slide images from n=5 different medical centers in the Netherlands and Germany. The training dataset consisted of n=52 CRC surgical resections or biopsies from a single center.",
        paper_source=PAPER2,
        question_type="numerical", difficulty="medium",
        relevant_phrases=["n=52", "n=27", "five", "5 different medical"],
        must_contain_keywords=["52", "27", "5", "medical centers"],
    ),

    # ── Comparative ──
    EvalItem(
        id="p2_q06",
        question="Which loss function performed best for histopathology segmentation and why?",
        reference_answer="The Lovasz-softmax loss performed best as it is a differentiable surrogate for the Jaccard index (IoU) and is well-suited for multi-class segmentation with imbalanced classes. Categorical cross-entropy performed suboptimally due to class imbalance issues where small tissue compartments (like erythrocytes) were dominated by larger ones (like muscle).",
        paper_source=PAPER2,
        question_type="comparative", difficulty="hard",
        relevant_phrases=["Lovasz-softmax", "best performing", "cross-entropy", "suboptimally", "class imbalance"],
        must_contain_keywords=["Lovasz", "loss", "segmentation"],
    ),
    EvalItem(
        id="p2_q07",
        question="How does this work differ from Korbar et al.'s previous study on colon biopsy classification?",
        reference_answer="Korbar et al. focused on classifying five non-neoplastic conditions (hyperplastic, sessile serrated, traditional serrated, tubular, tubulovillous/villous), whereas Bokhorst et al. addresses both neoplastic and non-neoplastic categories (high-risk tumor/HGD, low-grade dysplasia, hyperplasia, benign) and includes a full semantic segmentation pipeline with 14 tissue classes as an intermediate step.",
        paper_source=PAPER2,
        question_type="comparative", difficulty="hard",
        relevant_phrases=["Korbar", "five non-neoplastic", "non-neoplastic conditions"],
        must_contain_keywords=["Korbar", "non-neoplastic", "classification"],
    ),

    # ── Multi-hop ──
    EvalItem(
        id="p2_q08",
        question="What is the complete pipeline from input WSI to final biopsy classification?",
        reference_answer="1) WSI is tiled into 512x512μm patches, 2) U-Net segmentation model classifies each pixel into 14 tissue types, 3) From the resulting segmentation map, features are extracted: normalized histogram of tissue types, number/size of HGD/tumor clusters, 4) A random forest classifier uses these features to make the final 4-class prediction (high-risk, low-grade dysplasia, hyperplasia, benign).",
        paper_source=PAPER2,
        question_type="multi_hop", difficulty="medium",
        relevant_phrases=["segmentation map", "random forest", "normalized histogram", "clusters", "512x512"],
        must_contain_keywords=["segmentation", "random forest", "features"],
    ),
    EvalItem(
        id="p2_q09",
        question="Why is class imbalance a specific problem in histopathology image segmentation and how is it addressed?",
        reference_answer="Small tissue compartments (e.g., erythrocytes, lymphocytes) need to be correctly segmented alongside much larger components (e.g., muscle, stroma). Standard categorical cross-entropy is biased toward the over-represented class. This is addressed by: using weighted loss functions (Focal loss reduces contribution of easy examples), Bi-tempered loss (handles noise), and Lovasz-softmax (directly optimizes IoU which is insensitive to class size).",
        paper_source=PAPER2,
        question_type="multi_hop", difficulty="hard",
        relevant_phrases=["class imbalance", "over-represented", "under-represented", "focal loss", "penalty"],
        must_contain_keywords=["class imbalance", "focal loss", "cross-entropy"],
    ),

    # ── Summary ──
    EvalItem(
        id="p2_q10",
        question="What is the clinical motivation for developing automated colon biopsy classification?",
        reference_answer="Population screening programs in Europe target >110 million people annually, with ~5% requiring follow-up colonoscopy. This creates a massive increase in biopsies for pathologists. An automated system can pre-screen cases, prioritize high-risk biopsies, reduce pathologist workload, and assist in the objective quantification of tissue compartments for prognostic biomarkers like tumor-stroma ratio and tumor budding.",
        paper_source=PAPER2,
        question_type="summary", difficulty="medium",
        relevant_phrases=["110 million", "population screening", "burden", "pre-screening", "workload"],
        must_contain_keywords=["screening", "population", "pathologist", "workload"],
    ),
    EvalItem(
        id="p2_q11",
        question="What are the key contributions of the Bokhorst 2023 study?",
        reference_answer="1) Multi-class semantic segmentation of 14 tissue types in CRC H&E WSI, 2) Systematic comparison of 4 loss functions for histopathology segmentation, 3) Development of a CAD system for 4-class biopsy risk classification validated on >1,000 patients, 4) Making the segmentation model publicly available on Grand Challenge platform.",
        paper_source=PAPER2,
        question_type="summary", difficulty="easy",
        relevant_phrases=["contribution", "14 components", "loss function", "CAD", "Grand Challenge"],
        must_contain_keywords=["segmentation", "classification", "loss function"],
    ),

    # ── Terminology ──
    EvalItem(
        id="p2_q12",
        question="What is the Lovasz-softmax loss and why is it useful?",
        reference_answer="Lovasz-softmax is a differentiable surrogate for the Jaccard index (intersection-over-union metric). It's useful because it directly optimizes the IoU metric rather than pixel-wise accuracy, making it robust to class imbalance. It's based on the Lovasz extension of submodular set functions.",
        paper_source=PAPER2,
        question_type="terminology", difficulty="medium",
        relevant_phrases=["Lovasz-softmax", "differentiable", "Jaccard", "surrogate"],
        must_contain_keywords=["Lovasz", "Jaccard", "differentiable"],
    ),
    EvalItem(
        id="p2_q13",
        question="What is the U-Net architecture and why is it popular for medical image segmentation?",
        reference_answer="U-Net is an encoder-decoder CNN architecture with skip connections between corresponding encoder and decoder layers. It's popular for medical segmentation because skip connections preserve fine spatial details lost during downsampling, making it effective even with limited training data. The architecture was originally proposed by Ronneberger et al. (2015).",
        paper_source=PAPER2,
        question_type="terminology", difficulty="easy",
        relevant_phrases=["U-Net", "encoder-decoder", "skip", "Ronneberger"],
        must_contain_keywords=["U-Net", "encoder", "decoder"],
    ),

    # Additional P2 questions
    EvalItem(
        id="p2_q14",
        question="What public datasets were used for benchmarking the segmentation model?",
        reference_answer="Two publicly available datasets were used: GLAS (Gland Segmentation challenge) and CRAG (Colorectal Adenocarcinoma Gland challenge), both focused on gland segmentation in CRC histology images.",
        paper_source=PAPER2,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["GLAS", "CRAG", "publicly available", "gland"],
        must_contain_keywords=["GLAS", "CRAG", "public"],
    ),
    EvalItem(
        id="p2_q15",
        question="How were features extracted from the segmentation map for the random forest classifier?",
        reference_answer="Three types of features were extracted: 1) normalized histogram of all 14 tissue types (tissue composition percentages), 2) number of high-grade dysplasia/tumor clusters, and 3) average, minimum, and maximum size of these clusters. These features capture both tissue composition and architectural patterns.",
        paper_source=PAPER2,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["normalized histogram", "number of", "clusters", "average", "minimum", "maximum"],
        must_contain_keywords=["histogram", "clusters", "tumor"],
    ),
]

# ═══════════════════════════════════════════════════════════════
# Paper 3: Schoenpflug et al. 2023 — SemiCOL Multi-Task Learning (15 Qs)
# ═══════════════════════════════════════════════════════════════

PAPER3 = "paper3_MultiTask_SemiCOL_CRC_Histology.pdf"

_p3_questions = [
    EvalItem(
        id="p3_q01",
        question="What is the SemiCOL challenge and what tasks does it involve?",
        reference_answer="The SemiCOL (Semi-supervised learning for CRC detection) challenge 2023 provides partially annotated data for two tasks: tissue segmentation and tumor detection in H&E-stained CRC histology slides. It includes a small set with segmentation annotations and a larger set with only weak slide-level labels (tumor present: yes/no).",
        paper_source=PAPER3,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["SemiCOL", "partially annotated", "segmentation", "tumor detection"],
        must_contain_keywords=["SemiCOL", "challenge", "segmentation", "tumor detection"],
    ),
    EvalItem(
        id="p3_q02",
        question="What model architecture was used and what were its two branches?",
        reference_answer="A U-Net-based multi-task model with two branches: a decoder segmentation head for tissue segmentation and a fully connected classifier head for tumor detection. The classifier head is only used during training to leverage weakly annotated samples; during inference, only the segmentation branch is used.",
        paper_source=PAPER3,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["multi-task", "U-Net", "segmentation head", "classifier head", "fully connected"],
        must_contain_keywords=["U-Net", "multi-task", "segmentation", "classifier"],
    ),
    EvalItem(
        id="p3_q03",
        question="What were the final performance scores on the challenge validation set?",
        reference_answer="Arm 1: multi-task Dice score of 0.8655 for tissue segmentation and AUROC of 0.9725 for tumor detection. Arm 2: multi-task Dice score of 0.8515 and AUROC of 0.9750.",
        paper_source=PAPER3,
        question_type="numerical", difficulty="medium",
        relevant_phrases=["0.8655", "0.8515", "0.9725", "0.9750", "Dice score", "AUROC"],
        must_contain_keywords=["0.8655", "0.9725", "Dice", "AUROC"],
    ),
    EvalItem(
        id="p3_q04",
        question="What two color augmentation methods were investigated for domain generalization?",
        reference_answer="1) Channel-wise brightness and contrast variation by ±20%: represents many possible colors and shades, forcing the model to learn morphological features invariant to color variations. 2) Image-statistics-based color augmentation: swaps the low-frequency component (mean pixel value) between input and reference images from different scanner/institution domains, including references from both SemiCOL and MIDOG challenge datasets.",
        paper_source=PAPER3,
        question_type="comparative", difficulty="hard",
        relevant_phrases=["channel-wise", "brightness and contrast", "image-statistics", "mean", "MIDOG"],
        must_contain_keywords=["color augmentation", "channel-wise", "image-statistics"],
    ),
    EvalItem(
        id="p3_q05",
        question="How was the tumor detection score computed from the segmentation output?",
        reference_answer="Tumor detection score = (tumor + tumor stroma + ulcus necrosis) / (tumor + tumor stroma + ulcus necrosis + benign mucosa + submucosa), where each term refers to the number of predicted pixels for that class.",
        paper_source=PAPER3,
        question_type="numerical", difficulty="hard",
        relevant_phrases=["tumor detection score", "tumor + tumor stroma + ulcus necrosis", "benign mucosa + submucosa"],
        must_contain_keywords=["tumor detection score", "pixels", "necrosis"],
    ),
    EvalItem(
        id="p3_q06",
        question="What is the difference between Arm 1 and Arm 2 in the SemiCOL challenge?",
        reference_answer="Arm 1 only allows methods that use datasets provided by the SemiCOL challenge. Arm 2 allows using additional external data. The authors added MIDOG challenge 2022 references for image-statistics-based color augmentation in Arm 2.",
        paper_source=PAPER3,
        question_type="terminology", difficulty="easy",
        relevant_phrases=["Arm 1", "Arm 2", "only utilize the datasets provided", "additional data"],
        must_contain_keywords=["Arm 1", "Arm 2", "SemiCOL"],
    ),
    EvalItem(
        id="p3_q07",
        question="Which augmentation method contributed the most to performance improvement and how much?",
        reference_answer="Channel-wise color augmentation had the strongest effect, increasing multi-class Dice score by 0.0943 and AUROC by 0.21 on the external validation set. This reflects the effectiveness of color augmentation for generalization to different scanner types and staining protocols.",
        paper_source=PAPER3,
        question_type="numerical", difficulty="medium",
        relevant_phrases=["strongest", "0.0943", "0.21", "channel-wise", "color augmentation"],
        must_contain_keywords=["channel-wise", "0.0943", "improvement"],
    ),
    EvalItem(
        id="p3_q08",
        question="How was the training dataset prepared from the SemiCOL data?",
        reference_answer="The SemiCOL training set was split 80:20 (slide-level, stratified by domain and annotation). Segmentation set: 59,611 patches from 16 WSIs (training) and 12,834 patches from 4 WSIs (validation). Weakly annotated set: 399 WSIs (training) and 100 WSIs (validation). Images were downscaled to 5x magnification and tiled at 300x300 pixels. Segmentation tiles used 50% overlap (kept if ≥1% annotated pixels). Weakly annotated tiles had no overlap (kept if ≥50% tissue).",
        paper_source=PAPER3,
        question_type="multi_hop", difficulty="hard",
        relevant_phrases=["80:20", "59,611", "12,834", "399", "100", "5x", "300x300", "50% overlap"],
        must_contain_keywords=["59,611", "300x300", "overlap"],
    ),
    EvalItem(
        id="p3_q09",
        question="How does the multi-task model utilize weakly annotated data during training?",
        reference_answer="Each training batch is balanced between segmentation and weakly annotated samples. Tiles without segmentation annotation are ignored for the segmentation loss. Tiles with segmentation annotation get a tumor label if tumor tissue class is present. The model uses a weighted multi-task loss: Loss = w*CEsgm + (1-w)*BCEtd, where w=0.5, combining cross-entropy for segmentation and binary cross-entropy for tumor detection.",
        paper_source=PAPER3,
        question_type="multi_hop", difficulty="hard",
        relevant_phrases=["weighted multi-task loss", "0.5", "balanced", "weakly annotated"],
        must_contain_keywords=["multi-task loss", "cross-entropy", "weakly"],
    ),
    EvalItem(
        id="p3_q10",
        question="What training hyperparameters were used for the baseline model?",
        reference_answer="100 epochs, batch size 128, SGD with Nesterov momentum (lr=0.2, momentum=0.9, weight decay=5e-6), exponential learning rate decay (gamma=0.97). Geometric augmentations: flipping, transposing, random rotation (90°, 180°, 270°), scale variation ±10%, random crop to 260x260 pixels, all applied with 70% probability.",
        paper_source=PAPER3,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["100 epochs", "128", "SGD", "Nesterov", "0.2", "0.9", "0.97"],
        must_contain_keywords=["SGD", "Nesterov", "learning rate", "epochs"],
    ),
    EvalItem(
        id="p3_q11",
        question="What suggestions were made for future improvement?",
        reference_answer="1) Extension of the training set with active learning where additional annotations are provided by an expert pathologist, 2) Curating the tiles selected from the weakly annotated set instead of random sampling, potentially focusing on more informative or difficult examples.",
        paper_source=PAPER3,
        question_type="summary", difficulty="easy",
        relevant_phrases=["future work", "active learning", "curating", "expert pathologist"],
        must_contain_keywords=["active learning", "future", "expert"],
    ),
    EvalItem(
        id="p3_q12",
        question="What was the impact of adding the tumor detection branch compared to segmentation-only training?",
        reference_answer="Adding the tumor detection branch improved multi-class Dice score by 0.0997 and AUROC by 0.0975 on the external validation set, demonstrating that leveraging weakly annotated data through multi-task learning provides a moderate generalization improvement.",
        paper_source=PAPER3,
        question_type="numerical", difficulty="medium",
        relevant_phrases=["0.0997", "0.0975", "tumor detection branch", "moderate generalization"],
        must_contain_keywords=["tumor detection", "branch", "improvement"],
    ),
    EvalItem(
        id="p3_q13",
        question="Explain the three levels of performance evaluation: internal validation vs external validation vs test set.",
        reference_answer="Internal validation: 80:20 split of the SemiCOL training set, used for hyperparameter selection. External validation: the official SemiCOL validation set with 858 patches from 9 WSIs (segmentation) and 40 WSIs (detection), used for model selection and submission. Test set: the held-out SemiCOL challenge test set (not yet available at the time of writing), for final ranking.",
        paper_source=PAPER3,
        question_type="terminology", difficulty="medium",
        relevant_phrases=["internal validation", "external validation", "test set", "80:20", "858", "40"],
        must_contain_keywords=["internal", "external", "validation", "test"],
    ),
    EvalItem(
        id="p3_q14",
        question="What is test-time augmentation and what improvement did it provide?",
        reference_answer="Test-time augmentation (TTA) applies all 8 possible rotation and flip combinations during inference and aggregates predictions for more robust results. It provided a small additional improvement of approximately 0.01 for multi-class Dice score in both Arm 1 and Arm 2, while AUROC stayed approximately the same.",
        paper_source=PAPER3,
        question_type="terminology", difficulty="easy",
        relevant_phrases=["test-time augmentation", "8 possible rotation and flip", "0.01", "approximately"],
        must_contain_keywords=["test-time", "augmentation", "rotation", "flip"],
    ),
    EvalItem(
        id="p3_q15",
        question="Why is domain generalization important in histopathology and how did this study address it?",
        reference_answer="Histopathology images vary significantly across different scanners, staining protocols, and institutions (domains), causing models trained on one domain to perform poorly on others. This study addressed it through: 1) channel-wise color augmentation to simulate staining variation, 2) image-statistics-based augmentation using reference means from multiple scanner/institution domains (SemiCOL + MIDOG datasets), forcing the model to learn domain-invariant morphological features rather than relying on color cues.",
        paper_source=PAPER3,
        question_type="summary", difficulty="medium",
        relevant_phrases=["domain generalization", "different scanners", "staining protocols", "invariant"],
        must_contain_keywords=["domain", "generalization", "scanner", "staining"],
    ),
]

# ═══════════════════════════════════════════════════════════════
# Paper 4: Gülmez 2025 — DL CRC Detection Comprehensive Review (15 Qs)
# ═══════════════════════════════════════════════════════════════

PAPER4 = "paper4_CRC_Deep_Learning_Medical_Images_2025.pdf"

_p4_questions = [
    EvalItem(
        id="p4_q01",
        question="How many publications and datasets were analyzed in the Gülmez 2025 review?",
        reference_answer="The review analyzed 110 high-quality publications and 9 publicly accessible medical image datasets. The initial search yielded 384 articles, which were filtered to 247 after title/abstract screening, and 110 after full-text review.",
        paper_source=PAPER4,
        question_type="numerical", difficulty="easy",
        relevant_phrases=["110 high-quality", "9 publicly accessible", "384", "247"],
        must_contain_keywords=["110", "publications", "datasets"],
    ),
    EvalItem(
        id="p4_q02",
        question="What CNN architectures were most commonly used according to this review?",
        reference_answer="ResNet (40 implementations), VGG (18 implementations), and emerging transformer-based models (12 implementations). These were used across classification, object detection, and segmentation tasks.",
        paper_source=PAPER4,
        question_type="numerical", difficulty="easy",
        relevant_phrases=["ResNet (40", "VGG (18", "transformer-based (12", "40 implementations",
                          "18 implementations", "12 implementations"],
        must_contain_keywords=["ResNet", "40", "VGG", "18", "transformer", "12"],
    ),
    EvalItem(
        id="p4_q03",
        question="What explainable AI methods are discussed for medical diagnosis interpretation?",
        reference_answer="Grad-CAM and SHAP are the visualization techniques analyzed for explainable AI in medical diagnosis interpretation.",
        paper_source=PAPER4,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["Grad-CAM", "SHAP", "explainable AI", "visualization"],
        must_contain_keywords=["Grad-CAM", "SHAP", "explainable"],
    ),
    EvalItem(
        id="p4_q04",
        question="What optimization techniques were highlighted for DL model performance improvement?",
        reference_answer="Genetic algorithms and particle swarm optimization approaches were highlighted as hyperparameter optimization techniques to enhance model performance.",
        paper_source=PAPER4,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["genetic algorithms", "particle swarm optimization", "hyperparameter optimization"],
        must_contain_keywords=["genetic algorithm", "particle swarm", "optimization"],
    ),
    EvalItem(
        id="p4_q05",
        question="What search strategy and databases were used for the literature review?",
        reference_answer="Scopus, Web of Science, IEEE Xplore, and PubMed were searched using Boolean combinations: ('colorectal cancer' OR 'colon cancer') AND ('deep learning' OR 'convolutional neural network' OR 'artificial intelligence') AND ('detection' OR 'classification' OR 'segmentation') AND ('medical imaging' OR 'endoscopy' OR 'histopathology'). The search period was January 2019 through March 2025.",
        paper_source=PAPER4,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["Scopus", "Web of Science", "IEEE Xplore", "PubMed", "Boolean", "2019", "2025"],
        must_contain_keywords=["Scopus", "PubMed", "search", "2019", "2025"],
    ),
    EvalItem(
        id="p4_q06",
        question="What are the five novel aspects that distinguish this review from previous ones?",
        reference_answer="1) Quantitative trend analysis of publication patterns across geographical regions and time periods, 2) Comprehensive categorization and comparative assessment of DL architectures (110 publications), 3) Identification of the transition from CNNs to transformer-based models, 4) Analysis of the relationship between dataset characteristics and model performance, 5) Data-driven identification of research gaps based on comprehensive bibliometric analysis.",
        paper_source=PAPER4,
        question_type="summary", difficulty="hard",
        relevant_phrases=["novel aspects", "quantitative trend", "comprehensive categorization", "transformer-based",
                          "bibliometric"],
        must_contain_keywords=["novel", "quantitative", "transition", "transformer"],
    ),
    EvalItem(
        id="p4_q07",
        question="What are the main technical limitations identified in current CRC detection research?",
        reference_answer="Three main technical limitations: 1) Dataset scarcity — limited availability of large, diverse, annotated medical image datasets, 2) Computational constraints — training deep models requires significant GPU resources, 3) Standardization challenges — lack of standardized evaluation protocols and benchmarks across studies.",
        paper_source=PAPER4,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["dataset scarcity", "computational constraints", "standardization challenges"],
        must_contain_keywords=["dataset scarcity", "computational", "standardization"],
    ),
    EvalItem(
        id="p4_q08",
        question="What future research directions are proposed in the review?",
        reference_answer="Multimodal learning (combining imaging with genomic/clinical data) and federated learning (privacy-preserving distributed training across institutions) are proposed as future directions, based on publication trend analysis.",
        paper_source=PAPER4,
        question_type="summary", difficulty="easy",
        relevant_phrases=["multimodal learning", "federated learning", "future", "directions"],
        must_contain_keywords=["multimodal", "federated learning", "future"],
    ),
    EvalItem(
        id="p4_q09",
        question="According to the review, approximately how many new CRC cases and deaths occurred globally in 2020?",
        reference_answer="In 2020, approximately 1.9 million new CRC cases were diagnosed globally with 935,000 deaths. CRC accounts for about 10% of all cancer diagnoses and 9% of cancer-related deaths worldwide.",
        paper_source=PAPER4,
        question_type="numerical", difficulty="easy",
        relevant_phrases=["1.9 million", "10%", "9%"],
        must_contain_keywords=["1.9 million", "cancer", "deaths"],
    ),
    EvalItem(
        id="p4_q10",
        question="What inclusion and exclusion criteria were used for study selection?",
        reference_answer="Inclusion: January 2019-March 2025, peer-reviewed, AI-based CRC detection with medical images, original research with quantitative evaluation, English. Exclusion: reviews/editorials/letters, no experimental validation, insufficient technical details, non-peer-reviewed, general cancer detection without CRC focus, clinical-only without computational methods, duplicates.",
        paper_source=PAPER4,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["inclusion criteria", "exclusion criteria", "January 2019", "March 2025", "peer-reviewed"],
        must_contain_keywords=["inclusion", "exclusion", "criteria"],
    ),
    EvalItem(
        id="p4_q11",
        question="What is the significance of CNNs in colorectal cancer detection?",
        reference_answer="CNNs automatically extract relevant features from images without manual feature engineering. They have been employed to analyze histopathological images, enabling classification and segmentation of cancerous tissues with high accuracy, and studies show CNNs can outperform traditional diagnostic methods.",
        paper_source=PAPER4,
        question_type="terminology", difficulty="easy",
        relevant_phrases=["automatically extract", "feature engineering", "outperform traditional"],
        must_contain_keywords=["CNN", "feature", "classification", "segmentation"],
    ),
    EvalItem(
        id="p4_q12",
        question="What are the estimated US CRC statistics cited for 2021?",
        reference_answer="The American Cancer Society estimated about 104,000 new cases of colon cancer and 45,000 new cases of rectal cancer in the United States in 2021.",
        paper_source=PAPER4,
        question_type="numerical", difficulty="medium",
        relevant_phrases=["104,000", "45,000", "United States", "2021", "American Cancer Society"],
        must_contain_keywords=["104,000", "45,000", "colon", "rectal"],
    ),
    EvalItem(
        id="p4_q13",
        question="What is the range of imaging modalities reviewed for CRC detection?",
        reference_answer="The review covers medical imaging modalities including endoscopy images, histopathology whole-slide images, CT scans, and MRI. The search explicitly included terms for endoscopy, histopathology, and medical imaging.",
        paper_source=PAPER4,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["endoscopy", "histopathology", "CT", "MRI", "imaging modalities"],
        must_contain_keywords=["endoscopy", "histopathology", "CT", "MRI"],
    ),
    EvalItem(
        id="p4_q14",
        question="Why did the authors choose to focus on 2019-2025 for the review period?",
        reference_answer="The rapid evolution of deep learning architectures, particularly the emergence of transformer-based models, necessitated a focus on recent advances (January 2019 through March 2025). This period captures the transition from traditional CNNs to modern architectures in medical image analysis.",
        paper_source=PAPER4,
        question_type="multi_hop", difficulty="medium",
        relevant_phrases=["2019", "2025", "rapid evolution", "transformer", "transition"],
        must_contain_keywords=["2019", "2025", "review period"],
    ),
    EvalItem(
        id="p4_q15",
        question="How does this review bridge technical and clinical perspectives?",
        reference_answer="By synthesizing both quantitative performance metrics of DL architectures and clinical applicability considerations, the review provides a resource that bridges technical and clinical perspectives. It categorizes implementations by architecture type while discussing clinical relevance through explainable AI (Grad-CAM, SHAP), dataset limitations, and standardization challenges.",
        paper_source=PAPER4,
        question_type="summary", difficulty="hard",
        relevant_phrases=["bridges technical and clinical", "resource", "computer scientists", "medical practitioners"],
        must_contain_keywords=["bridge", "technical", "clinical", "comprehensive"],
    ),
]

# ═══════════════════════════════════════════════════════════════
# Paper 5: Bokhorst et al. 2023 SciRep — Multi-class Segmentation CRC (15 Qs)
# ═══════════════════════════════════════════════════════════════

PAPER5 = "paper5_DL_MultiClass_Segmentation_CRC_SciRep2023.pdf"

_p5_questions = [
    EvalItem(
        id="p5_q01",
        question="How many tissue compartments does the Bokhorst 2023 Scientific Reports segmentation model classify?",
        reference_answer="The model classifies 14 tissue compartments in H&E-stained whole-slide images of CRC, including normal vs low-grade dysplastic vs high-grade dysplastic/cancerous epithelium, various stroma types (lamina propria, submucosal, desmoplastic), muscle, nerve, adipose, mucus, necrosis & debris, and background.",
        paper_source=PAPER5,
        question_type="numerical", difficulty="easy",
        relevant_phrases=["14 tissue compartments", "fourteen different tissue", "n = 14"],
        must_contain_keywords=["14", "tissue", "compartments"],
    ),
    EvalItem(
        id="p5_q02",
        question="What four risk categories does the computer-aided diagnosis system classify biopsies into?",
        reference_answer="The four categories: 1) high-risk (tumor and high-grade dysplasia), 2) low-grade dysplasia, 3) hyperplasia, and 4) benign conditions.",
        paper_source=PAPER5,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["high-risk (tumor", "low-grade dysplasia", "hyperplasia", "benign"],
        must_contain_keywords=["high-risk", "low-grade dysplasia", "hyperplasia", "benign"],
    ),
    EvalItem(
        id="p5_q03",
        question="How was the segmentation model validated across different centers and scanners?",
        reference_answer="The model was validated on multi-centric data from five different medical centers in the Netherlands and Germany, with slides digitized using three different types of digital pathology scanners, ensuring robustness to staining variation and scanner differences.",
        paper_source=PAPER5,
        question_type="multi_hop", difficulty="medium",
        relevant_phrases=["five different medical centers", "three different", "scanners", "multi-centric"],
        must_contain_keywords=["multi-centric", "scanners", "medical centers"],
    ),
    EvalItem(
        id="p5_q04",
        question="How many patients were used for developing the segmentation algorithm (Dseg dataset)?",
        reference_answer="n=79 formalin-fixed paraffin-embedded tissue samples (surgical resections and biopsies) collected from four Dutch medical centers and one German medical center. All slides were H&E-stained in each center's pathology laboratory.",
        paper_source=PAPER5,
        question_type="numerical", difficulty="medium",
        relevant_phrases=["n = 79", "four Dutch", "one German", "formalin-fixed paraffin-embedded"],
        must_contain_keywords=["79", "formalin-fixed", "multi-centric"],
    ),
    EvalItem(
        id="p5_q05",
        question="What prognostic biomarkers in CRC can be assessed using the segmentation model?",
        reference_answer="The segmentation model can quantify: tumor-stroma ratio, tumor budding (small tumor clusters at invasive margin), tumor deposits (discrete cancer nodules in adipose tissue), and peri-neural invasion. These biomarkers currently rely on visual estimation by pathologists and suffer from subjectivity.",
        paper_source=PAPER5,
        question_type="summary", difficulty="medium",
        relevant_phrases=["tumor-stroma ratio", "tumor budding", "tumor deposits", "peri-neural"],
        must_contain_keywords=["tumor-stroma", "tumor budding", "biomarker"],
    ),
    EvalItem(
        id="p5_q06",
        question="What are the four grades of glandular formation from normal to cancer?",
        reference_answer="1) Normal glands: small, organized nuclei and round lumen. 2) Hyperplastic gland: small nuclei, saw-tooth like formed lumen. 3) Low-grade dysplasia: unorganized, stacked epithelium cells possibly with enlarged nuclei. 4) High-grade dysplasia/tumor: unorganized fusing glands that oppress the lumen.",
        paper_source=PAPER5,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["Normal glands", "Hyperplastic", "Low-grade dysplasia", "High-grade dysplasia"],
        must_contain_keywords=["normal gland", "hyperplastic", "dysplasia", "tumor"],
    ),
    EvalItem(
        id="p5_q07",
        question="What is the difference between this 2023 Scientific Reports paper and the 2023 Medical Image Analysis paper by the same first author?",
        reference_answer="Both papers present the same segmentation and classification approach. The Scientific Reports paper (Bokhorst et al. 2023) provides broader context on CRC diagnosis, detailed discussion of prognostic biomarkers, and potential integration scenarios for AI in pathology workflows. The Medical Image Analysis paper focuses more on technical aspects of loss functions and validation.",
        paper_source=PAPER5,
        question_type="comparative", difficulty="hard",
        relevant_phrases=["prognostic biomarkers", "integration", "scenarios", "workflow"],
        must_contain_keywords=["AI", "pathology", "diagnosis"],
    ),
    EvalItem(
        id="p5_q08",
        question="What scenarios for AI integration in pathology workflows are envisioned?",
        reference_answer="1) AI pre-reads cases, extracts relevant information to pre-fill the pathology report, and shows results to the pathologist who checks and signs off. 2) AI pre-scores cases based on risk and presents them to pathologists in order of diagnostic urgency, allowing prioritization of high-risk cases.",
        paper_source=PAPER5,
        question_type="summary", difficulty="medium",
        relevant_phrases=["pre-read cases", "pre-fill the report", "pre-score", "order of importance", "sign-off"],
        must_contain_keywords=["pre-read", "pre-score", "pathologist", "sign-off"],
    ),
    EvalItem(
        id="p5_q09",
        question="What is the role of cancer-associated stroma in CRC prognosis?",
        reference_answer="Cancer-associated stroma is a key component of the tumor microenvironment that influences tumor progression and therapy response. The tumor-stroma ratio (TSR) is a prognostic biomarker where a high proportion of stroma relative to tumor is associated with worse prognosis. The segmentation model enables objective quantification of stroma, replacing subjective visual estimation.",
        paper_source=PAPER5,
        question_type="multi_hop", difficulty="hard",
        relevant_phrases=["tumor-stroma ratio", "prognostic", "stroma", "microenvironment", "worse prognosis"],
        must_contain_keywords=["tumor-stroma ratio", "stroma", "prognostic"],
    ),
    EvalItem(
        id="p5_q10",
        question="How was the independent validation of the 4-class CAD system performed?",
        reference_answer="The CAD system was validated on an independent external dataset of polyps and biopsies from more than 1,000 patients. This dataset was separate from the Dseg dataset used for segmentation model development.",
        paper_source=PAPER5,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["> 1000 patients", "external dataset", "independent", "polyps and biopsies"],
        must_contain_keywords=["1000", "patients", "external"],
    ),
    EvalItem(
        id="p5_q11",
        question="What role does class imbalance play in segmentation loss function design?",
        reference_answer="Small tissue compartments (e.g., erythrocytes) must be correctly segmented alongside much larger components (e.g., muscle). Standard categorical cross-entropy is biased toward the over-represented class. This is why alternative loss functions (Focal loss, Bi-tempered loss, Lovasz-softmax) that are more robust to class imbalance were investigated.",
        paper_source=PAPER5,
        question_type="terminology", difficulty="medium",
        relevant_phrases=["class imbalance", "over-represented", "cross-entropy", "biased", "loss function"],
        must_contain_keywords=["class imbalance", "loss", "cross-entropy"],
    ),
    EvalItem(
        id="p5_q12",
        question="What are tumor deposits and how can AI assist in their assessment?",
        reference_answer="Tumor deposits are discrete nodules of cancer in pericolic/perirectal fat or adjacent mesentery. Currently assessed by visual inspection by pathologists. AI segmentation can automatically detect and quantify small tumor aggregates in adipose tissue, enabling more reproducible assessment of this prognostic feature.",
        paper_source=PAPER5,
        question_type="terminology", difficulty="medium",
        relevant_phrases=["tumor deposits", "discrete nodule", "pericolic", "adipose"],
        must_contain_keywords=["tumor deposit", "nodule", "adipose"],
    ),
    EvalItem(
        id="p5_q13",
        question="What is tumor budding and why is it prognostically important?",
        reference_answer="Tumor budding refers to small tumor clusters (up to four tumor cells) at the invasive margin of the tumor. It is an established prognostic biomarker in CRC associated with lymph node metastasis and poorer survival. AI-based detection of tumor buds can provide more reproducible quantification than manual counting.",
        paper_source=PAPER5,
        question_type="terminology", difficulty="easy",
        relevant_phrases=["tumor budding", "small tumor clusters", "up to four", "invasive margin"],
        must_contain_keywords=["tumor budding", "clusters", "invasive"],
    ),
    EvalItem(
        id="p5_q14",
        question="How does the study address the problem of histopathology staining variability?",
        reference_answer="By using a multi-centric cohort from five medical centers with H&E staining performed in each center's own laboratory, creating natural staining variability. The use of three different scanner types further ensures the model is robust to technical variation. This real-world heterogeneous data validates the model's generalizability.",
        paper_source=PAPER5,
        question_type="multi_hop", difficulty="medium",
        relevant_phrases=["large variety of staining", "three different", "scanners", "multi-centric"],
        must_contain_keywords=["staining", "scanner", "variability", "multi-centric"],
    ),
    EvalItem(
        id="p5_q15",
        question="What applications beyond diagnosis does the segmentation model enable?",
        reference_answer="Beyond diagnosis: research on the tumor microenvironment (tumor-stroma ratio, peri-neural invasion), histological features for prognosis (tumor shrinkage), identifying immune cells in different tissue compartments (spatial biology), computational biomarker development, and assisting in objective quantification of established prognostic factors.",
        paper_source=PAPER5,
        question_type="summary", difficulty="hard",
        relevant_phrases=["spatial biology", "tumor microenvironment", "tumor shrinkage", "computational biomarkers"],
        must_contain_keywords=["beyond diagnosis", "microenvironment", "biomarker", "spatial"],
    ),
]

# ═══════════════════════════════════════════════════════════════
# Paper 6: Babu et al. 2025 — AI in CRC Management (15 Qs)
# ═══════════════════════════════════════════════════════════════

PAPER6 = "paper6_CRC_AI_Narrative_Review_2025.pdf"

_p6_questions = [
    EvalItem(
        id="p6_q01",
        question="What are the three categories of AI applications in colonoscopy according to this review?",
        reference_answer="Computer-aided detection (CADe) for lesion detection, computer-aided diagnosis (CADx) for lesion characterization/optical biopsy, and computer-aided quality assessment (CADq) for procedure quality monitoring.",
        paper_source=PAPER6,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["CADe", "CADx", "CADq", "computer-aided detection", "diagnosis", "quality assessment"],
        must_contain_keywords=["CADe", "CADx", "CADq"],
    ),
    EvalItem(
        id="p6_q02",
        question="What is the adenoma miss rate during standard colonoscopy and what factors contribute to it?",
        reference_answer="The adenoma miss rate can be as high as 26%. Contributing factors include: failure to recognize small, proximal, non-polypoid lesions due to inexperience, fatigue, or distraction of endoscopists; incomplete exposure of colorectal mucosa; patient factors; and poor bowel preparation.",
        paper_source=PAPER6,
        question_type="numerical", difficulty="medium",
        relevant_phrases=["26%", "adenoma miss rate", "as high as", "inexperience", "fatigue", "distraction"],
        must_contain_keywords=["26%", "adenoma miss rate", "miss"],
    ),
    EvalItem(
        id="p6_q03",
        question="What did the Wallace et al. randomized trial show about AI-reduced miss rates?",
        reference_answer="The Wallace et al. trial (230 participants, tandem colonoscopy design) showed AI reduced the miss rate of colorectal neoplasia by about two times: AMR was 15.5% for AI-first vs 32.4% for non-AI-first colonoscopy. AI particularly helped reduce miss rates for smaller (<5 mm) and non-polypoid lesions in both proximal and distal colon.",
        paper_source=PAPER6,
        question_type="numerical", difficulty="hard",
        relevant_phrases=["Wallace", "230", "15.5%", "32.4%", "two times", "AI first"],
        must_contain_keywords=["Wallace", "15.5%", "32.4%", "miss rate"],
    ),
    EvalItem(
        id="p6_q04",
        question="What is the CAD EYE system and what performance did it achieve?",
        reference_answer="The CAD EYE system (Fujifilm Corporation, Tokyo, Japan) is a CADx system for optical diagnosis of colorectal lesions. Evaluated on 110 lesions, it achieved 81.8% accuracy, 76.3% sensitivity, 96.7% specificity, 98.5% PPV, and 60.4% NPV. Expert performance was 93.6% accuracy, 92.5% sensitivity, 96.7% specificity, 98.7% PPV, and 82.9% NPV.",
        paper_source=PAPER6,
        question_type="numerical", difficulty="hard",
        relevant_phrases=["CAD EYE", "Fujifilm", "81.8%", "76.3%", "96.7%", "93.6%"],
        must_contain_keywords=["CAD EYE", "Fujifilm", "81.8%"],
    ),
    EvalItem(
        id="p6_q05",
        question="What screening methods for CRC are discussed in the review?",
        reference_answer="Non-invasive: stool-based tests (FOBT, FIT, FIT-DNA), blood-based tests (SEPT9 DNA methylation test Epi proColon), imaging methods (colon capsule endoscopy CCE, computed tomographic colonography CTC, double contrast barium enema). Invasive: flexible sigmoidoscopy and colonoscopy (gold standard).",
        paper_source=PAPER6,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["FOBT", "FIT", "FIT-DNA", "SEPT9", "Epi proColon", "CCE", "CTC", "sigmoidoscopy"],
        must_contain_keywords=["FOBT", "FIT", "colonoscopy", "screening"],
    ),
    EvalItem(
        id="p6_q06",
        question="At what age does the USPSTF recommend starting CRC screening and for whom?",
        reference_answer="The USPSTF recommends preventive screening at 45 years of age for individuals without risk factors (Grade B recommendation). Those with family history or genetic predisposition should start earlier.",
        paper_source=PAPER6,
        question_type="numerical", difficulty="easy",
        relevant_phrases=["45 years", "USPSTF", "Grade B", "screening"],
        must_contain_keywords=["45", "USPSTF", "screening"],
    ),
    EvalItem(
        id="p6_q07",
        question="What is the clinical significance of colonoscopy as a CRC screening tool?",
        reference_answer="Colonoscopy is the gold standard screening test because it allows both detection and removal of precursor lesions (adenomas). The adenoma detection rate (ADR) is a key quality indicator, and colonoscopy has been shown to prevent CRC incidence and mortality.",
        paper_source=PAPER6,
        question_type="terminology", difficulty="easy",
        relevant_phrases=["gold standard", "adenoma detection rate", "ADR", "precursor lesions"],
        must_contain_keywords=["gold standard", "colonoscopy", "ADR"],
    ),
    EvalItem(
        id="p6_q08",
        question="What did the Xu et al. multicenter RCT demonstrate about AI-assisted colonoscopy?",
        reference_answer="The Xu et al. multicenter RCT of asymptomatic individuals found that AI-assisted colonoscopy raised the overall adenoma detection rate (ADR), advanced ADR, and ADR of both expert and non-expert attending endoscopists, demonstrating AI's ability to improve performance across all skill levels.",
        paper_source=PAPER6,
        question_type="multi_hop", difficulty="medium",
        relevant_phrases=["Xu et al", "multi centre", "AI-assisted", "ADR", "expert", "non-expert"],
        must_contain_keywords=["Xu", "AI-assisted", "ADR"],
    ),
    EvalItem(
        id="p6_q09",
        question="What role does AI play in surgical management of CRC?",
        reference_answer="AI assists in: preoperative imaging analysis for accurate localization of cancer spread, predicting survival rates/surgical success, identifying dissection planes during surgery, localizing cancer margins, generating real-time intraoperative images, and enabling robotic surgery and minimally invasive laparoscopic techniques.",
        paper_source=PAPER6,
        question_type="summary", difficulty="medium",
        relevant_phrases=["preoperative imaging", "dissection planes", "cancer margins", "robotic surgery", "real-time"],
        must_contain_keywords=["surgery", "robotic", "laparoscopic", "preoperative"],
    ),
    EvalItem(
        id="p6_q10",
        question="What genetic mutations are most commonly involved in CRC pathogenesis?",
        reference_answer="The APC gene (adenomatous polyposis coli) is the most common, followed by KRAS (Ki-ras2 Kirsten rat sarcoma viral oncogene homolog) and TP53 (tumor protein p53). These genetic alterations drive the progression from benign adenomas to malignant carcinomas through the adenoma-carcinoma sequence.",
        paper_source=PAPER6,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["APC", "KRAS", "TP53", "adenomatous polyposis coli", "genetic alteration"],
        must_contain_keywords=["APC", "KRAS", "TP53"],
    ),
    EvalItem(
        id="p6_q11",
        question="What percentage of CRC patients are diagnosed at advanced stages and what is the survival impact?",
        reference_answer="Approximately 60-70% of CRC patients are diagnosed at advanced stages, with liver metastases present in ~20% of cases. Five-year overall survival drops from 80-90% for localized disease to 10-15% for metastatic disease at diagnosis.",
        paper_source=PAPER6,
        question_type="numerical", difficulty="medium",
        relevant_phrases=["60-70%", "advanced stages", "80-90%", "10-15%", "metastatic"],
        must_contain_keywords=["60-70%", "advanced", "survival", "metastatic"],
    ),
    EvalItem(
        id="p6_q12",
        question="What did the Ahmad et al. study find about CADe in high-performing endoscopists?",
        reference_answer="Ahmad et al. found that CADe increased the polyp detection rate (PDR) but NOT the adenoma detection rate in high-performing endoscopists regularly using Endocuff Vision in the NHS Bowel Cancer Screening Program. In a BCSP setting with skilled endoscopists, CADe offered no additional advantage.",
        paper_source=PAPER6,
        question_type="multi_hop", difficulty="hard",
        relevant_phrases=["Ahmad", "Endocuff", "NHS", "BCSP", "no advantage", "high-performing"],
        must_contain_keywords=["Ahmad", "Endocuff", "CADe", "BCSP"],
    ),
    EvalItem(
        id="p6_q13",
        question="What are the differences between machine learning and deep learning as virtual AI components?",
        reference_answer="Machine learning (ML) determines sequences from pre-installed data using algorithms that learn from patterns. Deep learning (DL) uses multi-layer neural networks to automatically identify complex patterns in data. Both are virtual AI components, distinct from physical components like medical devices and robots.",
        paper_source=PAPER6,
        question_type="terminology", difficulty="easy",
        relevant_phrases=["machine learning", "deep learning", "multi-layer neural network", "virtual"],
        must_contain_keywords=["machine learning", "deep learning", "neural network"],
    ),
    EvalItem(
        id="p6_q14",
        question="How does the Mori et al. finding about AI increasing surveillance intervals affect healthcare costs?",
        reference_answer="Mori et al. found AI during colonoscopy increased the proportion of US and European patients requiring comprehensive colonoscopy surveillance by ~35% and ~20% respectively (absolute increases of 2.9% and 1.3%). While this may help prevent cancer through more appropriate surveillance, it significantly raises patient burden and healthcare costs.",
        paper_source=PAPER6,
        question_type="multi_hop", difficulty="hard",
        relevant_phrases=["Mori", "35%", "20%", "2.9%", "1.3%", "surveillance", "healthcare costs"],
        must_contain_keywords=["Mori", "surveillance", "healthcare costs"],
    ),
    EvalItem(
        id="p6_q15",
        question="Summarize the key benefits and current limitations of AI in CRC management.",
        reference_answer="Benefits: improved adenoma detection rates, reduced miss rates, better lesion characterization, surgical assistance (margin identification, robotic surgery), reduced operator-dependent variability. Limitations: still inferior to expert performance in some tasks, limited multicenter RCTs and real-world validation, mainly experimental stage in surgery, workflow integration challenges, potential increased healthcare costs from more intensive surveillance, and dependence on clinical expertise for implementation.",
        paper_source=PAPER6,
        question_type="summary", difficulty="hard",
        relevant_phrases=["limitation", "experimental stage", "benefits", "improved", "reduced"],
        must_contain_keywords=["benefits", "limitations", "AI", "CRC"],
    ),
]

# ═══════════════════════════════════════════════════════════════
# Paper 7: Sirinukunwattana et al. 2021 — imCMS Classification (15 Qs)
# ═══════════════════════════════════════════════════════════════

PAPER7 = "paper7_544.full.pdf"

_p7_questions = [
    EvalItem(
        id="p7_q01",
        question="What is imCMS and what does it predict from H&E images?",
        reference_answer="imCMS (image-based consensus molecular subtype) is a deep learning approach that predicts the four consensus molecular subtypes of colorectal cancer directly from standard H&E-stained histology sections, without requiring gene expression profiling.",
        paper_source=PAPER7,
        question_type="factoid", difficulty="easy",
        relevant_phrases=["imCMS", "image-based consensus molecular subtype", "H&E", "deep learning"],
        must_contain_keywords=["imCMS", "consensus molecular subtype", "H&E"],
    ),
    EvalItem(
        id="p7_q02",
        question="What three independent datasets were used for training and evaluation?",
        reference_answer="1) FOCUS trial: n=278 patients (training set), 2) GRAMPIAN cohort: n=144 rectal cancer biopsy patients (test set), 3) TCGA: n=430 patients (test set). Total of n=1,206 tissue sections with comprehensive multi-omic data.",
        paper_source=PAPER7,
        question_type="numerical", difficulty="medium",
        relevant_phrases=["FOCUS", "278", "GRAMPIAN", "144", "TCGA", "430", "1,206"],
        must_contain_keywords=["FOCUS", "GRAMPIAN", "TCGA", "1,206"],
    ),
    EvalItem(
        id="p7_q03",
        question="What AUC performance did imCMS achieve on the TCGA and GRAMPIAN test sets?",
        reference_answer="imCMS achieved AUC=0.84 on TCGA (n=431 slides) and AUC=0.85 on GRAMPIAN rectal cancer biopsies (n=265 slides) for classification of CMS from H&E images.",
        paper_source=PAPER7,
        question_type="numerical", difficulty="easy",
        relevant_phrases=["AUC = 0.84", "AUC = 0.85", "431", "265"],
        must_contain_keywords=["0.84", "0.85", "AUC"],
    ),
    EvalItem(
        id="p7_q04",
        question="What are the four CMS groups and their key characteristics?",
        reference_answer="CMS1 (14%): microsatellite instability immune, favorable prognosis in early-stage, adverse in metastatic setting. CMS2 (37%): canonical, epithelial, WNT/MYC signaling, intermediate prognosis. CMS3 (13%): epithelial with metabolic dysregulation, intermediate prognosis. CMS4 (23%): mesenchymal, TGF-β activation, poor prognosis. ~13% of samples are unclassifiable.",
        paper_source=PAPER7,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["CMS1 (14%)", "CMS2 (37%)", "CMS3 (13%)", "CMS4 (23%)", "immune", "canonical",
                          "metabolic", "mesenchymal"],
        must_contain_keywords=["CMS1", "CMS2", "CMS3", "CMS4", "14%", "37%"],
    ),
    EvalItem(
        id="p7_q05",
        question="How does imCMS handle samples previously unclassifiable by RNA expression profiling?",
        reference_answer="imCMS can classify samples previously unclassifiable by RNA expression profiling. It spatially resolves intratumoural heterogeneity by providing tile-level predictions, meaning different regions of the same tumor can be assigned different CMS calls, reflecting the heterogeneous nature of CRC.",
        paper_source=PAPER7,
        question_type="multi_hop", difficulty="hard",
        relevant_phrases=["unclassifiable", "heterogeneity", "tile level", "spatially resolves"],
        must_contain_keywords=["unclassifiable", "heterogeneity", "tile-level"],
    ),
    EvalItem(
        id="p7_q06",
        question="What is the current limitation of RNA-based CMS classification that imCMS addresses?",
        reference_answer="RNA-based CMS classification requires gene expression profiling which is costly, difficult to standardize, requires specialist bioinformatics expertise, and needs data storage infrastructure. H&E slides are inexpensive, universally available, and already part of routine pathology workflows. imCMS bridges this gap by predicting CMS from H&E images.",
        paper_source=PAPER7,
        question_type="comparative", difficulty="medium",
        relevant_phrases=["costly", "difficult to standardize", "bioinformatics", "inexpensive", "routine"],
        must_contain_keywords=["RNA", "gene expression", "H&E", "imCMS"],
    ),
    EvalItem(
        id="p7_q07",
        question="How was the ground truth CMS call established for training?",
        reference_answer="Ground truth CMS calls were established by matching random forest and single sample predictions from the CMS classifier (the gold standard RNA-based classifier developed by the CRC Subtyping Consortium).",
        paper_source=PAPER7,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["random forest", "single sample predictions", "CMS classifier", "ground truth"],
        must_contain_keywords=["random forest", "CMS classifier", "ground truth"],
    ),
    EvalItem(
        id="p7_q08",
        question="What genomic and epigenetic correlations did imCMS reproduce?",
        reference_answer="imCMS reproduced the expected correlations with genomic and epigenetic alterations that are characteristic of each CMS, and showed similar prognostic associations as transcriptomic CMS, validating that the image-based predictions capture the same biological signals as RNA-based classification.",
        paper_source=PAPER7,
        question_type="multi_hop", difficulty="hard",
        relevant_phrases=["genomic", "epigenetic", "correlations", "prognostic associations"],
        must_contain_keywords=["genomic", "epigenetic", "prognostic", "correlations"],
    ),
    EvalItem(
        id="p7_q09",
        question="What makes H&E-based molecular subtyping clinically valuable compared to RNA sequencing?",
        reference_answer="H&E-based subtyping is: 1) inexpensive (H&E is the standard stain in every pathology lab), 2) fast (no need for sequencing turnaround time), 3) universally available (every CRC patient has H&E slides), 4) spatially resolved (can capture intratumoural heterogeneity), 5) works on small samples like endoscopic biopsies, 6) interpretable (predictions can be linked to morphological features visible to pathologists).",
        paper_source=PAPER7,
        question_type="summary", difficulty="medium",
        relevant_phrases=["inexpensive", "simple, cheap", "routine workflows", "biopsies"],
        must_contain_keywords=["inexpensive", "H&E", "routine", "biopsies"],
    ),
    EvalItem(
        id="p7_q10",
        question="What is the significance of the study for clinical practice?",
        reference_answer="It shows that a prediction of RNA expression classifiers can be made from standard H&E images, opening the door to simple, cheap, and reliable biological stratification within routine pathology workflows. This could enable molecular subtyping for every CRC patient without the cost and complexity of genomic testing, and could be applied to existing retrospective cohorts with archived slides.",
        paper_source=PAPER7,
        question_type="summary", difficulty="easy",
        relevant_phrases=["simple, cheap and reliable", "routine workflows", "retrospective cohorts"],
        must_contain_keywords=["routine", "clinical", "stratification", "H&E"],
    ),
    EvalItem(
        id="p7_q11",
        question="How many tissue sections were used in total and from which cohorts?",
        reference_answer="n=1,206 tissue sections total: FOCUS trial (n=278, training), GRAMPIAN rectal cancer biopsies (n=144, test), and TCGA (n=430, test). Additional evaluations were performed on TCGA slides (n=431) and GRAMPIAN slides (n=265, multiple sections per patient).",
        paper_source=PAPER7,
        question_type="numerical", difficulty="hard",
        relevant_phrases=["1,206", "FOCUS", "278", "GRAMPIAN", "144", "TCGA", "430", "431", "265"],
        must_contain_keywords=["1,206", "FOCUS", "TCGA"],
    ),
    EvalItem(
        id="p7_q12",
        question="What prior work demonstrated the feasibility of image-based molecular classification?",
        reference_answer="Coudray et al. demonstrated that deep neural classification networks could detect targetable oncogenic driver mutations in lung cancer from histopathology images, establishing the principle that genotypes can be predicted from histological phenotypes using deep learning.",
        paper_source=PAPER7,
        question_type="multi_hop", difficulty="medium",
        relevant_phrases=["Coudray", "lung cancer", "driver mutations", "genotype"],
        must_contain_keywords=["Coudray", "lung", "mutations", "deep neural"],
    ),
    EvalItem(
        id="p7_q13",
        question="What role does the tumor microenvironment play in CMS classification and imCMS prediction?",
        reference_answer="The tumor microenvironment is a key component determining tumor progression and therapy response. Both tumor and non-tumor tissue contribute to image information on H&E slides and to the CMS classification at the transcriptional level. The composition of stromal and immune components visible in H&E images provides morphological cues that imCMS learns to associate with molecular subtypes.",
        paper_source=PAPER7,
        question_type="multi_hop", difficulty="hard",
        relevant_phrases=["tumor microenvironment", "tumor and non-tumor", "stromal", "immune", "morphological"],
        must_contain_keywords=["microenvironment", "stromal", "morphological", "CMS"],
    ),
    EvalItem(
        id="p7_q14",
        question="What are the implications of tile-level imCMS predictions for understanding tumor heterogeneity?",
        reference_answer="Tile-level predictions allow imCMS to spatially resolve intratumoural heterogeneity — different regions within the same tumor can be assigned to different CMS groups. This provides a novel insight into tumor biology that bulk RNA sequencing cannot offer, since RNA-seq averages the signal across the entire tissue sample and may miss regional variations in molecular subtype.",
        paper_source=PAPER7,
        question_type="multi_hop", difficulty="hard",
        relevant_phrases=["tile level", "spatially resolves", "intratumoural heterogeneity", "different regions"],
        must_contain_keywords=["tile", "spatial", "heterogeneity", "intratumoural"],
    ),
    EvalItem(
        id="p7_q15",
        question="What is the S:CORT consortium and why is it credited in this study?",
        reference_answer="The S:CORT consortium (Stratification in COloRecTal cancer) is a UK-based research consortium focused on developing better methods for stratifying CRC patients. This study was conducted 'on behalf of the S:CORT consortium,' indicating it was part of this broader collaborative research program aimed at improving CRC patient stratification using multi-omic approaches.",
        paper_source=PAPER7,
        question_type="factoid", difficulty="medium",
        relevant_phrases=["S:CORT", "consortium", "stratification", "colorectal"],
        must_contain_keywords=["S:CORT", "consortium"],
    ),
]

# ═══════════════════════════════════════════════════════════════
# Assemble full dataset
# ═══════════════════════════════════════════════════════════════

DATASET = (
    _p1_questions + _p2_questions + _p3_questions + _p4_questions
    + _p5_questions + _p6_questions + _p7_questions
)


def get_dataset() -> list[EvalItem]:
    """返回完整的评估数据集。"""
    return DATASET


def get_dataset_by_paper(paper: str) -> list[EvalItem]:
    """按 paper 名称筛选评估用例。"""
    return [item for item in DATASET if item.paper_source == paper]


def get_dataset_by_type(qtype: QuestionType) -> list[EvalItem]:
    """按问题类型筛选。"""
    return [item for item in DATASET if item.question_type == qtype]


def get_dataset_by_difficulty(difficulty: Difficulty) -> list[EvalItem]:
    """按难度筛选。"""
    return [item for item in DATASET if item.difficulty == difficulty]


def dataset_stats() -> dict:
    """返回数据集的统计信息。"""
    stats = {
        "total": len(DATASET),
        "by_paper": {},
        "by_type": {},
        "by_difficulty": {},
    }
    for item in DATASET:
        stats["by_paper"][item.paper_source] = stats["by_paper"].get(item.paper_source, 0) + 1
        stats["by_type"][item.question_type] = stats["by_type"].get(item.question_type, 0) + 1
        stats["by_difficulty"][item.difficulty] = stats["by_difficulty"].get(item.difficulty, 0) + 1
    return stats
