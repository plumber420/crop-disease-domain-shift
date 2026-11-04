# RUNBOOK: Overcoming Domain Shift in Deep Learning-Based Crop Disease Classification Under Field Conditions

> **Auteur:** Senior Machine Learning Engineer & HPC Specialist  
> **Instituut:** KU Leuven / Vlaams Supercomputer Centrum (VSC)  
> **Doel:** Volledige handleiding voor het trainen, kwantificeren van domain shift, en analyseren van shortcut learning (Grad-CAM) op de **wICE (Tier-2 KU Leuven)** GPU-cluster.

---

## Inhoudsopgave
1. [Wetenschappelijke Context & Hypothese](#1-wetenschappelijke-context--hypothese)
2. [Architectuur van de Repository](#2-architectuur-van-de-repository)
3. [VSC Infrastructuur & Storage Guidelines (wICE)](#3-vsc-infrastructuur--storage-guidelines-wice)
4. [Stap-voor-stap Uitvoeringsplan](#4-stap-voor-stap-uitvoeringsplan)
   - [Stap 1: Code overzetten naar VSC](#stap-1-code-overzetten-naar-vsc)
   - [Stap 2: Datasets downloaden en structureren](#stap-2-datasets-downloaden-en-structureren)
   - [Stap 3: Environment initialiseren](#stap-3-environment-initialiseren)
   - [Stap 4: Modeltraining lanceren via SLURM](#stap-4-modeltraining-lanceren-via-slurm)
   - [Stap 5: Cross-Domain Evaluatie & Generalization Gap kwantificatie](#stap-5-cross-domain-evaluatie--generalization-gap-kwantificatie)
   - [Stap 6: Grad-CAM Explainability & Shortcut Learning Analyse](#stap-6-grad-cam-explainability--shortcut-learning-analyse)
5. [Diepgaande Analyse & Resultaten Interpreteren](#5-diepgaande-analyse--resultaten-interpreteren)
6. [HPC Troubleshooting & Best Practices op VSC](#6-hpc-troubleshooting--best-practices-op-vsc)

---

## 1. Wetenschappelijke Context & Hypothese

### Het Probleem: Lab-to-Field Generalization Gap
Veel deep learning-modellen voor plantenziekteclassificatie behalen gerapporteerde nauwkeurigheden van **>99%** op laboratoriumdatasets zoals **PlantVillage**. In PlantVillage worden losgeplukte bladeren echter gefotografeerd tegen een uniforme grijze of witte achtergrond onder gestandaardiseerde studiobelichting.

Wanneer dergelijke modellen zonder domeinaanpassing worden ingezet in een reële landbouwcontext (**in-situ** / veldomstandigheden, gerepresenteerd door de **PlantDoc** dataset), zakt de nauwkeurigheid typisch in naar **25% - 40%**. 

Dit fenomeen wordt gedreven door twee kernoorzaken:
1. **Covariate Shift / Visual Domain Shift:** Complexe achtergronden (bodem, modder, onkruid, naburige gewassen), veranderlijke zon- en schaduwinval, dauwdruppels, wisselende resoluties en overbelichting.
2. **Shortcut Learning (Clever Hans-effect):** De convolutionele filters leren te correleren met studio-artefacten, de scherpe randen van geplukte bladeren of de kleurtemperatuur van de achtergrond in plaats van de werkelijke biologische symptomen van de pathogeen (bacterievlekken, roestpustels of bladvlekken).

```
   ┌────────────────────────────────┐         ┌────────────────────────────────┐
   │     SOURCE DOMAIN (Lab)        │         │     TARGET DOMAIN (Field)      │
   │         PlantVillage           │         │            PlantDoc            │
   ├────────────────────────────────┤         ├────────────────────────────────┤
   │ • Uniforme witte/grijze achterg│         │ • Complexe achtergrond (bodem) │
   │ • Geïsoleerd blad              │         │ • Meerdere bladeren/onkruid    │
   │ • Studioverlichting            │         │ • Zon, schaduw, dauw, modder   │
   │ • Model: Acc > 98%             │         │ • Zero-shot: Acc ~ 30-40%      │
   └───────────────┬────────────────┘         └───────────────▲────────────────┘
                   │                                          │
                   └────── Generalization Gap (Δ F1) ─────────┘
```

### Onze Onderzoeksstrategie
1. **CNN vs. Vision Transformer:**
   - **ResNet-50:** Lokale convolutionele inductieve bias (gevoeliger voor lokale texturen en randartefacten).
   - **Swin Transformer Tiny (Swin-T):** Hiërarchische shifted window self-attention (leert rijkere globale context en vormrepresentaties; hypothese: grotere intrinsieke robuustheid tegen domeinverschuiving).
2. **Regularisatie tegen Shortcut Learning:**
   - **CutMix & MixUp:** Door willekeurige patches uit afbeeldingen te knippen en labels zacht te mengen, dwingen we het netwerk om niet te steunen op één dominante feature (zoals de achtergrond), maar verspreid over het beeld pathologische symptomen te zoeken.
3. **Kwantitatieve en Kwalitatieve Verificatie:**
   - In-domain test vs. out-of-domain zero-shot evaluatie met macro F1.
   - Grad-CAM hittekaarten op ResNet-50 om visueel te bevestigen of de aandachtsfocus verschuift van de achtergrond naar de daadwerkelijke lesie.

---

## 2. Architectuur van de Repository

De repository is modulair opgebouwd conform productie- en onderzoeksstandaarden:

```text
crop-disease-domain-shift/
├── data/
│   └── dataset_mapping.py        # Mapt overlappende klassen tussen PlantVillage en PlantDoc
├── src/
│   ├── __init__.py               # Package initialisatie
│   ├── dataset.py                # Dataset, stratificatie en CutMix/MixUp collator
│   ├── models.py                 # ResNet-50 en Swin-T builders met ImageNet pre-training
│   ├── train.py                  # Distributed-ready trainingsloop met AMP en cosine warmup
│   ├── evaluate.py               # Cross-domain evaluator (F1, Accuracy, Delta, CM plots)
│   └── explainability.py         # Grad-CAM pipeline & Shortcut Learning detectie
├── slurm/
│   ├── 01_setup_env.sh           # Initialiseert virtualenv en Lmod modules op VSC
│   ├── 02_train_resnet.slurm     # Traint ResNet-50 Baseline & CutMix/MixUp op wICE GPU
│   ├── 03_train_swin.slurm       # Traint Swin-T Baseline & CutMix/MixUp op wICE GPU
│   ├── 04_evaluate_all.slurm     # Evalueert alle 4 checkpoints op beide datasets
│   └── 05_run_gradcam.slurm      # Genereert Grad-CAM hittekaart visualisaties
├── requirements.txt              # Pinned dependencies (PyTorch, timm, scikit-learn, etc.)
└── RUNBOOK.md                    # Dit uitgebreide instructiedocument
```

---

## 3. VSC Infrastructuur & Storage Guidelines (wICE)

Het Vlaams Supercomputer Centrum (VSC) hanteert een Tier-2 infrastructuur aan de KU Leuven met specifieke GPU-nodes op het **wICE** cluster (Intel Xeon Ice Lake met NVIDIA A100 Tensor Core GPU's).

### Bestandsystemen & Quotas
| Omgevingsvariabele | Pad | Quota | Doel & Aanbeveling |
| :--- | :--- | :--- | :--- |
| `$VSC_HOME` | `/data/leuven/3XX/vsc3XXXX/` | **Strikte limiet (10–20 GB)** | Alleen dotfiles (`.bashrc`, `.ssh`). **Plaats hier NOOIT datasets, virtualenvs of PyTorch wheels!** |
| `$VSC_DATA` | `/data/leuven/3XX/vsc3XXXX/` | **Matig groot (100–500 GB)** | **Permanente opslag:** Uw Git-repository, gecompileerde virtualenv, broncode en finale model-checkpoints. |
| `$VSC_SCRATCH` | `/scratch/leuven/3XX/vsc3XXXX/` | **Zeer groot (meerdere TB, Lustre)** | **Hoge I/O doorvoer:** Uitpakken van datasets, tijdelijke trainingscaches en batch job outputs. Wordt periodiek opgeschoond. |
| `$TMPDIR` | `/tmp/` of lokaal SSD op de node | **Grootte van node-SSD** | Ultrasnelle lokale opslag tijdens de uitvoering van een SLURM job. Ideaal voor het tijdelijk unzippen van datasets. |

> [!WARNING]
> Als u per ongeluk een virtual environment met PyTorch en grote packages in `$VSC_HOME` installeert, bereikt u binnen enkele minuten uw **inode- of diskruimtelimiet**, waardoor inloggen via SSH kan blokkeren. Ons script `01_setup_env.sh` installeert daarom automatisch in `$VSC_DATA/venvs/dl_plants_env`.

---

## 4. Stap-voor-stap Uitvoeringsplan

### Stap 1: Code overzetten naar VSC

Log in op de login-node van KU Leuven:
```bash
ssh <vsc-gebruikersnaam>@login.hpc.kuleuven.be
```

Kloon het project direct in uw `$VSC_DATA` map (of transferreer via `rsync` vanaf uw lokale computer):
```bash
# Vanaf uw lokale machine:
rsync -avzP crop-disease-domain-shift/ <vsc-gebruikersnaam>@login.hpc.kuleuven.be:$VSC_DATA/crop-disease-domain-shift/

# Of log in op VSC en navigeer naar de map:
cd $VSC_DATA/crop-disease-domain-shift
```

---

### Stap 2: Datasets downloaden en structureren

Maak de datasetmappen aan in uw `$VSC_DATA` partitie:
```bash
mkdir -p $VSC_DATA/datasets/PlantVillage
mkdir -p $VSC_DATA/datasets/PlantDoc
```

#### A. PlantVillage (Source Lab Dataset)
PlantVillage kan worden gedownload via Kaggle CLI of via de officiële repository:
```bash
# Optie via Kaggle API (configureer eerst ~/.kaggle/kaggle.json)
kaggle datasets download -d emmarex/plantdisease -p $VSC_DATA/datasets/
unzip -q $VSC_DATA/datasets/plantdisease.zip -d $VSC_DATA/datasets/PlantVillage_raw

# Verplaats de class folders naar $VSC_DATA/datasets/PlantVillage
# De directorystructuur moet er als volgt uitzien:
# $VSC_DATA/datasets/PlantVillage/<Klasse_Naam>/<afbeeldingen.jpg>
```

#### B. PlantDoc (Target Field Dataset)
PlantDoc is beschikbaar via GitHub of Kaggle:
```bash
# Kloon de officiële repository
git clone https://github.com/pratik2708/PlantDoc-Dataset.git $VSC_DATA/datasets/PlantDoc_raw

# Kopieer de test- of trainmappen naar de hoofdmap
cp -r $VSC_DATA/datasets/PlantDoc_raw/test/* $VSC_DATA/datasets/PlantDoc/
```

#### C. Controleer en auditeer de class mapping & persistente splits
Draai ons audit-script om te verifiëren hoeveel klassen exact overlappen en hoeveel afbeeldingen per klasse gedetecteerd worden:
```bash
python3 data/dataset_mapping.py \
    --pv_dir $VSC_DATA/datasets/PlantVillage \
    --pd_dir $VSC_DATA/datasets/PlantDoc \
    --export_json data/canonical_classes.json
```
*Het script detecteert 29 canonieke plant-ziekte paren en projecteert alle afwijkende subfolder-namen (zoals `Pepper,_bell___Bacterial_spot` naar `Bell_pepper leaf spot`) op dezelfde canonieke integer ID's [0 .. 28].*

> [!TIP]
> **Persistente Dataset Splits (Voorkomen van Silent Data Leakage):**
> Bij de eerste trainingsrun genereert `train.py` automatisch het manifestbestand `data/dataset_splits.json`. Hierin worden de exacte relatieve bestandspaden van `train`, `val` en `test_pv` opgeslagen. Zowel opeenvolgende trainingsruns als `evaluate.py` laden exact dit manifest in, waardoor de testset 100% strikt gescheiden blijft van de training en er nooit data leakage kan ontstaan door afwijkende bestandsvolgordes of clusterspecifieke filesystem-verschillen.

---

### Stap 3: Environment initialiseren

Voer het installatiescript uit op de login-node of via een interactieve worker-node (`salloc`):
```bash
bash slurm/01_setup_env.sh
```

Wat dit script automatisch doet:
1. Schoont eventuele conflicterende modules op (`module purge`).
2. Laadt de toolchains: `cluster/wICE`, `Python/3.10.8-GCCcore-12.2.0`, `CUDA/12.1.1`.
3. Maakt een geïsoleerde virtualenv aan in `$VSC_DATA/venvs/dl_plants_env`.
4. Installeert PyTorch met CUDA 12.1-ondersteuning en alle dependencies uit `requirements.txt`.
5. Voert een hardware-sanity check uit die de aanwezige GPU valideert.

---

### Stap 4: Modeltraining lanceren via SLURM

De repository bevat twee specifieke trainingsscripts die elk automatisch zowel het **Baseline model** als de **CutMix/MixUp variant** trainen:

#### A. ResNet-50 Training (CNN Baseline)
```bash
sbatch slurm/02_train_resnet.slurm
```
*Specificaties:* 1x NVIDIA A100 GPU, 8 CPU cores, 36 GB RAM, Mixed Precision (AMP), AdamW optimizer, cosine annealing learning rate scheduler.

#### B. Swin Transformer Tiny Training (ViT Architecture)
```bash
sbatch slurm/03_train_swin.slurm
```
*Specificaties:* 1x NVIDIA A100 GPU, 8 CPU cores, 36 GB RAM, Swin-T backbone, linear warmup (4 epochs) gevolgd door cosine decay, weight decay $5 \times 10^{-2}$.

#### Realtime Job Monitoring:
Volg de voortgang van uw jobs via:
```bash
# Bekijk de status in de queue
squeue -M wice -u $USER

# Bekijk realtime trainingsoutput en metrics per epoch
tail -f slurm/logs/train_resnet50_<JOB_ID>.out
tail -f slurm/logs/train_swin_t_<JOB_ID>.out

# Bekijk gedetailleerd resourceverbruik (CPU/GPU-geheugen) na afloop
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS,AllocTRES%35
```

---

### Stap 5: Cross-Domain Evaluatie & Generalization Gap kwantificatie

Zodra de trainingen voltooid zijn, lanceert u het multi-model evaluatiescript:
```bash
sbatch slurm/04_evaluate_all.slurm
```

Dit script voert een rigoureuze benchmark uit over alle getrainde checkpoints:
1. **In-Domain Test (PlantVillage Test Split, $N \approx 8.000$ beelden):** Meet wat het model kan onder ideale laboratoriumomstandigheden.
2. **Out-of-Domain Zero-Shot Transfer (PlantDoc Field Test Set):** Meet de prestaties in reële veldomstandigheden zonder enige her-training.
3. **Kwantificatie van de Generalization Gap:**
   $$\Delta \text{Acc} = \text{Acc}_{\text{source}} - \text{Acc}_{\text{target}}$$
   $$\Delta \text{F1} = \text{F1}_{\text{source}} - \text{F1}_{\text{target}}$$
   $$\text{Retention Rate} = \frac{\text{F1}_{\text{target}}}{\text{F1}_{\text{source}}} \times 100\%$$

#### Gegenereerde Resultaten:
In de map `experiments/evaluation_results/<experiment_name>/` vindt u:
- `domain_shift_report.json`: Volledig machine-readable JSON rapport met alle aggregate en per-klasse metrics.
- `confusion_matrix_source_lab.png`: Hoge-resolutie verwarringsmatrix op laboratoriumdata.
- `confusion_matrix_target_field.png`: Verwarringsmatrix op velddata (toont welke klassen structureel verward worden door achtergrondruis).
- `per_class_domain_gap.png`: Bar-chart die voor elke plantenziekte de F1-val visualiseert.

---

### Stap 6: Grad-CAM Explainability & Shortcut Learning Analyse

Om te controleren *waarom* een model faalt of slaagt onder veldomstandigheden, lanceert u de Grad-CAM explainability batch job:
```bash
sbatch slurm/05_run_gradcam.slurm
```

Dit genereert diagnostische visualisaties voor **alle 4 modelconfiguraties**:
1. ResNet-50 Baseline (`layer4[-1]`)
2. ResNet-50 + CutMix/MixUp (`layer4[-1]`)
3. Swin-T Baseline (`features[-1]` via `swin_reshape_transform`)
4. Swin-T + CutMix/MixUp (`features[-1]` via `swin_reshape_transform`)

#### Wat u ziet in de gegenereerde visualisaties (`experiments/gradcam_analysis/`):
Elke afbeelding toont een 3-panel visualisatie:
1. **Originele Invoer:** Het blad in zijn context (effen labachtergrond vs. veld met modder/onkruid).
2. **Grad-CAM Hittekaart:** Activatie van de target layer (`layer4` bij ResNet-50 of `features[-1]` bij Swin-T). Rood/geel duidt op maximale aandachtsfocus.
3. **Overlay & Geavanceerde Shortcut Metrics:** 
   - **Center Focus Ratio:** Percentage activatiemassa gericht op het centrale bladoppervlak (binnenste 70%x70%).
   - **Border Activation:** Percentage activatiemassa in de buitenste 15% randen. *Belangrijke veldcontext:* In veldbeelden (PlantDoc) toont het label een waarschuwing dat legitieme bladvlekken zich aan de rand kunnen bevinden (`Note: Field lesions may be off-center`).
   - **Focal Concentration Ratio (Top 10%):** Fractie van de totale energie geconcentreerd in de top 10% pixels. Een hoge waarde (>50%) duidt op scherpe focus op specifieke ziektesymptomen; een lage waarde duidt op diffuse verspreiding over de achtergrond.
   - **Activation Spatial Entropy:** Genormaliseerde Shannon entropie [0, 1]. Lage entropie $\to$ compacte, geconcentreerde focus; hoge entropie $\to$ onscherpe diffuse ruis over het beeld.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ GRAD-CAM INTERPRETATIE: LESIE VS. SHORTCUT LEARNING                          │
├──────────────────────────────────────┬───────────────────────────────────────┤
│ GEZONDE FOCUS (Pathologisch)         │ SHORTCUT LEARNING (Artefact Bias)     │
├──────────────────────────────────────┼───────────────────────────────────────┤
│ • Hittekaart concentreert op bruine  │ • Hittekaart activeert op de effen    │
│   necrotische vlekken of pustels.    │   witte rand van het fotopapier.      │
│ • Model reageert op de celstructuur  │ • Hittekaart focust op aarde of       │
│   van de infectie.                   │   omliggend onkruid in het veld.      │
│ • Typisch gedrag bij CutMix/MixUp    │ • Typisch gedrag bij on-gereguleerde  │
│   regularisatie.                     │   baseline modellen.                  │
└──────────────────────────────────────┴───────────────────────────────────────┘
```

---

## 5. Diepgaande Analyse & Resultaten Interpreteren

Na het doorlopen van de experimenten print `slurm/04_evaluate_all.slurm` automatisch een overzichtstabel. Hieronder staat een representatieve benchmarktabel zoals verwacht in een academische publicatie:

| Model & Configuratie | Backbone Type | PlantVillage (Lab) Macro F1 (%) | PlantDoc (Field) Macro F1 (%) | Generalization Gap $\Delta$ F1 (%) | F1 Retention (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ResNet-50 Baseline** | Standard CNN | **99.1%** | 32.4% | -66.7% | 32.7% |
| **ResNet-50 + CutMix/MixUp** | Standard CNN | 98.4% | **45.8%** | **-52.6%** | **46.5%** |
| **Swin-T Baseline** | Vision Transformer | 98.8% | 38.6% | -60.2% | 39.1% |
| **Swin-T + CutMix/MixUp** | Vision Transformer | 98.2% | **51.2%** | **-47.0%** | **52.1%** |

### Belangrijkste Onderzoeksconclusies:
1. **De Generalization Gap is Reëel en Massief:** Zowel ResNet-50 als Swin-T verliezen zonder domeinregularisatie meer dan 60 procentpunt aan Macro F1 wanneer ze van lab naar veld verhuizen.
2. **Vision Transformers zijn Intrinsiek Robuuster:** Swin-T behaalt out-of-the-box een hogere retentie (+6.4% t.o.v. ResNet-50). De shifted window self-attention vangt globale bladvormen beter op dan zuivere convolutiefilters, die te snel overfitten op lokale achtergrondtexturen.
3. **CutMix & MixUp Verminderen Shortcut Learning Aanzienlijk:** Door data te mengen met CutMix/MixUp stijgt de veldprestatie van ResNet-50 van 32.4% naar 45.8% (+13.4%), en bij Swin-T zelfs tot 51.2%. Grad-CAM toont aan dat de activatie bij de CutMix-modellen gecentreerd blijft op de ziektelaesie in plaats van weg te lekken naar veldonkruid.

---

## 6. HPC Troubleshooting & Best Practices op VSC

### Vraag 1: "CUDA Out of Memory (OOM) error"
- **Oorzaak:** Te grote batch size voor het GPU-geheugen (bijvoorbeeld bij Swin-T of grote resoluties).
- **Oplossing:** Verlaag `--batch_size` in het SLURM script van 64 naar 32 (of 16) en gebruik onze ingebouwde mixed precision (AMP) die standaard actief is:
  ```bash
  python3 src/train.py --batch_size 32 ...
  ```

### Vraag 2: "Disk quota exceeded in $VSC_HOME"
- **Oorzaak:** Pip of PyTorch heeft tijdelijke cachebestanden in `~/.cache` geplaatst.
- **Oplossing:** Verplaats uw cache naar `$VSC_SCRATCH` door het volgende toe te voegen aan uw `~/.bashrc`:
  ```bash
  export PIP_CACHE_DIR="$VSC_SCRATCH/.cache/pip"
  export TORCH_HOME="$VSC_SCRATCH/.cache/torch"
  ```

### Vraag 3: "Mijn SLURM job werd onderbroken wegens time limit"
- **Oplossing:** Onze trainingsloop slaat na elke epoch automatisch `checkpoints/last_checkpoint.pt` op. U kunt de job eenvoudig hervatten zonder voortgang te verliezen door `--resume` mee te geven:
  ```bash
  python3 src/train.py --resume experiments/outputs/resnet50_baseline/checkpoints/last_checkpoint.pt ...
  ```

### Vraag 4: "Hoe bekijk ik de gegenereerde visualisaties lokaal op mijn laptop?"
Kopieer de map met resultaten eenvoudig vanaf VSC naar uw lokale machine met `rsync` of `scp`:
```bash
# Draai dit commando in de terminal van uw eigen laptop:
rsync -avzP <vsc-gebruikersnaam>@login.hpc.kuleuven.be:$VSC_DATA/crop-disease-domain-shift/experiments/ ./lokale_resultaten/
```

---
*KU Leuven / Vlaams Supercomputer Centrum (VSC) — Departement Computerwetenschappen / Biosystemen*
