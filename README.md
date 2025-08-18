# CobberLearnChem — Companion Software for *Foundations of Machine Learning for Chemistry*

This repository hosts the companion software and materials for the textbook **Foundations of Machine Learning for Chemistry** (Ulness, Mork, Mach). The suite provides chapter-aligned, hands-on apps that help students explore data, train models, and evaluate results in a chemistry context. :contentReference[oaicite:0]{index=0}

## What’s here

- **Chapter-aligned apps** launched from the CobberLearnChem main window (e.g., “Chapter 9: Similarity, Clustering, and Classification”). Each app presents interactive controls and visualizations that match the lab activities in the book. :contentReference[oaicite:1]{index=1}
- **Datasets & models** used by the apps (placed in a `DataSets_Models/` folder alongside the executable). :contentReference[oaicite:2]{index=2}
- **Docs** connecting activities to portfolio work (Coding and Ethics) and to the three-domain framing (Data → Models → Results). :contentReference[oaicite:3]{index=3}

## Install (Windows 64-bit)

1. Download the **CobberLearnChem Windows installer** from the companion site.  
2. Run the installer and choose an install location.  
3. In the install folder, locate `CobberLearnChem1.exe`. (Optional) Create a desktop shortcut.  
4. Double-click to launch. A small terminal appears briefly; after ~30–40 seconds the main window opens.  
5. If you hit issues, consult the Installation Guide on the companion site.  
*Source: book “Launching Your Digital Lab” section.* :contentReference[oaicite:4]{index=4}

### Add datasets and models (one-time)

1. From the same site, download the **Data and models** zip.  
2. Extract and move the folder **`DataSets_Models`** into the **same folder** as `CobberLearnChem1.exe`.  
3. Launch the app and choose **Chapter 9: Similarity, Clustering, and Classification** to verify:  
   - Increase **Number of Clusters** → **Initialize Centroids** (plot should update).  
   - Click **Run Full Algorithm** (messages appear in the log). :contentReference[oaicite:5]{index=5}

> Companion website link is provided in the book’s front-matter and support pages (installers, datasets, and tutorials are hosted there). :contentReference[oaicite:6]{index=6}

## Using the suite in your course

- The book structures learning across **Data**, **Models**, and **Results** domains; each chapter’s app aligns with the corresponding lab so students can manipulate parameters, visualize outcomes, and record evidence for their portfolios. :contentReference[oaicite:7]{index=7}
- Most chapters include: **Ethics Reflection**, **Vibe Coding Project**, and **In the Wild** spotlights; apps are designed to support those workflows. :contentReference[oaicite:8]{index=8}

## Repository layout (example)

CobberLearnChem/
├─ apps/ # chapter-aligned launchers or modules
├─ data/ # small example CSVs for docs/tests
├─ docs/ # quickstart & chapter mapping notes
├─ tools/ # utility scripts for packaging/builds
├─ requirements.txt # for running dev utilities from source
└─ README.md


> On Windows, end users run `CobberLearnChem1.exe` with `DataSets_Models/` beside it. Instructors/developers can use the Python sources in `apps/` for demos or extension.

## Portfolios (for students)

- **Coding Portfolio (public):** push your lab code, figures (predicted-vs-actual, residual plots), and READMEs.  
- **Ethics Portfolio (private):** write reflections (Overleaf/LaTeX) on model use, collaboration with AI, and scientific responsibility.  
These practices are emphasized throughout the text and mirrored in the app design. :contentReference[oaicite:9]{index=9}

## Contributing

We welcome fixes and enhancements that improve pedagogy or stability.

1. Open an issue describing your change.  
2. Fork and create a feature branch.  
3. Keep PRs small and well-scoped; include brief usage notes or screenshots where relevant.

## Troubleshooting

- **Main window doesn’t appear:** allow 30–40 seconds after launch; keep the small terminal window open/minimized. :contentReference[oaicite:10]{index=10}  
- **App can’t find data/models:** ensure `DataSets_Models/` sits **next to** `CobberLearnChem1.exe`. :contentReference[oaicite:11]{index=11}  
- **Installation issues:** use the companion site’s Installation Guide and video tutorials listed in the book. :contentReference[oaicite:12]{index=12}

## License

- **Textbook excerpts** follow the license specified in the book.  
- **Code in this repository**: MIT (or your preferred OSI license—update this line and add a `LICENSE` file).

## Citation

If this software or its materials support your work, please cite the textbook and companion software:

> Ulness, D. J.; Mork, P. S.; Mach, J. R. *Foundations of Machine Learning for Chemistry.* Concordia College, 2025.  
> Companion software: CobberLearnChem (this repository).

