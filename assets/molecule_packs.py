# ==============================================================================
# molecule_packs.py
# -----------------
# This file contains the curated lists of molecules for the
# Cheminformatics Similarity Explorer project.
#
# Storing them here keeps the main application code clean while ensuring
# deployment with PyInstaller remains simple and robust.
# ==============================================================================


# ------------------------------------------------------------------------------
# Pack 1: Amino Acids (A single pack, no Main/Extended needed)
# ------------------------------------------------------------------------------

AMINO_ACID_PACK = [
    "alanine", "arginine", "asparagine", "aspartic acid",
    "cysteine", "glutamine", "glutamic acid", "glycine",
    "histidine", "isoleucine", "leucine", "lysine",
    "methionine", "phenylalanine", "proline", "serine",
    "threonine", "tryptophan", "tyrosine", "valine"
]


# ------------------------------------------------------------------------------
# Pack 2: Freshman Chemistry
# ------------------------------------------------------------------------------

FRESHMAN_MAIN_PACK = [
    "water",
    "ammonia",
    "methane",
    "ethanol",
    "acetone",
    "acetic acid",
    "benzene",
    "glucose",
    "caffeine",
    "serotonin",
    "sulfuric acid",
    "histamine"
]

FRESHMAN_EXTENDED_PACK = [
    "water", "carbon dioxide", "oxygen", "ammonia", "methane",
    "acetate", "carbonate", "nitrate", "phosphate", "cyanide",
    "ethanol", "acetone", "acetic acid", "formaldehyde", "butane",
    "benzene", "toluene", "phenol", "glucose", "urea", "lactic acid",
    "caffeine", "nicotine", "serotonin", "dopamine", "epinephrine",
    "hydrochloric acid", "sulfuric acid", "ammonium", "bicarbonate",
    "hydroxide", "nitrate", "histamine"
]


# ------------------------------------------------------------------------------
# Pack 3: Common Drugs
# ------------------------------------------------------------------------------

DRUG_MAIN_PACK = [
    "metformin",
    "acetaminophen",
    "ibuprofen",
    "acetylsalicylic acid",
    "morphine",
    "sertraline",
    "amoxicillin",
    "oseltamivir",
    "sildenafil",
    "loratadine",
    "caffeine",
    "nicotine"
]

DRUG_EXTENDED_PACK = [
    "metformin", "empagliflozin", "dapagliflozin",
    "canagliflozin", "acetaminophen", "ibuprofen",
    "naproxen", "acetylsalicylic acid", "celecoxib",
    "morphine", "fentanyl", "oxycodone", "fluoxetine",
    "sertraline", "escitalopram", "alprazolam",
    "lorazepam", "methylphenidate", "amphetamine",
    "lisdexamfetamine", "bupropion", "amoxicillin",
    "penicillin G", "ciprofloxacin", "acyclovir",
    "oseltamivir", "nirmatrelvir", "remdesivir",
    "levonorgestrel", "estrogen", "testosterone",
    "tamoxifen", "sildenafil", "tadalafil",
    "loratadine", "diphenhydramine", "cetirizine",
    "pseudoephedrine", "guaifenesin", "fluticasone",
    "albuterol", "montelukast", "caffeine",
    "nicotine", "tetrahydrocannabinol", "cannabidiol",
    "melatonin", "acetylcholine", "naloxone",
    "methadone", "ketamine", "ibogaine"
]


# ------------------------------------------------------------------------------
# Pack 4: Organic Chemistry Functional Groups
# ------------------------------------------------------------------------------

ORGANIC_MAIN_PACK = [
    "hexane",           # Alkane
    "cyclohexane",      # Cycloalkane
    "1-hexene",         # Alkene
    "benzene",          # Aromatic
    "ethanol",          # Alcohol
    "diethyl ether",    # Ether
    "acetone",          # Ketone
    "acetic acid",      # Carboxylic Acid
    "ethyl acetate",    # Ester
    "ethylamine",       # Amine
    "acetamide",        # Amide
    "chloroform"        # Halogenated
]

ORGANIC_EXTENDED_PACK = [
    "methane", "ethane", "propane", "butane", "pentane", "hexane", "cyclohexane",
    "ethene", "propene", "1-butene", "1-pentene", "1-hexene", "1-butyne", "2-butyne",
    "benzene", "toluene", "p-xylene", "anisole", "styrene", "phenol",
    "methanol", "ethanol", "isopropanol", "butanol", "glycerol",
    "dimethyl ether", "diethyl ether", "tetrahydrofuran", "1,4-dioxane",
    "formaldehyde", "acetaldehyde", "acetone", "butanone", "benzaldehyde",
    "formic acid", "acetic acid", "propionic acid", "butyric acid", "benzoic acid",
    "methyl acetate", "ethyl acetate", "methyl benzoate", "isoamyl acetate",
    "methylamine", "ethylamine", "aniline", "trimethylamine", "diisopropylamine",
    "acetamide", "benzamide", "dimethylformamide",
    "acetonitrile", "butanenitrile", "ethanethiol", "thiophenol",
    "urea", "acetyl chloride", "chloroform", "carbon tetrachloride"
]