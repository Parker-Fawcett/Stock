"""Ticker universes. Current constituents = survivorship-biased by
definition (his finale lesson). Fine for plumbing; not for claiming edge.
"""
SP100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "COST",
    "JPM", "V", "MA", "XOM", "UNH", "LLY", "JNJ", "WMT", "BAC", "PG",
    "ORCL", "HD", "CVX", "MRK", "ABBV", "KO", "CSCO", "CRM", "ACN",
    "AMD", "LIN", "TXN", "DHR", "DIS", "VZ", "PFE", "T", "INTC",
    "CMCSA", "NEE", "UPS", "QCOM", "HON", "AMGN", "LOW", "SBUX",
    "IBM", "GE", "SPGI", "INTU", "MDT", "CAT", "GS", "DE", "BLK",
    "AXP", "PLD", "GILD", "BK", "ADP", "TJX", "VRTX", "REGN", "MU",
    "PANW", "LRCX", "KLAC", "ADI", "MCHP", "CSX", "FDX", "EMR",
    "ETN", "PH", "ROST", "ODFL", "CTAS", "PAYX", "MNST", "KDP",
    "KHC", "MDLZ", "PM", "MO", "CL", "ECL", "SHW", "APD", "NEM",
    "DUK", "SO", "AEP", "XEL", "WEC", "ES", "AWK", "CNP", "NI",
    "AMT", "CCI", "EQIX", "WY", "VICI", "ARE", "DLR", "O",
]

# 100 names, every 6th of the S&P SmallCap 600 list (Wikipedia, Sep 2026).
# Current constituents = survivorship-biased; worse here than large caps.
SMALLCAP = [
    "AAMI", "ACA", "AGX", "AMTM", "ARR", "AWR", "BFAM", "BMI", "BTU",
    "CC", "CON", "CRI", "CWEN", "DAN", "DFIN", "DXC", "EMN", "EPC",
    "FBP", "FMC", "GFF", "GPOR", "HAYW", "HMN", "IIPR", "JOE", "KMT",
    "KSS", "LEG", "LTH", "MAN", "MHK", "MTX", "NGVT", "NPO", "OGN",
    "PBI", "PMT", "RCUS", "RHP", "SAFE", "SHO", "SM", "SXT", "TGTX",
    "UA", "UVV", "VSXY", "WHD", "WU", "ACAD", "ADAM", "AGYS", "ALRM",
    "AOSL", "AZTA", "BL", "CARG", "CENTA", "CLSK", "COLL", "CSW",
    "DAVE", "EBC", "EYE", "FHB", "FRPT", "GSHD", "HLIT", "HUBG",
    "INDV", "IRDM", "KLIC", "LKFN", "LYFT", "MBGL", "MIR", "MSEX",
    "NBHC", "NSSC", "OPLN", "PAYO", "PENN", "PLUS", "PRDO", "PSMT",
    "QDEL", "REYN", "RUSHA", "SDGR", "SHEN", "SPNT", "STRA", "TMDX",
    "UCB", "UPBD", "VCTR", "VSAT", "WDFC", "WSC",
]

# 100 names, every 4th of the S&P MidCap 400 list (Wikipedia, Sep 2026).
# Fresh universe: never evaluated in this project before this run.
MIDCAP = [
    "AA", "AFG", "ALK", "AM", "APG", "ASB", "AVNT", "AYI", "BCO",
    "BJ", "BURL", "CAVA", "CDP", "CLF", "CNM", "CR", "CTRE", "CXT",
    "DKS", "DT", "EHC", "ENS", "ESNT", "FBIN", "FLG", "FNB", "FR",
    "GATX", "GME", "GXO", "HL", "HRB", "IRT", "KBH", "KNF", "LAD",
    "MLI", "MP", "MTG", "MUSA", "NLY", "NVT", "OGE", "ONTO", "OVV",
    "PB", "PII", "POR", "PSN", "RBC", "RLI", "RRX", "SAIC", "SF",
    "SNX", "SSD", "SUI", "THG", "TOL", "TTC", "ULS", "VFC", "VNT",
    "WCC", "WLK", "WTRG", "AAL", "AMH", "AVAV", "BSY", "CART", "CGNX",
    "COKE", "CRUS", "DOCU", "ENTG", "EXPO", "GBCI", "HIMS", "HWC", "ILMN",
    "KRYS", "LFUS", "LSCC", "MEDP", "MTSI", "NTNX", "OLED", "OZK", "PPC",
    "RMBS", "SANM", "SHC", "SLAB", "SOLS", "TCBI", "TXRH", "UTHR", "VNOM",
    "WTFC",
]
