"""The ticker universe, pinned in source rather than scraped.

Why a hardcoded list instead of pulling S&P 500 constituents from Wikipedia at
fetch time: the fixture is supposed to be reproducible. A scrape makes the
dataset depend on whatever Wikipedia looked like that morning, and index
membership changes. Pinning the list here means `git log` shows exactly when the
universe changed and why.

Selection criteria: large-cap US common stocks with continuous 5-minute
liquidity, spread across sectors so the cross-section isn't all mega-cap tech.
ETFs are excluded — the paper models equities, and an ETF's microstructure
(creation/redemption, no idiosyncratic earnings jumps) is a different process.
"""

from __future__ import annotations

# Deliberately sector-diversified. yfinance will drop anything delisted or
# renamed; fetch_bars.py reports what it lost rather than failing silently.
UNIVERSE: tuple[str, ...] = (
    # Technology / semiconductors
    "AAPL", "MSFT", "NVDA", "AVGO", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT",
    "LRCX", "KLAC", "ADI", "NXPI", "MCHP", "ON", "SWKS", "MPWR", "TER", "ENTG",
    "CRM", "ORCL", "ADBE", "NOW", "INTU", "IBM", "ACN", "CSCO", "ANET", "PANW",
    "SNOW", "DDOG", "CRWD", "ZS", "NET", "MDB", "TEAM", "WDAY", "HUBS", "SPLK",
    "APH", "GLW", "HPQ", "HPE", "DELL", "STX", "WDC", "NTAP", "KEYS", "FTNT",
    # Communication services / media
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR",
    "EA", "TTWO", "WBD", "PARA", "LYV", "OMC", "IPG", "MTCH", "PINS", "SNAP",
    # Consumer discretionary
    "AMZN", "TSLA", "HD", "LOW", "MCD", "SBUX", "NKE", "TJX", "BKNG", "ABNB",
    "MAR", "HLT", "CMG", "YUM", "DRI", "ROST", "ORLY", "AZO", "LULU", "DECK",
    "GM", "F", "RIVN", "LCID", "APTV", "BWA", "LEA", "DHI", "LEN", "NVR",
    "PHM", "WHR", "MHK", "EBAY", "ETSY", "W", "CHWY", "DPZ", "WING", "TSCO",
    # Consumer staples
    "PG", "KO", "PEP", "COST", "WMT", "TGT", "DG", "DLTR", "KR", "SYY",
    "MDLZ", "GIS", "K", "HSY", "STZ", "KHC", "CAG", "CPB", "SJM", "HRL",
    "CL", "KMB", "CHD", "CLX", "EL", "MO", "PM", "KDP", "MNST", "CELH",
    # Financials
    "BRK-B", "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "USB", "PNC",
    "TFC", "COF", "AXP", "BK", "STT", "NTRS", "FITB", "HBAN", "RF", "KEY",
    "CFG", "MTB", "ZION", "CMA", "ALLY", "DFS", "SYF", "V", "MA", "PYPL",
    "FI", "FIS", "GPN", "BLK", "BX", "KKR", "APO", "ARES", "TROW", "BEN",
    "SPGI", "MCO", "MSCI", "ICE", "CME", "NDAQ", "CBOE", "MKTX", "COIN", "HOOD",
    "AIG", "MET", "PRU", "AFL", "ALL", "TRV", "PGR", "CB", "HIG", "L",
    # Health care
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD", "BIIB",
    "VRTX", "REGN", "MRNA", "ILMN", "INCY", "ALNY", "BMRN", "NBIX", "SRPT", "EXAS",
    "TMO", "DHR", "ABT", "SYK", "BSX", "MDT", "EW", "ZBH", "BAX", "BDX",
    "ISRG", "DXCM", "PODD", "RMD", "HOLX", "ALGN", "IDXX", "WST", "STE", "COO",
    "CVS", "CI", "ELV", "HUM", "CNC", "MCK", "COR", "CAH", "HCA", "UHS",
    # Industrials
    "GE", "HON", "RTX", "BA", "LMT", "NOC", "GD", "LHX", "TDG", "HWM",
    "CAT", "DE", "CMI", "PCAR", "ETN", "EMR", "PH", "ROK", "DOV", "ITW",
    "MMM", "SWK", "IR", "XYL", "AME", "FTV", "GGG", "NDSN", "SNA", "LECO",
    "UPS", "FDX", "UNP", "CSX", "NSC", "ODFL", "JBHT", "CHRW", "EXPD", "LSTR",
    "DAL", "UAL", "AAL", "LUV", "ALK", "URI", "PWR", "EME", "J", "ACM",
    # Energy
    "XOM", "CVX", "COP", "EOG", "PXD", "OXY", "HES", "DVN", "FANG", "MRO",
    "APA", "CTRA", "SLB", "HAL", "BKR", "NOV", "FTI", "PSX", "VLO", "MPC",
    "KMI", "WMB", "OKE", "TRGP", "LNG", "EQT", "AR", "RRC", "SWN", "CHK",
    # Materials
    "LIN", "APD", "SHW", "ECL", "DD", "DOW", "LYB", "PPG", "RPM", "ALB",
    "NEM", "FCX", "GOLD", "AA", "X", "NUE", "STLD", "CLF", "RS", "CMC",
    "VMC", "MLM", "CRH", "IP", "PKG", "WRK", "AMCR", "SEE", "BALL", "CCK",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "WEC", "ES",
    "PEG", "SRE", "PCG", "EIX", "FE", "AEE", "CMS", "DTE", "CNP", "NI",
    # Real estate
    "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "VTR", "DLR",
    "AVB", "EQR", "ESS", "MAA", "UDR", "INVH", "ARE", "BXP", "VNO", "HST",
)


def universe(limit: int | None = None) -> list[str]:
    """Deduplicated ticker list, order preserved.

    `limit` truncates for the SMOKE preset, which only needs enough names to
    prove the fetch/assembly path works.
    """
    seen: dict[str, None] = {}
    for t in UNIVERSE:
        seen.setdefault(t, None)
    out = list(seen)
    return out[:limit] if limit else out
