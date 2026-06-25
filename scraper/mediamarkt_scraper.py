"""
MediaMarkt API Scraper (requests-basiert, kein Playwright nötig)
Strategie: GraphQL-Suggestions-API + Produktdetailseite für Preise
Verwendung: python -m scraper.mediamarkt_scraper [--queries N] [--output PATH]
"""

import argparse
import csv
import json
import logging
import re
import time
import random
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL   = "https://www.mediamarkt.de"
GRAPHQL_URL = f"{BASE_URL}/api/v1/graphql"

HEADERS = {
    "accept":                     "*/*",
    "accept-language":            "de-DE",
    "apollographql-client-name":  "pwa-client-pqm",
    "apollographql-client-version": "8.443.2",
    "content-type":               "application/json",
    "x-cacheable":                "true",
    "x-mms-country":              "DE",
    "x-mms-language":             "de",
    "x-mms-salesline":            "Media",
    "x-operation":                "GetSuggestions",
    "user-agent":                 "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "referer":                    f"{BASE_URL}/de/search.html",
}

PERSISTED_QUERY_HASH = "f7f3ddfed0e6e079298f6429ea0da4ecf24f49b271bf72dcfe5da56e43ca13ce"

COFR_CONFIG = {
    "isEnabled": True,
    "baseDomain": BASE_URL,
    "channel": "DESKTOP",
    "isLegacyDataExcluded": False,
    "features": {
        "badges":          {"isFreeShippingBadgeIncluded": False},
        "crossSalesLine":  {"isEnabled": True, "isOutputForced": False},
        "onlineStatus":    {"isPermanentlyNaIndexEnabled": True},
        "pickup":          {"isStrictPickupDisplayStatusEnabled": False},
        "price": {
            "strikePriceTypes":                   [{"strikePriceType": "lop"}, {"strikePriceType": "rrp", "shouldBeStruck": True, "showDiscountBadge": True, "isLegalTextInlineAllowed": False}],
            "isBasePriceRequiredFlagRespected":    False,
            "isDiscountLabelEnabled":              True,
            "isDiscountPercentageShown":           True,
            "isPromoPriceFiltered":                True,
            "isHistoryChartEnabled":               False,
            "discountPercentageMinimum":           10,
        },
        "delivery": {
            "isDeliveryStatusByEarliestDateEnabled": True,
            "isLocationSourcingEnabled":             True,
        },
        "refurbishedGoods": {"isEnabled": True},
    },
    "client": {},
}

EXTENSIONS = {
    "persistedQuery": {"version": 1, "sha256Hash": PERSISTED_QUERY_HASH},
    "pwa": {
        "captureChannel":                "DESKTOP",
        "salesLine":                     "Media",
        "country":                       "DE",
        "language":                      "de",
        "globalLoyaltyProgram":          True,
        "isOneAccountProgramActive":     True,
        "shouldInactiveContractsBeHidden": True,
    },
}

# Search terms to maximise product diversity
SEARCH_QUERIES = [
    "laptop intel core i3", "laptop intel core i5", "laptop intel core i7", "laptop intel core i9",
    "laptop intel core ultra", "laptop amd ryzen 3", "laptop amd ryzen 5", "laptop amd ryzen 7",
    "laptop amd ryzen 9", "macbook air", "macbook pro", "lenovo thinkpad", "lenovo ideapad",
    "lenovo yoga", "hp elitebook", "hp probook", "hp envy", "hp spectre", "dell xps",
    "dell latitude", "dell inspiron", "asus zenbook", "asus vivobook", "asus rog",
    "acer aspire", "acer swift", "acer predator", "microsoft surface laptop",
    "samsung galaxy book", "msi laptop", "lg gram", "gaming laptop rtx 4060",
    "gaming laptop rtx 4070", "laptop 15 zoll", "laptop 14 zoll", "laptop 13 zoll",
    "ultrabook", "business laptop", "laptop windows 11", "chromebook",
]

# ─── Spec parsers (from product name string) ──────────────────────────────────

def _extract_ram(name: str) -> int | None:
    m = re.search(r"(\d+)\s*GB\s*RAM", name, re.I)
    return int(m.group(1)) if m else None

def _extract_storage(name: str) -> tuple[int | None, str | None]:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(GB|TB)\s*(SSD|HDD|eMMC)?", name, re.I)
    if not m:
        return None, None
    val = float(m.group(1).replace(",", "."))
    gb  = int(val * 1024) if m.group(2).upper() == "TB" else int(val)
    typ = (m.group(3) or "SSD").upper()
    return gb, typ

def _extract_display(name: str) -> float | None:
    m = re.search(r"(\d+[.,]\d+)\s*(?:Zoll|\"|-Zoll|Inch)", name, re.I)
    if m:
        return float(m.group(1).replace(",", "."))
    m2 = re.search(r"(\d{2}[.,]\d)\s*\"", name)
    return float(m2.group(1).replace(",", ".")) if m2 else None

def _extract_cpu(name: str) -> str | None:
    patterns = [
        r"(Intel[®\s]*Core[™\s]*(?:Ultra\s*)?\w+[\s-]\w+)",
        r"(AMD\s+Ryzen\s+\d+\s+\w+)",
        r"(Apple\s+M\d+(?:\s+(?:Pro|Max|Ultra))?)",
        r"(Snapdragon[\s\w]+)",
        r"(Qualcomm[\s\w]+)",
    ]
    for pat in patterns:
        m = re.search(pat, name, re.I)
        if m:
            return m.group(1).strip()
    return None

def _extract_os(name: str) -> str | None:
    if re.search(r"macOS|Mac\s*OS", name, re.I):
        return "macOS"
    if re.search(r"Windows\s*11\s*Pro", name, re.I):
        return "Windows 11 Pro"
    if re.search(r"Windows\s*11", name, re.I):
        return "Windows 11 Home"
    if re.search(r"Windows\s*10", name, re.I):
        return "Windows 10"
    if re.search(r"Chrome\s*OS|ChromeOS", name, re.I):
        return "ChromeOS"
    if re.search(r"Linux|Ubuntu|FreeDOS", name, re.I):
        return "Linux"
    return None

def _extract_resolution(name: str) -> str | None:
    m = re.search(r"(\d{3,4})\s*[x×]\s*(\d{3,4})", name)
    return f"{m.group(1)}x{m.group(2)}" if m else None

# ─── API helpers ──────────────────────────────────────────────────────────────

def _suggestions_url(query: str) -> str:
    variables = {
        "query":                              query,
        "hasMarketplace":                     True,
        "isCustomerBehaviorInfluenceActive":  True,
        "locale":                             "de-DE",
        "salesLine":                          "Media",
        "isRefurbishedGoodsActive":           True,
        "isPdpFaqSectionActive":              True,
        "shouldIncludeYourekoRatingExp1150":  True,
        "isDemonstrationModelAvailabilityActive": True,
        "isCrossLinkingActive":               False,
        "isPdpLoyaltyPointsActive":           True,
        "cofrConfig":                         COFR_CONFIG,
    }
    params = {
        "operationName": "GetSuggestions",
        "variables":     json.dumps(variables, separators=(",", ":")),
        "extensions":    json.dumps(EXTENSIONS, separators=(",", ":")),
    }
    return f"{GRAPHQL_URL}?{urlencode(params)}"


def fetch_suggestions(query: str, session: requests.Session) -> list[dict]:
    url = _suggestions_url(query)
    try:
        resp = session.get(url, headers=HEADERS, timeout=15, verify=False)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("suggestV4", {}).get("productSuggestions", [])
    except Exception as exc:
        log.warning("Suggestions query '%s' failed: %s", query, exc)
        return []


def fetch_price(product_url: str, session: requests.Session) -> float | None:
    """Scrape price from product detail page (JSON-LD or meta tag)."""
    url = BASE_URL + product_url if product_url.startswith("/") else product_url
    try:
        resp = session.get(url, headers={**HEADERS, "x-operation": "ProductPage"}, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")

        # JSON-LD structured data
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                ld = json.loads(script.string or "")
                if isinstance(ld, list):
                    ld = ld[0]
                offers = ld.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0]
                price_str = str(offers.get("price", ""))
                if price_str:
                    return float(price_str.replace(",", "."))
            except Exception:
                continue

        # og:price meta tag
        meta = soup.find("meta", {"property": "og:price:amount"}) or \
               soup.find("meta", {"itemprop": "price"})
        if meta:
            val = meta.get("content", "")
            return float(val.replace(",", ".").replace(".", "", val.count(".") - 1)) if val else None

    except Exception as exc:
        log.debug("Price fetch failed for %s: %s", product_url, exc)
    return None


def parse_product(raw: dict, session: requests.Session) -> dict | None:
    core  = raw.get("cofrCoreFeature", {})
    name  = core.get("productName", "")
    if not name:
        return None

    brand = core.get("manufacturerName", "")
    ean   = core.get("ean", "")
    url   = core.get("urlRelative", "")
    rating_stats = core.get("reviewStatistics") or {}

    storage_gb, storage_type = _extract_storage(name)

    # Price: try fetching from detail page
    price = None
    if url:
        price = fetch_price(url, session)
        time.sleep(random.uniform(1.5, 3.0))

    return {
        "name":         name,
        "brand":        brand,
        "price":        price,
        "ean":          ean,
        "cpu":          _extract_cpu(name),
        "gpu":          None,   # not reliably in name; enriched later if needed
        "ram_gb":       _extract_ram(name),
        "storage_gb":   storage_gb,
        "storage_type": storage_type,
        "display_inch": _extract_display(name),
        "resolution":   _extract_resolution(name),
        "os":           _extract_os(name),
        "weight_kg":    None,
        "rating":       round(rating_stats.get("averageOverallRating", 0) or 0, 2) or None,
        "review_count": rating_stats.get("totalReviewCount"),
        "url":          BASE_URL + url if url else "",
        "scraped_at":   datetime.now().isoformat(),
    }


# ─── Main scraper ─────────────────────────────────────────────────────────────

def run(max_queries: int = len(SEARCH_QUERIES), output_path: Path | None = None) -> list[dict]:
    if output_path is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = RAW_DIR / f"mediamarkt_laptops_{date_str}.csv"

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    session = requests.Session()
    session.headers.update(HEADERS)
    session.verify = False   # WSL2 SSL fix

    seen_ids: set[str] = set()
    results:  list[dict] = []

    queries = SEARCH_QUERIES[:max_queries]
    log.info("Starting scrape with %d search queries", len(queries))

    for qi, query in enumerate(queries, 1):
        log.info("[%d/%d] Query: '%s'", qi, len(queries), query)
        suggestions = fetch_suggestions(query, session)
        log.info("  → %d suggestions", len(suggestions))

        for raw in suggestions:
            core = raw.get("cofrCoreFeature", {})
            pid  = core.get("id", "") or core.get("ean", "")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)

            record = parse_product(raw, session)
            if record and record.get("price"):
                results.append(record)
                log.info("  ✓ %s | €%.0f | %s",
                         record["brand"], record["price"], record["name"][:60])
            elif record:
                results.append(record)
                log.info("  ~ %s | price=None | %s",
                         record["brand"], record["name"][:60])

        time.sleep(random.uniform(2, 4))

    # Write CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    log.info("Done — %d unique laptops → %s", len(results), output_path)
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MediaMarkt Laptop API Scraper")
    parser.add_argument("--max-queries", type=int, default=len(SEARCH_QUERIES),
                        help=f"Number of search queries to run (max {len(SEARCH_QUERIES)})")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    run(max_queries=args.max_queries, output_path=output)


if __name__ == "__main__":
    main()
