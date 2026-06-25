import random

BASE_URL = "https://www.mediamarkt.de"
LAPTOPS_URL = "https://www.mediamarkt.de/de/category/_laptops-680851.html"

# Milliseconds — random jitter between requests
RATE_LIMIT_MIN_MS = 3000
RATE_LIMIT_MAX_MS = 6000

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
]

# CSS selectors for product listing page
LISTING_SELECTORS = {
    "product_links": "a[data-test='mms-product-card-link']",
    "next_page":     "a[aria-label='Nächste Seite'], button[aria-label='Nächste Seite']",
}

# CSS selectors for product detail page
DETAIL_SELECTORS = {
    "name":          "h1[data-test='mms-pdp-heading']",
    "price":         "span[data-test='mms-pdp-price'] span",
    "rating":        "span[data-test='mms-pdp-rating-count']",
    "specs_table":   "table, dl[data-test='mms-pdp-attributes']",
    "spec_label":    "dt, th",
    "spec_value":    "dd, td",
}

# Spec label → DataFrame column mapping (German → English key)
SPEC_LABEL_MAP = {
    "prozessor":                "cpu",
    "prozessortyp":             "cpu",
    "cpu":                      "cpu",
    "grafikkarte":              "gpu",
    "grafikprozessor":          "gpu",
    "gpu":                      "gpu",
    "arbeitsspeicher":          "ram_gb",
    "ram":                      "ram_gb",
    "speicherkapazität":        "storage_gb",
    "festplattenkapazität":     "storage_gb",
    "festplattentyp":           "storage_type",
    "speichertyp":              "storage_type",
    "displaygröße":             "display_inch",
    "bildschirmgröße":          "display_inch",
    "bildschirmdiagonale":      "display_inch",
    "auflösung":                "resolution",
    "displayauflösung":         "resolution",
    "betriebssystem":           "os",
    "os":                       "os",
    "gewicht":                  "weight_kg",
    "akkukapazität":            "battery_wh",
    "akkulaufzeit":             "battery_hours",
    "farbe":                    "color",
    "marke":                    "brand",
    "hersteller":               "brand",
}

OUTPUT_COLUMNS = [
    "name", "brand", "price", "cpu", "gpu", "ram_gb", "storage_gb",
    "storage_type", "display_inch", "resolution", "weight_kg",
    "battery_wh", "battery_hours", "os", "color", "rating",
    "url", "scraped_at",
]


def random_delay_ms() -> int:
    return random.randint(RATE_LIMIT_MIN_MS, RATE_LIMIT_MAX_MS)
