"""
notebooksbilliger.de Scraper (Playwright + stealth)
Strategie: headless Chromium mit Bot-Fingerprint-Maskierung,
           JSON-LD für saubere Daten, Spec-Tabelle als Fallback.

HINWEIS: NBB blockiert Datacenter-IPs. Diesen Scraper vom Heimnetzwerk
         (Residential-IP) ausführen.

Verwendung:
  python -m scraper.nbb_scraper [--max-pages N] [--listing-only] [--output PATH]
  python -m scraper.nbb_scraper --check          # Verbindungstest
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright, Page, BrowserContext

try:
    from playwright_stealth import Stealth
    _STEALTH = Stealth()
    _HAS_STEALTH = True
except Exception:
    _STEALTH = None
    _HAS_STEALTH = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR     = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL    = "https://www.notebooksbilliger.de"
LISTING_URL = f"{BASE_URL}/notebooks+laptops"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

OUTPUT_COLUMNS = [
    "name", "brand", "price", "cpu", "gpu",
    "ram_gb", "storage_gb", "storage_type",
    "display_inch", "resolution", "os",
    "weight_kg", "rating", "review_count",
    "url", "scraped_at", "source",
]


# ─── Text-extraction helpers ──────────────────────────────────────────────────

def _extract_ram(text: str) -> int | None:
    m = re.search(r"(\d+)\s*GB\s*(?:RAM|Arbeits)", text, re.I)
    if not m:
        m = re.search(r"RAM[:\s]+(\d+)\s*GB", text, re.I)
    return int(m.group(1)) if m else None

def _extract_storage(text: str) -> tuple[int | None, str | None]:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(GB|TB)\s*(SSD|HDD|eMMC|NVMe)?", text, re.I)
    if not m:
        return None, None
    val = float(m.group(1).replace(",", "."))
    gb  = int(val * 1024) if m.group(2).upper() == "TB" else int(val)
    typ = (m.group(3) or "SSD").upper()
    return gb, "SSD" if typ == "NVME" else typ

def _extract_display(text: str) -> float | None:
    m = re.search(r'(\d{2}[.,]\d)\s*(?:Zoll|"|cm\b|inch)', text, re.I)
    if m:
        val = float(m.group(1).replace(",", "."))
        # cm → inch if "cm" was matched
        if "cm" in m.group(0).lower():
            val = round(val / 2.54, 1)
        return val
    return None

def _extract_resolution(text: str) -> str | None:
    m = re.search(r"(\d{3,4})\s*[x×]\s*(\d{3,4})", text)
    return f"{m.group(1)}x{m.group(2)}" if m else None

def _extract_weight(text: str) -> float | None:
    m = re.search(r"(\d+[.,]\d+)\s*kg", text, re.I)
    return float(m.group(1).replace(",", ".")) if m else None

def _extract_os(text: str) -> str | None:
    if re.search(r"macOS|Mac\s*OS", text, re.I):           return "macOS"
    if re.search(r"Windows\s*11\s*Pro", text, re.I):       return "Windows 11 Pro"
    if re.search(r"Windows\s*11", text, re.I):             return "Windows 11 Home"
    if re.search(r"Windows\s*10", text, re.I):             return "Windows 10"
    if re.search(r"Chrome\s*OS|ChromeOS", text, re.I):     return "ChromeOS"
    if re.search(r"Linux|Ubuntu|FreeDOS|Ohne\s*OS", text, re.I): return "Linux"
    return None

def _extract_cpu(text: str) -> str | None:
    for pat in [
        r"(Intel[®\s]*Core[™\s]*(?:Ultra\s*)?\w+[\s-]\w+(?:[\s-]\w+)?)",
        r"(AMD\s+Ryzen\s+\d+\s+\w+(?:\s+\w+)?)",
        r"(Apple\s+M\d+(?:\s+(?:Pro|Max|Ultra))?)",
        r"(Snapdragon[\s\w]+X[\s\w]*)",
        r"(Qualcomm[\s\w]+)",
    ]:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1).strip()
    return None

def _extract_gpu(text: str) -> str | None:
    for pat in [
        r"(NVIDIA\s+GeForce\s+(?:RTX|GTX)\s+\d+\w*(?:\s+\w+)?)",
        r"(AMD\s+Radeon\s+(?:RX\s+)?\w+(?:\s+\w+)?)",
        r"(Intel\s+(?:Arc|Iris\s+Xe|UHD)[\s\w]*)",
        r"(Apple\s+GPU(?:\s+\d+-core)?)",
    ]:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1).strip()
    return None

def _extract_brand(name: str) -> str:
    known = [
        "Apple", "Lenovo", "HP", "Dell", "Asus", "Acer",
        "Microsoft", "Samsung", "MSI", "LG", "Huawei",
        "Toshiba", "Razer", "Alienware", "Gigabyte", "Honor",
    ]
    for b in known:
        if name.lower().startswith(b.lower()):
            return b
    return name.split()[0] if name else ""

def _parse_price(text: str) -> float | None:
    t = text.replace("\xa0", "").replace(" ", "")
    m = re.search(r"(\d[\d.]*,\d{2})", t)  # German: 1.234,56
    if m:
        return float(m.group(1).replace(".", "").replace(",", "."))
    m2 = re.search(r"(\d[\d,]*\.\d{2})", t)  # dot-decimal
    if m2:
        return float(m2.group(1).replace(",", ""))
    m3 = re.search(r"(\d+)", t)
    return float(m3.group(1)) if m3 else None


# ─── Browser helpers ──────────────────────────────────────────────────────────

async def _delay(lo: float = 1.5, hi: float = 3.5) -> None:
    await asyncio.sleep(random.uniform(lo, hi))


async def _dismiss_overlays(page: Page) -> None:
    for sel in [
        "button[data-testid='uc-accept-all-button']",
        "button.consent-accept-all",
        "#onetrust-accept-btn-handler",
        "button:has-text('Alle akzeptieren')",
        "button:has-text('Akzeptieren')",
        "button:has-text('Zustimmen')",
        ".cookie-consent__accept",
        ".js-cookie-consent-agree",
        "[class*='cookie'] button:has-text('OK')",
    ]:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=1500):
                await btn.click()
                log.info("Dismissed overlay: %s", sel)
                await asyncio.sleep(0.6)
                return
        except Exception:
            pass


async def _make_context(pw) -> tuple:
    """Create a stealth browser + context."""
    browser = await pw.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-setuid-sandbox",
              "--disable-blink-features=AutomationControlled"],
    )
    ctx: BrowserContext = await browser.new_context(
        viewport={"width": 1440, "height": 900},
        user_agent=UA,
        locale="de-DE",
        timezone_id="Europe/Berlin",
        ignore_https_errors=True,
        extra_http_headers={
            "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        },
    )
    # Basic webdriver flag removal (playwright-stealth handles the rest)
    await ctx.add_init_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});"
        "Object.defineProperty(navigator,'languages',{get:()=>['de-DE','de','en']});"
    )
    return browser, ctx


# ─── Product detail scraper ───────────────────────────────────────────────────

async def _scrape_detail(page: Page, url: str) -> dict | None:
    """Scrape one product detail page. Returns None on failure."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        await _delay(0.8, 1.8)
        await _dismiss_overlays(page)
    except Exception as exc:
        log.warning("Detail load failed %s: %s", url, exc)
        return None

    # ── JSON-LD (cleanest source) ─────────────────────────────────────────────
    name, price, brand = "", None, ""
    rating, review_count = None, None
    try:
        for script in await page.locator("script[type='application/ld+json']").all():
            raw = await script.inner_text()
            ld = json.loads(raw)
            if isinstance(ld, list):
                ld = ld[0]
            if ld.get("@type") in ("Product", "ItemPage"):
                name  = ld.get("name", "")
                brand = (ld.get("brand") or {}).get("name", "") or _extract_brand(name)
                offers = ld.get("offers", {})
                if isinstance(offers, list):
                    offers = offers[0]
                price_str = str(offers.get("price", ""))
                if price_str:
                    try:
                        price = float(price_str.replace(",", "."))
                    except ValueError:
                        pass
                agg = ld.get("aggregateRating") or {}
                if agg.get("ratingValue"):
                    rating = float(agg["ratingValue"])
                if agg.get("reviewCount"):
                    review_count = int(agg["reviewCount"])
                break
    except Exception:
        pass

    # ── h1 fallback for name ──────────────────────────────────────────────────
    if not name:
        try:
            name = (await page.locator("h1").first.inner_text()).strip()
        except Exception:
            pass
    if not name:
        return None

    if not brand:
        brand = _extract_brand(name)

    # ── price fallback: look for price elements ───────────────────────────────
    if not price:
        for price_sel in [
            "[itemprop='price']", "[class*='price--main']",
            "[class*='offer-price']", "[class*='product-price']",
            "[data-testid*='price']", ".price",
        ]:
            try:
                el = page.locator(price_sel).first
                if await el.is_visible(timeout=800):
                    attr = await el.get_attribute("content")
                    price = _parse_price(attr or await el.inner_text())
                    if price:
                        break
            except Exception:
                pass

    # ── spec table ────────────────────────────────────────────────────────────
    spec_text = ""
    for spec_sel in [
        "table[class*='spec']", "[class*='technical-detail']",
        "[class*='product-spec']", "[class*='merkmal']", "table",
    ]:
        try:
            els = await page.locator(spec_sel).all()
            for el in els[:3]:
                spec_text += " " + await el.inner_text()
        except Exception:
            pass

    combined = name + " " + spec_text[:4000]

    storage_gb, storage_type = _extract_storage(combined)
    return {
        "name":         name,
        "brand":        brand,
        "price":        price,
        "cpu":          _extract_cpu(combined),
        "gpu":          _extract_gpu(combined),
        "ram_gb":       _extract_ram(combined),
        "storage_gb":   storage_gb,
        "storage_type": storage_type,
        "display_inch": _extract_display(combined),
        "resolution":   _extract_resolution(combined),
        "os":           _extract_os(combined),
        "weight_kg":    _extract_weight(combined),
        "rating":       rating,
        "review_count": review_count,
        "url":          url,
        "scraped_at":   datetime.now().isoformat(),
        "source":       "notebooksbilliger.de",
    }


# ─── Listing scraper ──────────────────────────────────────────────────────────

# Ordered from most-specific to generic; first match wins
_CARD_SELECTORS = [
    "[class*='product-card']",
    "[class*='product-item']",
    "[class*='product-list-item']",
    "article[class*='product']",
    "li[class*='product']",
    "[data-testid*='product']",
    "[class*='offerList-item']",
    "[class*='listing-item']",
]

_PRICE_SELECTORS = [
    "[class*='price--main']", "[class*='offer-price']",
    "[itemprop='price']", "[class*='product-price']",
    "[data-testid*='price']", "[class*='price']",
]

_NAME_SELECTORS = [
    "h2", "h3", "[class*='product-title']",
    "[class*='product-name']", "[class*='title']",
]


async def _scrape_listing_inline(page: Page) -> list[tuple[dict, str]]:
    """
    Extract (partial_record, detail_url) pairs from the current listing page.
    Returns listing-quality data + URL for optional follow-up detail scraping.
    """
    results: list[tuple[dict, str]] = []

    cards = []
    for sel in _CARD_SELECTORS:
        try:
            found = await page.locator(sel).all()
            if len(found) >= 3:
                cards = found
                log.debug("Card selector matched (%d): %s", len(found), sel)
                break
        except Exception:
            pass

    if not cards:
        # Last resort: grab all anchors with product-like hrefs
        log.warning("No product cards found; trying link fallback")
        return results

    for card in cards:
        try:
            text = (await card.inner_text()).strip()

            # URL
            url = ""
            try:
                href = await card.locator("a[href]").first.get_attribute("href")
                if href:
                    url = href if href.startswith("http") else BASE_URL + href
            except Exception:
                pass

            # Name
            name = ""
            for nsel in _NAME_SELECTORS:
                try:
                    el = card.locator(nsel).first
                    if await el.is_visible(timeout=400):
                        name = (await el.inner_text()).strip()
                        break
                except Exception:
                    pass
            if not name:
                name = text.split("\n")[0].strip()

            # Price
            price = None
            for psel in _PRICE_SELECTORS:
                try:
                    el = card.locator(psel).first
                    if await el.is_visible(timeout=400):
                        attr_val = await el.get_attribute("content")
                        price = _parse_price(attr_val or await el.inner_text())
                        if price:
                            break
                except Exception:
                    pass

            # Rating
            rating, review_count = None, None
            try:
                rt = card.locator("[itemprop='ratingValue'], [class*='rating']").first
                rv = await rt.get_attribute("content") or await rt.inner_text()
                m = re.search(r"(\d[.,]\d)", rv)
                if m:
                    rating = float(m.group(1).replace(",", "."))
            except Exception:
                pass
            try:
                rc = card.locator("[itemprop='reviewCount'], [class*='review-count']").first
                rc_text = await rc.inner_text()
                m = re.search(r"(\d+)", rc_text)
                if m:
                    review_count = int(m.group(1))
            except Exception:
                pass

            if not name:
                continue

            combined = name + " " + text
            storage_gb, storage_type = _extract_storage(combined)
            record = {
                "name":         name,
                "brand":        _extract_brand(name),
                "price":        price,
                "cpu":          _extract_cpu(combined),
                "gpu":          _extract_gpu(combined),
                "ram_gb":       _extract_ram(combined),
                "storage_gb":   storage_gb,
                "storage_type": storage_type,
                "display_inch": _extract_display(combined),
                "resolution":   _extract_resolution(combined),
                "os":           _extract_os(combined),
                "weight_kg":    _extract_weight(combined),
                "rating":       rating,
                "review_count": review_count,
                "url":          url,
                "scraped_at":   datetime.now().isoformat(),
                "source":       "notebooksbilliger.de",
            }
            results.append((record, url))
        except Exception as exc:
            log.debug("Card parse error: %s", exc)

    return results


async def _next_page_url(page: Page, page_num: int) -> str | None:
    """Return URL of page N+1, or None if no next page exists."""
    for sel in [
        f"a[href*='seite={page_num + 1}']",
        f"a[href*='page={page_num + 1}']",
        "a[rel='next']",
        "a:has-text('Nächste Seite')",
        "a:has-text('weiter')",
        "[class*='pagination'] li:last-child a",
    ]:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=1200):
                href = await el.get_attribute("href")
                if href:
                    return href if href.startswith("http") else BASE_URL + href
        except Exception:
            pass

    # Construct URL with seite parameter
    cur = page.url
    if "seite=" in cur:
        return re.sub(r"seite=\d+", f"seite={page_num + 1}", cur)
    sep = "&" if "?" in cur else "?"
    return f"{cur}{sep}seite={page_num + 1}"


# ─── Orchestrator ─────────────────────────────────────────────────────────────

async def run(
    max_pages: int = 10,
    detail_pages: bool = True,
    output_path: Path | None = None,
) -> list[dict]:
    if output_path is None:
        output_path = RAW_DIR / f"nbb_laptops_{datetime.now().strftime('%Y-%m-%d')}.csv"

    records:   list[dict] = []
    seen_urls: set[str]   = set()

    async with async_playwright() as pw:
        browser, ctx = await _make_context(pw)
        page = await ctx.new_page()

        if _HAS_STEALTH:
            await _STEALTH.apply_stealth_async(page)
            log.info("playwright-stealth applied")

        # Warm-up: homepage → pick up session cookies + pass initial checks
        log.info("Warm-up: loading homepage…")
        try:
            await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=20000)
            await _delay(1.5, 2.5)
            await _dismiss_overlays(page)
        except Exception as exc:
            log.warning("Homepage issue: %s", exc)

        # Navigate to laptop category
        log.info("Loading laptop listing: %s", LISTING_URL)
        try:
            await page.goto(LISTING_URL, wait_until="domcontentloaded", timeout=30000)
            await _delay(2, 3)
            await _dismiss_overlays(page)
        except Exception as exc:
            log.error("Listing page failed: %s", exc)
            await browser.close()
            return []

        body  = await page.inner_text("body")
        title = await page.title()
        html  = await page.content()
        if _is_blocked(body, title, html):
            log.error(
                "IP is blocked by NBB bot-protection.\n"
                "  → Run this scraper from a residential network (home broadband).\n"
                "  → Datacenter / VPS / WSL2 IPs are blocked by NBB.\n"
                "  → Run --check first to verify: python -m scraper.nbb_scraper --check\n"
                "  → Fallback: python -m scraper.generate_dataset --n 600"
            )
            await browser.close()
            return []

        for page_num in range(1, max_pages + 1):
            log.info("── Page %d/%d  %s", page_num, max_pages, page.url)

            pairs = await _scrape_listing_inline(page)
            log.info("  Listing cards: %d", len(pairs))

            if not pairs and page_num > 1:
                log.info("Empty page — stopping.")
                break

            if detail_pages:
                listing_url_snapshot = page.url
                for i, (listing_rec, url) in enumerate(pairs, 1):
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    log.info("  [%d/%d] %s", i, len(pairs), url[-75:])
                    detail = await _scrape_detail(page, url)
                    rec = detail if detail else listing_rec
                    records.append(rec)
                    log.info("    %-12s %-8s %s",
                             rec["brand"],
                             f"€{rec['price']:.0f}" if rec.get("price") else "€?",
                             rec["name"][:55])
                    await _delay(1.5, 3.0)

                # Return to listing page before paginating
                try:
                    await page.goto(listing_url_snapshot,
                                    wait_until="domcontentloaded", timeout=25000)
                    await _delay(1, 2)
                except Exception:
                    pass
            else:
                for listing_rec, url in pairs:
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        records.append(listing_rec)

            if page_num < max_pages:
                next_url = await _next_page_url(page, page_num)
                if not next_url:
                    log.info("No next page — done.")
                    break
                try:
                    await page.goto(next_url, wait_until="domcontentloaded", timeout=25000)
                    await _delay(2, 4)
                    await _dismiss_overlays(page)
                except Exception as exc:
                    log.error("Next page failed: %s", exc)
                    break

        await browser.close()

    if records:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        log.info("Saved %d records → %s", len(records), output_path)
    else:
        log.warning("No records collected. Check if IP is blocked.")

    return records


def _is_blocked(body: str, title: str = "", html: str = "") -> bool:
    body_l  = body.lower()
    title_l = title.lower()
    html_l  = html.lower()
    return (
        # Explicit bot blocks
        "bot detected"    in body_l or
        "bot protection"  in body_l or
        "bot detected"    in title_l or
        # NBB's disguised challenge page (error-box + tiny HTML)
        ("nicht gefunden" in body_l and "error-box" in html_l) or
        ("uups"           in body_l and len(body_l) < 500) or
        # Cloudflare / generic challenge markers
        "challenge-form"  in html_l or
        "cf-browser-verification" in html_l or
        "ray id"          in body_l
    )


async def _check_connectivity() -> bool:
    """Connectivity test against the actual listing page (not just homepage)."""
    async with async_playwright() as pw:
        browser, ctx = await _make_context(pw)
        page = await ctx.new_page()
        if _HAS_STEALTH:
            await _STEALTH.apply_stealth_async(page)

        blocked = True
        try:
            # Warm-up: homepage (usually accessible even on datacenter IPs)
            await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(1.5)
            for sel in ["button:has-text('Akzeptieren')", "#onetrust-accept-btn-handler"]:
                try:
                    btn = page.locator(sel).first
                    if await btn.is_visible(timeout=1000):
                        await btn.click(); await asyncio.sleep(0.4); break
                except Exception:
                    pass

            # Real check: listing page (this is where datacenter IPs get blocked)
            await page.goto(LISTING_URL, wait_until="domcontentloaded", timeout=25000)
            await asyncio.sleep(3)
            body  = await page.inner_text("body")
            title = await page.title()
            html  = await page.content()
            blocked = _is_blocked(body, title, html)
        except Exception as exc:
            log.error("Connection error: %s", exc)
            blocked = True
        finally:
            await browser.close()

    if blocked:
        print("❌  notebooksbilliger.de: GEBLOCKT (IP-basierte Bot-Protection)")
        print("   → Datacenter- und WSL2-IPs werden auf der Listing-Seite geblockt.")
        print("   → Scraper vom Heimnetzwerk (Residential-IP) ausführen.")
        print("   → Fallback: python -m scraper.generate_dataset --n 600")
        return False
    else:
        print("✅  notebooksbilliger.de: erreichbar — Scraper kann gestartet werden")
        print(f"   → python -m scraper.nbb_scraper --max-pages 10")
        return True


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="notebooksbilliger.de Laptop Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Hinweis: NBB blockiert Datacenter-IPs.\n"
            "Diesen Scraper vom Heimnetzwerk (Residential-IP) ausführen.\n\n"
            "Fallback für Training ohne echte Daten:\n"
            "  python -m scraper.generate_dataset --n 600"
        ),
    )
    parser.add_argument("--max-pages", type=int, default=5,
                        help="Anzahl Listing-Seiten (default: 5, ~20 Produkte/Seite)")
    parser.add_argument("--listing-only", action="store_true",
                        help="Keine Detail-Seiten besuchen (schneller, weniger Daten)")
    parser.add_argument("--output", type=str, default=None,
                        help="Ausgabe-CSV (default: data/raw/nbb_laptops_YYYY-MM-DD.csv)")
    parser.add_argument("--check", action="store_true",
                        help="Verbindungstest: prüft ob NBB erreichbar ist")
    args = parser.parse_args()

    if args.check:
        asyncio.run(_check_connectivity())
        return

    asyncio.run(run(
        max_pages=args.max_pages,
        detail_pages=not args.listing_only,
        output_path=Path(args.output) if args.output else None,
    ))


if __name__ == "__main__":
    main()
