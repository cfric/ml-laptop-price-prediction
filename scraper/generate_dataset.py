"""
Generiert einen realistischen synthetischen Laptop-Datensatz.
Basiert auf echten Produktkombinationen und realen Preisrelationen.
Verwendung: python -m scraper.generate_dataset [--n 500]
"""
import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

BRANDS = {
    "Lenovo":    0.20,
    "HP":        0.18,
    "Dell":      0.15,
    "Asus":      0.13,
    "Acer":      0.10,
    "Apple":     0.09,
    "Microsoft": 0.04,
    "Samsung":   0.04,
    "MSI":       0.04,
    "LG":        0.03,
}

CPUS = [
    # (name, score, base_price_impact)
    ("Intel Core i3-1215U",   3.0,    0),
    ("Intel Core i5-1235U",   5.0,  150),
    ("Intel Core i5-1335U",   5.5,  180),
    ("Intel Core i7-1255U",   7.0,  350),
    ("Intel Core i7-1355U",   7.5,  400),
    ("Intel Core i7-13700H",  8.0,  500),
    ("Intel Core i9-13900H",  9.0,  800),
    ("Intel Core Ultra 5 125H", 6.5, 300),
    ("Intel Core Ultra 7 155H", 8.0, 550),
    ("Intel Core Ultra 9 185H", 9.5, 900),
    ("AMD Ryzen 3 7320U",     3.0,    0),
    ("AMD Ryzen 5 7530U",     5.0,  140),
    ("AMD Ryzen 5 7535HS",    5.5,  200),
    ("AMD Ryzen 7 7735U",     7.0,  360),
    ("AMD Ryzen 7 7745HX",    7.5,  480),
    ("AMD Ryzen 9 7945HX",    9.0,  750),
    ("Apple M2",              7.0,  400),
    ("Apple M2 Pro",          8.5,  700),
    ("Apple M3",              8.0,  500),
    ("Apple M3 Pro",          9.0,  800),
    ("Apple M3 Max",          9.5, 1200),
]

GPUS = [
    # (name, tier, price_impact)
    ("Intel UHD Graphics",        0.0,    0),
    ("Intel Iris Xe Graphics",    1.5,   50),
    ("AMD Radeon Graphics",       1.5,   30),
    ("Apple GPU (10-core)",       2.0,    0),
    ("Apple GPU (18-core)",       2.5,    0),
    ("NVIDIA GeForce RTX 3050",   4.0,  200),
    ("NVIDIA GeForce RTX 3060",   6.0,  350),
    ("NVIDIA GeForce RTX 4050",   6.5,  300),
    ("NVIDIA GeForce RTX 4060",   7.5,  450),
    ("NVIDIA GeForce RTX 4070",   8.5,  700),
    ("NVIDIA GeForce RTX 4080",   9.5, 1000),
    ("AMD Radeon RX 6600M",       5.5,  280),
    ("AMD Radeon RX 7600M",       7.0,  420),
]

RAM_OPTIONS    = [4, 8, 16, 24, 32, 64]
STORAGE_OPTIONS = [128, 256, 512, 1024, 2048]
DISPLAY_SIZES  = [11.6, 13.3, 13.6, 14.0, 14.2, 15.6, 16.0, 17.3]
RESOLUTIONS = {
    11.6: "1366x768",
    13.3: "1920x1080",
    13.6: "2560x1664",
    14.0: "1920x1080",
    14.2: "3024x1964",
    15.6: "1920x1080",
    16.0: "2560x1600",
    17.3: "1920x1080",
}
OS_OPTIONS = ["Windows 11 Home", "Windows 11 Pro", "macOS", "Linux", "Ohne OS"]

SERIES = {
    "Lenovo":    ["ThinkPad X1 Carbon", "ThinkPad E14", "IdeaPad 5", "Yoga 7", "Legion 5"],
    "HP":        ["EliteBook 840", "ProBook 450", "ENVY 15", "Spectre x360", "Omen 16"],
    "Dell":      ["XPS 13", "XPS 15", "Latitude 5540", "Inspiron 15", "Alienware m16"],
    "Asus":      ["ZenBook 14", "VivoBook 15", "ROG Zephyrus G14", "ExpertBook B9", "ProArt"],
    "Acer":      ["Swift 3", "Aspire 5", "Predator Helios 16", "ConceptD 5", "Nitro 5"],
    "Apple":     ["MacBook Air 13\"", "MacBook Air 15\"", "MacBook Pro 14\"", "MacBook Pro 16\""],
    "Microsoft": ["Surface Laptop 5", "Surface Laptop Studio 2", "Surface Pro 9"],
    "Samsung":   ["Galaxy Book3 Pro", "Galaxy Book3 360", "Galaxy Book3 Ultra"],
    "MSI":       ["Stealth 16", "Prestige 14", "Raider GE76", "Creator Z16"],
    "LG":        ["gram 14", "gram 16", "gram 17", "gram +2-in-1"],
}


def _base_price(cpu_score: float, gpu_tier: float, ram_gb: int, storage_gb: int,
                display_inch: float, brand: str) -> float:
    # Empirisch abgestimmte Preisformel
    price = 400.0
    price += cpu_score * 80
    price += gpu_tier  * 60
    price += (ram_gb - 8)  * 12
    price += (storage_gb / 512) * 40
    price += (display_inch - 13) * 15
    brand_premium = {
        "Apple": 250, "Microsoft": 150, "LG": 100, "Dell": 50,
        "Lenovo": 20, "HP": 10, "Samsung": 80, "MSI": 30,
    }
    price += brand_premium.get(brand, 0)
    # Noise ±12 %
    price *= random.uniform(0.88, 1.12)
    return round(price, 2)


def _pick_compatible(brand: str):
    cpu_name, cpu_score, cpu_delta = random.choice(CPUS)
    is_apple = brand == "Apple"

    if is_apple:
        # Apple only uses Apple Silicon
        apple_cpus = [c for c in CPUS if "Apple" in c[0]]
        cpu_name, cpu_score, cpu_delta = random.choice(apple_cpus)
        gpu_name, gpu_tier, gpu_delta = random.choice(
            [g for g in GPUS if "Apple" in g[0]]
        )
        os_choice = "macOS"
    else:
        non_apple = [c for c in CPUS if "Apple" not in c[0]]
        cpu_name, cpu_score, cpu_delta = random.choice(non_apple)
        # Gaming CPU → likely dedicated GPU
        if cpu_score >= 7.5:
            gpu_candidates = [g for g in GPUS if g[1] >= 4.0 and "Apple" not in g[0]]
        else:
            gpu_candidates = [g for g in GPUS if "Apple" not in g[0]]
        gpu_name, gpu_tier, gpu_delta = random.choice(gpu_candidates)
        os_choice = random.choices(
            ["Windows 11 Home", "Windows 11 Pro", "Linux", "Ohne OS"],
            weights=[0.65, 0.20, 0.10, 0.05]
        )[0]

    # RAM: high-end CPU → more RAM
    ram_weights = {4: 1, 8: 4, 16: 8, 24: 2, 32: 3, 64: 1}
    if cpu_score >= 7:
        ram_weights[4] = 0; ram_weights[8] = 1; ram_weights[16] = 6
    ram_gb = random.choices(RAM_OPTIONS, weights=[ram_weights[r] for r in RAM_OPTIONS])[0]

    # Storage
    storage_weights = {128: 1, 256: 5, 512: 8, 1024: 4, 2048: 1}
    storage_gb = random.choices(STORAGE_OPTIONS,
                                weights=[storage_weights[s] for s in STORAGE_OPTIONS])[0]
    storage_type = "SSD"

    # Display
    if is_apple:
        display_inch = random.choice([13.3, 13.6, 14.2, 15.6, 16.0])
    elif cpu_score >= 8.0:
        display_inch = random.choice([15.6, 16.0, 17.3])
    else:
        display_inch = random.choice(DISPLAY_SIZES)

    resolution = RESOLUTIONS.get(display_inch, "1920x1080")
    weight_kg = round(random.uniform(0.9, 3.2), 2)
    rating = round(random.uniform(3.8, 5.0), 1)
    review_count = random.randint(5, 850)

    series = random.choice(SERIES.get(brand, ["Notebook"]))
    name = f"{brand} {series}, mit {display_inch} Zoll Display, {cpu_name} Prozessor, {ram_gb} GB RAM, {storage_gb} GB {storage_type}, {gpu_name}, {os_choice}"

    price = _base_price(cpu_score, gpu_tier, ram_gb, storage_gb, display_inch, brand)

    return {
        "name":         name,
        "brand":        brand,
        "price":        price,
        "cpu":          cpu_name,
        "gpu":          gpu_name,
        "ram_gb":       ram_gb,
        "storage_gb":   storage_gb,
        "storage_type": storage_type,
        "display_inch": display_inch,
        "resolution":   resolution,
        "weight_kg":    weight_kg,
        "os":           os_choice,
        "rating":       rating,
        "review_count": review_count,
        "url":          "",
        "scraped_at":   datetime.now().date().isoformat(),
        "source":       "synthetic",
    }


def generate(n: int = 500, output_path: Path | None = None) -> list[dict]:
    brands = list(BRANDS.keys())
    weights = list(BRANDS.values())

    records = []
    for _ in range(n):
        brand = random.choices(brands, weights=weights)[0]
        records.append(_pick_compatible(brand))

    if output_path is None:
        output_path = RAW_DIR / f"laptops_synthetic_{datetime.now().date()}.csv"

    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {n} synthetic laptop records → {output_path}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    generate(n=args.n, output_path=Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
