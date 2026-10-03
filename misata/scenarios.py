"""Row-level scenarios: one latent story per row, rendered into every column.

A product's name, description, brand and price band, or a ticket's subject,
description, priority and resolution, describe ONE thing. Drawn column by
column from independent pools they describe different things: a linen dress
"lightweight and portable with a modern aesthetic", a ticket about a double
charge whose resolution resets a password. This module draws the thing first
(a :class:`ProductFrame` or :class:`TicketFrame`) and renders each column from
it, so the columns agree by construction.

Everything here is fictional and offline: brands, products and support
issues are written for Misata, and no row of any real dataset is used.

Capacity, not pool size, is what keeps a large table from repeating itself.
Names and descriptions are compositions (brand x attribute x noun x variant;
opener x features x use x spec), so a 100,000-row catalogue does not read the
same sentence every few rows. ``tests/test_scenarios.py`` enumerates the
grammars and gates their capacity.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


# ── shared helpers ───────────────────────────────────────────────────────────

def zipf_weights(n: int, s: float = 1.0, q: float = 2.0) -> np.ndarray:
    """Zipf-Mandelbrot weights ``(k + q) ** -s`` for ranks 1..n, normalised.

    Real label columns (cities, employers, brands, job titles) are skewed: a
    few values cover much of the table and a long tail covers the rest. A
    uniform draw over a list is the most common tell in synthetic text."""
    if n <= 0:
        return np.array([])
    w = (np.arange(1, n + 1) + q) ** -float(s)
    return w / w.sum()


def zipf_choice(rng: np.random.Generator, pool: Sequence, size: int, *,
                s: float = 1.0, q: float = 2.0, key: Optional[str] = None,
                ranked: bool = False) -> np.ndarray:
    """Draw ``size`` values from ``pool`` with Zipf-Mandelbrot weights.

    ``ranked=True`` means the pool is already in popularity order (cities by
    population): rank follows list order. Otherwise ``key`` seeds a stable
    permutation, so which value is common is fixed per column, not always the
    first one written in the list."""
    pool = list(pool)
    if not pool:
        return np.array([], dtype=object)
    order = np.arange(len(pool))
    if not ranked:
        perm = np.random.default_rng(zlib.crc32(str(key or pool[0]).encode("utf-8")))
        order = perm.permutation(len(pool))
    idx = rng.choice(len(pool), size=size, p=zipf_weights(len(pool), s, q))
    arr = np.empty(len(pool), dtype=object)
    arr[:] = pool
    return arr[order[idx]]


_SLOT = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::(\d+)-(\d+))?\}")


def _fill(template: str, rng: np.random.Generator, slots: Dict[str, str]) -> str:
    """Fill ``{name}`` from ``slots`` and ``{n:lo-hi}`` with an integer."""
    def sub(m: re.Match) -> str:
        name, lo, hi = m.group(1), m.group(2), m.group(3)
        if lo is not None:
            return str(int(rng.integers(int(lo), int(hi) + 1)))
        if name not in slots:
            raise KeyError(f"scenario slot '{name}' is not defined")
        return str(slots[name])
    return _SLOT.sub(sub, template)


def _pick(rng: np.random.Generator, pool):
    """One entry of a list; ``rng.choice`` converts the list to an array on
    every call, which dominated per-row rendering."""
    return pool[int(rng.integers(len(pool)))]


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _clean_values(values: Optional[Sequence], size: int) -> Optional[List[str]]:
    if values is None:
        return None
    out = []
    for v in list(values)[:size]:
        t = "" if v is None else str(v).strip()
        out.append("" if t.lower() in ("nan", "none", "nat", "<na>") else t)
    if len(out) < size:
        out += [""] * (size - len(out))
    return out


# ── products ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProductFamily:
    key: str
    label: str                 # display category when none is declared
    nouns: Sequence[str]
    attrs: Sequence[str]
    materials: Sequence[str]
    features: Sequence[str]    # noun phrases with an article: "a padded strap"
    uses: Sequence[str]
    specs: Sequence[str]       # complete sentences, may carry {n:lo-hi}
    variants: Sequence[str]
    brands: Sequence[str]
    price_median: float        # USD, for ordering prices across categories
    material_in_name: bool = False


_F = ProductFamily
PRODUCT_FAMILIES: Dict[str, ProductFamily] = {f.key: f for f in [
    _F("electronics", "Electronics",
       ["Wireless Headphones", "Bluetooth Speaker", "Laptop Stand", "Mechanical Keyboard",
        "Gaming Mouse", "4K Monitor", "Webcam", "Desk Lamp", "Power Bank", "Wireless Earbuds",
        "Smart Plug", "Action Camera", "Tablet", "Smartwatch", "E-Reader", "Dash Cam",
        "Portable SSD", "Wi-Fi Router", "USB-C Hub", "Soundbar", "Phone Charger",
        "Fitness Tracker", "Smart Thermostat", "Video Doorbell", "Microphone",
        "Graphics Tablet", "Projector", "Security Camera"],
       ["Compact", "Portable", "Ultra-Slim", "Noise-Cancelling@Headphones|Earbuds|Microphone", "Rechargeable", "Smart",
        "Wireless", "Pro", "Waterproof@Speaker|Camera|Earbuds|Smartwatch|Tracker", "Foldable@Stand|Headphones|Keyboard|Lamp", "Fast-Charging@Charger|Power Bank|Hub", "Low-Latency@Headphones|Earbuds|Mouse|Keyboard|Microphone|Webcam"],
       ["anodised aluminium@Stand|Hub|Monitor|Lamp|Tablet|SSD|Speaker|Keyboard|Microphone", "recycled plastic", "matte polycarbonate", "brushed steel@Stand|Lamp|Speaker|Smartwatch|Microphone",
        "soft-touch silicone@Earbuds|Smartwatch|Tracker|Mouse|Speaker"],
       ["a {n:10-60}-hour battery@Headphones|Speaker|Earbuds|Power Bank|Smartwatch|Tracker|Microphone|Camera|Reader", "USB-C fast charging@Headphones|Speaker|Earbuds|Power Bank|Smartwatch|Tracker|Charger|Tablet|Reader|Mouse|Microphone|Camera", "Bluetooth 5.3@Headphones|Speaker|Earbuds|Keyboard|Mouse|Smartwatch|Tracker|Soundbar", "a braided cable@Charger|Power Bank|Hub|Keyboard|Mouse|Microphone",
        "a magnetic mount@Camera|Charger|Lamp|Webcam|Dash", "voice-assistant support@Speaker|Smart|Soundbar|Doorbell|Thermostat", "an app for firmware updates@Smart|Router|Camera|Doorbell|Thermostat|Headphones|Earbuds|Speaker|Smartwatch|Tracker|Soundbar",
        "a fold-flat hinge@Stand|Headphones|Lamp|Tablet", "dual microphones@Headphones|Earbuds|Webcam|Speaker|Doorbell|Camera", "an IPX{n:4-7} rating@Speaker|Earbuds|Camera|Smartwatch|Tracker|Doorbell", "a carry pouch@Headphones|Earbuds|SSD|Speaker|Power Bank|Mouse|Camera",
        "a {n:1-3}-year warranty", "automatic standby", "multipoint pairing@Headphones|Earbuds|Keyboard|Mouse|Speaker",
        "a backlit control panel@Keyboard|Router|Thermostat|Projector|Soundbar", "a {n:2-4}-port design@Hub|Charger|Power Bank|Router"],
       ["working from home", "long commutes", "travel", "gaming sessions", "video calls",
        "a small desk", "streaming", "the home office", "students", "content creators"],
       ["Charges fully in about {n:1-3} hours.@Headphones|Speaker|Earbuds|Power Bank|Smartwatch|Tracker|Camera|Mouse", "Weighs {n:90-900} g.",
        "Works with Windows, macOS, iOS and Android.@Keyboard|Mouse|Webcam|Hub|SSD|Headphones|Earbuds|Microphone|Tablet|Speaker", "Cable and quick-start guide included.",
        "Supports firmware updates over the companion app.@Smart|Router|Camera|Doorbell|Thermostat|Headphones|Earbuds|Speaker|Smartwatch|Tracker|Soundbar", "Plug-and-play, no drivers needed.@Keyboard|Mouse|Webcam|Hub|SSD|Microphone",
        "Ships with a {n:1-2} m USB-C cable.@Keyboard|Mouse|Webcam|Hub|SSD|Microphone|Headphones|Speaker|Power Bank|Camera"],
       ["- Black", "- White", "- Graphite", "- Silver", "2nd Gen", "Mini", "Max", "Lite",
        "(2025)", "128GB@Tablet|SSD|Smartwatch|Reader", "256GB@Tablet|SSD|Reader", "- Midnight Blue"],
       ["Voltra", "Nexion", "Aurelo", "Kestrel", "Lumio", "Orbix", "Zentra", "Sonaro",
        "Helix", "Quill & Bolt", "Arcwave", "Pixelon"],
       79.0),
    _F("clothing", "Clothing",
       ["Rain Jacket", "Oxford Shirt", "Chino Trousers", "V-Neck Sweater", "Yoga Leggings",
        "Summer Dress", "Running Shorts", "Puffer Jacket", "Graphic Tee", "Straight-Leg Jeans",
        "Midi Skirt", "Hoodie", "Blazer", "Cargo Shorts", "Scarf", "Beanie", "Polo Shirt",
        "Cardigan", "Joggers", "Denim Jacket", "Wrap Dress", "Fleece Pullover", "Tank Top",
        "Overshirt", "Trench Coat", "Sweatpants"],
       ["Classic", "Relaxed", "Slim-Fit", "Lightweight", "Oversized", "Cropped", "Everyday",
        "Waterproof@Jacket|Coat", "Stretch@Jeans|Trousers|Leggings|Chino|Shorts|Joggers", "Vintage-Wash@Jeans|Tee|Jacket|Hoodie", "Tailored@Blazer|Trousers|Shirt|Coat", "Ribbed@Sweater|Beanie|Tank|Cardigan|Dress"],
       ["organic cotton", "merino wool@Sweater|Beanie|Scarf|Cardigan|Pullover|Tee", "linen@Shirt|Dress|Trousers|Shorts|Overshirt|Skirt|Blazer", "recycled polyester@Jacket|Leggings|Shorts|Puffer|Coat|Tank|Pullover", "French terry@Hoodie|Joggers|Sweatpants|Pullover", "cotton twill@Chino|Trousers|Shorts|Overshirt|Trench|Jacket|Blazer", "TENCEL lyocell@Dress|Shirt|Skirt|Tee|Tank", "brushed fleece@Pullover|Hoodie|Joggers|Jacket|Sweatpants", "stretch denim@Jeans|Jacket|Skirt|Shorts", "cashmere blend@Sweater|Scarf|Beanie|Cardigan|Coat"],
       ["a relaxed fit", "a tailored fit", "reinforced seams", "side pockets@Jacket|Trousers|Shorts|Hoodie|Joggers|Dress|Coat|Cardigan|Skirt|Jeans|Sweatpants|Blazer",
        "a two-way zip@Jacket|Hoodie|Pullover|Coat", "an adjustable hood@Jacket|Hoodie|Coat", "a soft brushed interior@Hoodie|Joggers|Sweatpants|Pullover|Jacket|Leggings", "flatlock stitching@Leggings|Shorts|Tank|Tee|Joggers",
        "a drawstring waist@Shorts|Joggers|Sweatpants|Trousers|Leggings", "a dropped shoulder@Tee|Hoodie|Sweater|Pullover|Cardigan|Shirt|Overshirt", "a hidden phone pocket@Leggings|Shorts|Joggers|Jacket|Sweatpants",
        "moisture-wicking fabric@Leggings|Shorts|Tank|Tee|Joggers|Polo", "a curved hem@Shirt|Tee|Overshirt|Tank|Polo", "ribbed cuffs@Sweater|Hoodie|Cardigan|Joggers|Pullover|Sweatpants|Jacket"],
       ["everyday wear", "the office", "weekend trips", "layering in autumn", "warm evenings",
        "the gym@Leggings|Shorts|Tank|Joggers|Hoodie|Tee", "travel", "rainy commutes@Jacket|Coat", "lounging at home@Hoodie|Joggers|Sweatpants|Pullover|Cardigan|Tee"],
       ["Machine wash cold, tumble dry low.", "Hand wash recommended.",
        "True to size; size up for a looser fit.", "Model is {n:170-190} cm and wears a size M.@Jacket|Shirt|Trousers|Sweater|Dress|Tee|Jeans|Hoodie|Blazer|Cardigan|Coat|Polo",
        "Pre-shrunk.", "Made in Portugal.", "Made in Vietnam.", "Dry clean only.@Blazer|Coat|Cashmere"],
       ["- Navy", "- Black", "- Olive", "- Heather Grey", "- Ivory", "- Rust", "- Sage",
        "- Size S", "- Size M", "- Size L", "- Size XL"],
       ["Northfold", "Marlowe", "Juniper & Co", "Aster", "Fennick", "Calder", "Wren",
        "Solace", "Halden", "Common Thread"],
       45.0, material_in_name=True),
    _F("footwear", "Shoes",
       ["Running Shoes", "Chelsea Boots", "Canvas Sneakers", "Hiking Boots", "Loafers",
        "Slides", "Sandals", "Trail Runners", "High-Top Sneakers", "Ankle Boots",
        "Slip-On Shoes", "Court Shoes", "Clogs", "Walking Shoes"],
       ["Lightweight", "Waterproof@Boots|Hiking|Trail|Walking", "Cushioned", "Classic", "Low-Profile", "Grippy",
        "Everyday", "Wide-Fit", "Breathable"],
       ["full-grain leather@Boots|Loafers|Shoes|Sandals|Sneakers", "suede@Boots|Loafers|Shoes|Sneakers", "recycled mesh@Running|Trail|Walking|Sneakers", "canvas@Sneakers|Slip-On|High-Top", "vegan leather", "knit@Running|Sneakers|Slip-On|Walking"],
       ["a cushioned midsole", "a rubber outsole", "a padded collar", "a removable insole",
        "a pull tab@Boots|Sneakers|Runners|Shoes", "a reinforced toe cap", "a lugged sole@Hiking|Trail|Boots", "a breathable lining"],
       ["daily runs@Running|Trail", "city walking", "long days on your feet", "light trails@Hiking|Trail|Walking",
        "the office@Loafers|Chelsea|Court|Ankle", "summer weekends"],
       ["Fits true to size.", "Half sizes available.", "Wipe clean with a damp cloth.",
        "Heel drop {n:4-10} mm.@Running|Trail|Walking"],
       ["- Size 8", "- Size 9", "- Size 10", "- Size 11", "- White", "- Black", "- Tan",
        "- Chestnut"],
       ["Stride", "Fennick", "Trailborn", "Halden", "Cobble & Co", "Northfold"],
       85.0, material_in_name=True),
    _F("home", "Home",
       ["Throw Blanket", "Sheet Set", "Memory Foam Pillow", "Aroma Diffuser", "Air Purifier",
        "Cordless Vacuum", "Table Lamp", "Bath Towel Set", "Duvet Cover", "Wall Mirror",
        "Storage Basket", "Scented Candle", "Blackout Curtains", "Area Rug", "Photo Frame",
        "Laundry Hamper", "Doormat", "Shower Curtain", "Cushion Cover", "Wall Clock"],
       ["Soft@Blanket|Towel|Pillow|Throw|Cushion|Sheet|Duvet|Rug", "Minimalist", "Handwoven@Rug|Basket|Throw|Blanket|Doormat", "Washable@Rug|Pillow|Cushion|Blanket", "Compact@Vacuum|Purifier|Diffuser|Lamp", "Oversized@Throw|Blanket|Mirror|Clock|Pillow", "Quiet@Vacuum|Purifier|Diffuser|Clock", "Modern", "Rustic@Frame|Clock|Mirror|Basket|Candle|Lamp", "Plush@Blanket|Throw|Rug|Pillow|Towel"],
       ["cotton percale@Sheet|Duvet|Pillow", "linen@Sheet|Duvet|Cushion|Curtain|Throw|Blanket", "jute@Rug|Basket|Doormat|Hamper", "bamboo@Towel|Basket|Hamper|Frame|Sheet", "ceramic@Lamp|Diffuser|Candle", "oak@Frame|Mirror|Clock|Lamp", "recycled glass@Candle|Lamp|Diffuser", "velvet@Cushion|Curtain|Throw", "wool@Throw|Blanket|Rug|Cushion", "recycled plastic@Vacuum|Purifier|Hamper|Diffuser|Curtain", "organic cotton@Towel|Blanket|Throw|Sheet|Rug"],
       ["a neutral colourway", "easy-care materials", "a {n:1-5}-year guarantee", "a neutral finish@Lamp|Mirror|Frame|Clock|Basket|Hamper|Candle|Diffuser", "hidden fixings@Mirror|Clock|Frame|Curtains", "a non-slip base@Rug|Doormat|Lamp|Basket|Hamper|Board|Scale|Mixer|Blender|Bed|Bowl", "a removable cover@Pillow|Cushion|Duvet|Hamper",
        "a hand-finished edge@Rug|Towel|Blanket|Throw|Mirror|Frame|Doormat|Basket|Cushion", "a timer function@Diffuser|Purifier|Lamp", "a whisper-quiet motor@Vacuum|Purifier|Diffuser",
        "a washable filter@Vacuum|Purifier", "pre-drilled holes@Mirror|Clock|Frame", "a soft-touch weave@Blanket|Towel|Sheet|Rug|Cushion|Duvet|Throw"],
       ["the living room", "small flats", "guest rooms", "everyday use", "the bedroom",
        "entryways", "rentals"],
       ["Machine washable at 40°C.@Blanket|Sheet|Towel|Duvet|Cushion|Curtain|Throw|Pillow", "Spot clean only.@Rug|Cushion|Throw|Curtains|Basket|Doormat|Pillow|Hamper", "Assembly takes about {n:5-30} minutes.@Mirror|Hamper|Lamp|Clock|Rack",
        "Measures {n:30-200} x {n:30-200} cm.", "Covered by a {n:1-5}-year warranty."],
       ["- Set of 2", "- Set of 4", "- Large", "- Small", "- Natural", "- Charcoal",
        "- Oatmeal", "- Queen", "- King"],
       ["Hearthwell", "Oakline", "Nordhaus", "Casa Verde", "Linden", "Tidewater",
        "Ember & Ash", "Kinfolk Home"],
       39.0, material_in_name=True),
    _F("kitchen", "Kitchen",
       ["Cast Iron Skillet", "Knife Set", "Cookware Set", "Cutting Board", "French Press",
        "Pour-Over Kettle", "Dinnerware Set", "Food Storage Containers", "Baking Mat",
        "Kitchen Scale", "Stand Mixer", "Blender", "Air Fryer", "Toaster", "Coffee Grinder",
        "Spice Rack", "Dutch Oven", "Water Bottle", "Travel Mug", "Salad Spinner"],
       ["Non-Stick@Skillet|Cookware|Mat|Air Fryer", "Pre-Seasoned@Skillet|Dutch Oven", "Stainless@Knife|Kettle|Bottle|Mug|Rack|Cookware", "Insulated@Bottle|Mug", "Compact", "Professional@Knife|Mixer|Blender|Grinder|Cookware", "Stackable@Containers|Dinnerware", "Leak-Proof@Containers|Bottle|Mug", "Digital@Scale|Air Fryer|Toaster|Kettle"],
       ["18/10 stainless steel@Knife|Cookware|Kettle|Bottle|Mug|Rack|Toaster|Press|Mixer|Grinder|Scale|Spinner", "cast iron@Skillet|Dutch Oven", "acacia wood@Board|Rack", "borosilicate glass@Containers|Press|Kettle|Bottle", "stoneware@Dinnerware|Mug|Dutch Oven", "BPA-free plastic@Containers|Spinner|Bottle|Blender|Air Fryer", "carbon steel@Knife|Skillet", "enamelled cast iron@Dutch Oven|Skillet", "silicone@Mat|Rack|Containers"],
       ["a {n:1-10}-year guarantee", "easy-clean surfaces", "a compact footprint", "stay-cool handles@Skillet|Cookware|Dutch Oven|Kettle", "a non-slip base@Board|Scale|Mixer|Blender|Rack|Spinner|Grinder", "a pour spout@Kettle|Press|Skillet|Blender|Cookware", "a tempered glass lid@Cookware|Dutch Oven|Skillet",
        "an induction-ready base@Skillet|Cookware|Dutch Oven|Kettle", "measurement markings@Blender|Containers|Kettle|Press|Bottle|Mixer", "a locking lid@Containers|Bottle|Mug|Blender|Lunch",
        "{n:3-12} speed settings@Mixer|Blender|Grinder", "a removable blade@Blender|Grinder", "an auto shut-off@Kettle|Air Fryer|Toaster|Blender|Mixer|Grinder"],
       ["weeknight cooking", "small kitchens", "meal prep", "baking@Mat|Mixer|Scale|Dutch Oven", "camping@Bottle|Mug|Skillet|Press|Kettle",
        "the morning coffee@Press|Kettle|Grinder|Mug", "entertaining"],
       ["Dishwasher safe.@Board|Dinnerware|Containers|Mat|Bottle|Mug|Spinner|Cookware", "Hand wash to keep the finish.", "Oven safe to {n:200-260}°C.@Skillet|Dutch Oven|Cookware|Mat|Dinnerware",
        "Holds {n:1-6} litres.@Dutch Oven|Kettle|Blender|Air Fryer|Mixer|Containers|Cookware", "Works on gas, electric and induction hobs.@Skillet|Cookware|Dutch Oven|Kettle"],
       ["- 10\"@Skillet", "- 12\"@Skillet", "- 6-Piece@Set", "- 12-Piece@Set", "- Black", "- Sage", "- Cream",
        "- 1.5L@Kettle|Blender|Containers|Bottle", "- 500ml@Bottle|Mug|Containers"],
       ["Hearthwell", "Copperleaf", "Kitchenry", "Old Mill", "Larder & Co", "Saltbox"],
       42.0, material_in_name=True),
    _F("furniture", "Furniture",
       ["Office Chair", "Bookshelf", "Coffee Table", "Desk", "Bed Frame", "Sofa",
        "Bar Stool", "Dining Chair", "Nightstand", "TV Stand", "Shoe Rack", "Armchair",
        "Wardrobe", "Side Table"],
       ["Mid-Century", "Ergonomic@Chair|Desk", "Adjustable@Chair|Desk|Stool|Bed", "Modular", "Solid Wood@Table|Desk|Bookshelf|Bed|Nightstand|Wardrobe|Stand", "Folding@Chair|Table|Desk|Stool",
        "Minimalist", "Upholstered@Chair|Sofa|Armchair|Bed|Stool"],
       ["solid oak", "walnut veneer@Table|Desk|Bookshelf|Nightstand|Stand|Wardrobe", "powder-coated steel@Stool|Desk|Rack|Table|Bed|Chair", "rattan@Chair|Armchair|Table|Nightstand", "boucle@Sofa|Armchair|Chair|Stool", "pine@Bookshelf|Bed|Rack|Wardrobe|Nightstand"],
       ["adjustable feet", "soft-close drawers@Desk|Nightstand|TV Stand|Wardrobe|Side Table", "lumbar support@Chair|Armchair|Sofa", "a cable cut-out@Desk|TV Stand",
        "removable cushions@Sofa|Armchair|Chair|Stool", "an anti-tip kit", "{n:2-5} shelves@Bookshelf|Shoe Rack|TV Stand|Wardrobe", "a weight limit of {n:80-150} kg"],
       ["small spaces", "the home office", "living rooms", "studio flats", "dining rooms"],
       ["Assembly required; tools included.", "Ships flat-packed in {n:1-3} boxes.",
        "Wipe clean with a dry cloth.", "Measures {n:40-200} cm wide."],
       ["- Oak", "- Walnut", "- Black", "- White", "- Grey"],
       ["Nordhaus", "Oakline", "Linden", "Fernwood", "Studio Arlo"],
       189.0, material_in_name=True),
    _F("beauty", "Beauty",
       ["Vitamin C Serum", "Moisturiser", "Night Cream", "Sunscreen SPF 50",
        "Cleansing Water", "Clay Mask", "Hair Oil", "Shampoo", "Conditioner", "Lipstick",
        "Eyeshadow Palette", "Mascara", "Setting Powder", "Tinted Moisturiser",
        "Brow Pencil", "Sheet Masks", "Body Lotion", "Hand Cream", "Eye Cream",
        "Face Cleanser", "Toner"],
       ["Hydrating", "Brightening@Serum|Mask|Cream|Toner|Moisturiser", "Fragrance-Free", "Gentle", "Long-Wear@Lipstick|Mascara|Eyeshadow|Powder|Pencil", "Matte@Lipstick|Powder|Eyeshadow|Sunscreen", "Nourishing", "Lightweight", "Overnight@Cream|Mask|Serum"],
       ["hyaluronic acid", "niacinamide@Serum|Moisturiser|Toner|Cream", "shea butter@Cream|Lotion|Lipstick|Conditioner", "squalane", "aloe vera", "jojoba oil", "ceramides@Cream|Moisturiser|Lotion|Cleanser", "green tea extract"],
       ["a non-greasy finish", "a pump bottle@Serum|Moisturiser|Cleanser|Shampoo|Conditioner|Lotion|Oil|Toner", "a fresh citrus scent@Shampoo|Conditioner|Lotion|Hand Cream|Cleanser", "no added fragrance",
        "a recyclable tube@Cream|Sunscreen|Cleanser|Lotion|Moisturiser|Mascara", "a buildable formula@Lipstick|Eyeshadow|Mascara|Powder|Tinted|Brow", "a travel-size option"],
       ["dry skin", "sensitive skin", "daily use", "oily skin@Cleanser|Toner|Mask|Powder|Moisturiser|Sunscreen", "a morning routine", "a night routine@Cream|Serum|Mask|Oil", "all skin types"],
       ["Dermatologist tested.", "Vegan and cruelty-free.", "Contains {n:30-200} ml.",
        "Patch test before first use.", "Apply morning and evening to clean skin.@Serum|Moisturiser|Cream|Toner"],
       ["30ml", "50ml", "100ml", "Travel Size", "- Shade 02", "- Shade 05", "Unscented"],
       ["Lumière", "Botanica", "Velour", "Pure Theory", "Saffron Lane", "Dewy Days"],
       24.0),
    _F("sports", "Sports",
       ["Resistance Bands", "Adjustable Dumbbells", "Yoga Mat", "Pull-Up Bar", "Jump Rope",
        "Foam Roller", "Cycling Helmet", "Tennis Racket", "Basketball", "Football",
        "Swim Goggles", "Gym Bag", "Kettlebell", "Camping Tent", "Sleeping Bag",
        "Hiking Backpack", "Trekking Poles", "Water Bottle", "Bike Light", "Climbing Chalk Bag"],
       ["Non-Slip@Mat", "Lightweight", "Adjustable@Dumbbells|Helmet|Poles|Rope|Bar", "Heavy-Duty", "Packable@Tent|Sleeping|Backpack|Bag", "Anti-Fog@Goggles", "Insulated@Bottle|Sleeping", "Pro", "Compact"],
       ["natural rubber@Bands|Mat|Basketball|Football|Goggles", "TPE foam@Mat|Roller", "ripstop nylon@Tent|Sleeping|Backpack|Bag", "cast iron@Kettlebell|Dumbbells", "aluminium alloy@Poles|Bar|Racket|Light|Bottle", "recycled polyester@Bag|Backpack|Sleeping|Bands"],
       ["a durable build", "a {n:1-2}-year guarantee", "a lightweight design", "a carry strap@Mat|Bag|Tent|Sleeping|Roller|Backpack", "anti-slip texture@Mat|Bar|Dumbbells|Kettlebell|Racket|Poles", "{n:3-6} resistance levels@Bands", "a ventilated shell@Helmet",
        "a quick-release buckle@Helmet|Backpack|Bag|Light", "a rain cover@Backpack|Bag|Tent", "reflective details@Helmet|Backpack|Bag|Light", "a padded grip@Racket|Rope|Poles|Bar|Dumbbells|Kettlebell"],
       ["home workouts@Bands|Dumbbells|Mat|Bar|Rope|Roller|Kettlebell", "the gym@Bands|Dumbbells|Mat|Bag|Rope|Roller|Kettlebell|Bottle", "weekend hikes@Backpack|Poles|Bottle|Tent", "yoga classes@Mat|Roller|Bands", "camping trips@Tent|Sleeping|Backpack|Light|Bottle", "training sessions", "commuting by bike@Helmet|Light|Backpack"],
       ["Weighs {n:200-2500} g.", "Supports up to {n:80-150} kg.@Bar|Roller|Mat", "Wipe clean after use.",
        "Packs down to {n:20-45} cm.@Tent|Sleeping|Backpack|Poles"],
       ["- Blue", "- Red", "- Black", "- Size 5@Football|Basketball", "- Medium@Bands|Helmet|Backpack|Bag|Tent", "- Large@Bands|Helmet|Backpack|Bag|Tent", "- 6mm@Mat", "- 15kg@Kettlebell|Dumbbells"],
       ["Peakform", "Stride", "Trailborn", "Vantage", "Ironbark", "Swiftline", "Summitry"],
       35.0),
    _F("toys", "Toys",
       ["Building Blocks Set", "Wooden Train Set", "Plush Bear", "Jigsaw Puzzle",
        "Board Game", "Remote Control Car", "Doll House", "Art Kit", "Science Kit",
        "Stacking Rings", "Kite", "Play Kitchen", "Card Game", "Marble Run", "Ride-On Scooter",
        "Magnetic Tiles", "Puppet Theatre", "Drum Set"],
       ["Classic", "Wooden@Blocks|Train|Puzzle|Kitchen|House|Rings|Theatre", "Educational", "Glow-in-the-Dark@Puzzle|Tiles|Kite|Blocks", "Rainbow", "Junior",
        "Deluxe", "Travel"],
       ["FSC-certified wood@Blocks|Train|Kitchen|House|Rings|Theatre|Marble", "recycled plastic@Car|Scooter|Tiles|Kite|Drum|Blocks|Rings", "soft plush@Bear|Puppet", "cardboard@Puzzle|Board|Card|Kit"],
       ["{n:24-1000} pieces@Blocks|Puzzle|Tiles|Marble|Train", "a storage tin@Blocks|Puzzle|Card|Art|Tiles|Marble", "rounded edges", "rechargeable batteries@Car|Scooter|Drum",
        "illustrated instructions", "{n:2-6} play modes@Car|Kitchen|Drum|Science|Tiles", "a carry case"],
       ["rainy afternoons", "family game night@Board|Card|Puzzle", "toddlers@Blocks|Rings|Bear|Train|Tiles|Drum", "budding scientists@Science|Marble|Blocks|Tiles",
        "birthday gifts", "travel"],
       ["Recommended for ages {n:3-10} and up.", "Not suitable for children under 3.",
        "Batteries included.@Car|Drum|Science", "Contains small parts.", "For {n:2-6} players.@Board|Card"],
       ["- 100 Pieces", "- 500 Pieces", "- Pastel", "- Primary Colours", "Mini", "XL"],
       ["Little Oak", "Tumbletown", "Brightbox", "Kitebird", "Pip & Pals"],
       29.0),
    _F("books", "Books",
       ["Field Guide", "Cookbook", "Novel", "Short Story Collection", "Memoir", "Atlas",
        "Workbook", "Travel Guide", "Poetry Collection", "Graphic Novel", "History",
        "Biography", "Puzzle Book", "Picture Book"],
       ["Illustrated", "Pocket", "Collector's", "Annotated", "Complete", "Beginner's",
        "Revised"],
       ["paperback", "hardcover", "clothbound"],
       ["{n:120-640} pages", "full-colour photographs@Guide|Cookbook|Atlas|Picture", "an index", "a foreword by the editor",
        "maps and diagrams@Guide|Atlas|History", "a ribbon marker"],
       ["curious readers", "gift giving", "beginners", "book clubs", "long train rides"],
       ["Published {n:2015-2025}.", "{n:120-640} pages.", "Also available as an e-book."],
       ["(Paperback)", "(Hardcover)", "(2nd Edition)", "- Illustrated Edition"],
       ["Wren Press", "Kestrel Books", "Lantern House", "Old Harbour Publishing"],
       18.0),
    _F("grocery", "Grocery",
       ["Ground Coffee", "Green Tea", "Olive Oil", "Almond Butter", "Granola", "Pasta",
        "Hot Sauce", "Honey", "Dark Chocolate", "Rolled Oats", "Sea Salt", "Protein Bars",
        "Sparkling Water", "Basmati Rice", "Maple Syrup", "Trail Mix", "Peanut Butter",
        "Coconut Milk"],
       ["Organic", "Single-Origin@Coffee|Tea|Chocolate", "Cold-Pressed@Olive Oil", "Small-Batch", "Unsweetened@Butter|Milk|Granola|Water", "Gluten-Free@Pasta|Granola|Oats|Bars", "Smoked@Salt|Hot Sauce", "Wildflower@Honey", "Roasted@Coffee|Almond|Peanut|Mix"],
       ["Arabica beans@Coffee", "whole grains@Granola|Oats|Pasta|Rice|Bars", "Spanish olives@Olive Oil", "roasted almonds@Almond|Granola|Mix|Bars", "cacao@Chocolate", "green tea leaves@Tea", "wildflower nectar@Honey", "peanuts@Peanut", "chillies@Hot Sauce", "coconut@Coconut", "maple sap@Syrup", "sea water@Salt"],
       ["no added sugar", "a resealable bag@Coffee|Granola|Oats|Mix|Rice|Tea|Bars", "a glass jar@Honey|Butter|Sauce|Syrup|Salt", "{n:8-30} servings",
        "fair-trade sourcing", "a rich, nutty flavour@Butter|Granola|Mix|Coffee|Chocolate"],
       ["breakfast", "the pantry", "snacking on the go", "baking", "weeknight dinners"],
       ["Best before {n:6-24} months from packing.", "Store in a cool, dry place.",
        "May contain traces of nuts.", "Net weight {n:100-1000} g."],
       ["250g", "500g", "1kg", "Pack of 6", "Pack of 12", "750ml"],
       ["Harvest Table", "Golden Acre", "Wildroot", "Old Mill", "Sunny Ridge", "Larder & Co"],
       9.0),
    _F("pets", "Pet Supplies",
       ["Dog Bed", "Cat Tree", "Dog Lead", "Chew Toy", "Cat Litter", "Dog Food",
        "Pet Carrier", "Feeding Bowl", "Grooming Brush", "Dog Harness", "Scratching Post",
        "Treat Pouch"],
       ["Orthopaedic@Bed", "Washable@Bed|Carrier", "Reflective@Lead|Harness", "Durable", "Calming@Bed|Toy", "Grain-Free@Food", "Adjustable@Lead|Harness|Carrier", "Clumping@Litter", "Natural"],
       ["memory foam@Bed", "natural rubber@Toy|Bowl", "stainless steel@Bowl|Brush", "sisal rope@Tree|Post|Toy", "nylon webbing@Lead|Harness|Pouch|Carrier", "natural clay@Litter", "real chicken@Food"],
       ["pet-safe materials", "an easy-clean finish", "a durable build", "a non-slip base@Bed|Bowl|Tree|Post", "a removable cover@Bed|Carrier", "reflective stitching@Lead|Harness", "a padded handle@Lead|Carrier|Harness",
        "a quick-release clip@Lead|Harness|Pouch"],
       ["large dogs", "indoor cats", "puppies", "daily walks", "travel"],
       ["Machine washable.@Bed|Carrier|Pouch", "Fits pets up to {n:5-40} kg.@Bed|Carrier|Harness|Tree", "Supervise during play.@Toy|Tree|Post"],
       ["- Small", "- Medium", "- Large", "- Grey", "- Navy", "2kg@Food|Litter", "10kg@Food|Litter"],
       ["Pawsome", "Waggle", "Furrow & Co", "Tailwind"],
       27.0),
    _F("garden", "Garden",
       ["Garden Hose", "Pruning Shears", "Raised Planter", "Solar Lights", "Bird Feeder",
        "Watering Can", "Patio Chair", "Compost Bin", "Seed Starter Kit", "Garden Gloves",
        "Lawn Sprinkler", "Plant Pot"],
       ["Expandable", "Weatherproof", "Heavy-Duty", "Self-Watering", "Ergonomic",
        "Solar-Powered"],
       ["galvanised steel", "terracotta", "cedar wood", "recycled plastic", "powder-coated steel"],
       ["drainage holes", "a brass fitting", "UV-resistant finish", "a locking handle",
        "{n:6-12} spray patterns"],
       ["balconies", "vegetable patches", "small gardens", "patios", "spring planting"],
       ["Leave outside year-round.", "Measures {n:20-120} cm.", "Rinse after use."],
       ["- Green", "- Terracotta", "- 15m", "- 30m", "- Set of 3"],
       ["Greenhaven", "Fernwood", "Bloom & Root", "Allotment Co"],
       32.0),
    _F("office", "Office Supplies",
       ["Notebook", "Gel Pens", "Desk Organiser", "Planner", "Sticky Notes", "Stapler",
        "Filing Box", "Highlighters", "Whiteboard", "Desk Mat", "Fountain Pen", "Label Maker"],
       ["Dotted", "Refillable", "A5", "Recycled", "Magnetic", "Quick-Dry", "Undated"],
       ["recycled paper", "vegan leather", "bamboo", "aluminium", "cork"],
       ["{n:80-240} pages", "an elastic closure", "a pen loop", "{n:6-24} colours",
        "a lay-flat binding"],
       ["planning the week", "the home office", "students", "journalling", "meetings"],
       ["{n:80-120} gsm paper.", "Pack of {n:3-24}.", "Acid-free pages."],
       ["- Black", "- Sage", "- Pack of 6", "- Pack of 12", "- A4", "- A5"],
       ["Quillery", "Paperline", "Inkwell & Co", "Deskwise"],
       14.0),
    _F("automotive", "Automotive",
       ["Car Phone Mount", "Tyre Inflator", "Jump Starter", "Seat Covers", "Dash Camera",
        "Car Vacuum", "Floor Mats", "Wiper Blades", "Roof Box", "Car Charger"],
       ["Universal", "Heavy-Duty", "Portable", "All-Weather", "Magnetic", "Cordless"],
       ["rubber", "aluminium", "neoprene", "ABS plastic"],
       ["a {n:12-24} V plug", "an LED torch", "a digital gauge", "a universal fit",
        "a storage bag"],
       ["road trips", "winter driving", "daily commutes", "SUVs", "family cars"],
       ["Fits most vehicles.", "Cable length {n:1-4} m.", "Check fitment before ordering."],
       ["- Black", "- Pair", "- 22\"", "- 26\""],
       ["Roadwise", "Torque & Co", "Milepost", "Gearline"],
       38.0),
    _F("jewelry", "Jewellery",
       ["Pendant Necklace", "Hoop Earrings", "Stud Earrings", "Bracelet", "Ring",
        "Watch", "Charm Bracelet", "Cufflinks", "Anklet", "Chain"],
       ["Minimal", "Dainty", "Classic", "Hammered", "Stackable", "Vintage"],
       ["sterling silver", "14k gold vermeil", "stainless steel", "rose gold plate"],
       ["a lobster clasp", "an adjustable chain", "a gift box", "hypoallergenic posts",
        "a polished finish"],
       ["everyday wear", "gifts", "layering", "special occasions"],
       ["Nickel-free.", "Chain length {n:40-50} cm.", "Avoid contact with water and perfume."],
       ["- Gold", "- Silver", "- Rose Gold", "- 18\""],
       ["Aurelia", "Ondine", "Mira & Co", "Silverline"],
       55.0, material_in_name=True),
    _F("health", "Health",
       ["Vitamin D Tablets", "Multivitamin", "Omega-3 Capsules", "Protein Powder",
        "Electrolyte Tablets", "Magnesium Capsules", "First Aid Kit", "Digital Thermometer",
        "Massage Gun", "Heating Pad", "Blood Pressure Monitor", "Probiotic Capsules"],
       ["Daily", "High-Strength", "Vegan", "Unflavoured", "Slow-Release", "Compact"],
       ["plant-based capsules", "whey isolate", "fish oil", "magnesium glycinate"],
       ["{n:30-120} servings", "no artificial colours", "a child-resistant cap",
        "an easy-read display", "{n:3-6} intensity levels"],
       ["daily wellness", "after workouts", "the travel bag", "busy weeks"],
       ["Do not exceed the stated dose.", "Store below 25°C.",
        "Consult your doctor if pregnant or taking medication."],
       ["- 60 Capsules", "- 90 Tablets", "- 500g", "- 1kg", "- Vanilla", "- Chocolate"],
       ["VitaCore", "Wellspring", "Pure Theory", "Northwell"],
       22.0),
    _F("baby", "Baby",
       ["Baby Carrier", "Swaddle Blanket", "Bottle Set", "Teething Ring", "Baby Monitor",
        "Changing Mat", "Sleepsuit", "High Chair", "Bath Seat", "Play Mat"],
       ["Soft", "Organic", "Breathable", "Foldable", "Easy-Clean", "Anti-Colic"],
       ["organic cotton", "muslin", "food-grade silicone", "bamboo viscose"],
       ["poppers down the front", "a padded headrest", "BPA-free parts", "a night-light",
        "adjustable straps"],
       ["newborns", "night feeds", "nursery essentials", "travel"],
       ["Suitable from birth.", "Machine washable at 30°C.", "Meets EN 71 safety standards."],
       ["- 0-3 Months", "- 3-6 Months", "- Pack of 3", "- Oat", "- Sage"],
       ["Little Oak", "Nestling", "Pip & Pals", "Cradle Co"],
       34.0),
    _F("tools", "Tools",
       ["Cordless Drill", "Screwdriver Set", "Tool Box", "Tape Measure", "Spirit Level",
        "Socket Set", "Work Light", "Utility Knife", "Stud Finder", "Glue Gun", "Ladder",
        "Workbench"],
       ["Cordless", "Heavy-Duty", "Compact", "Magnetic", "Professional", "Folding"],
       ["chrome vanadium steel", "aluminium", "hardened steel", "impact-resistant plastic"],
       ["a {n:12-20} V battery", "an LED work light", "a carry case", "a belt clip",
        "{n:20-120} pieces", "a soft-grip handle"],
       ["DIY projects", "the garage", "flat-pack furniture", "small repairs", "trades"],
       ["Battery and charger included.", "Covered by a {n:2-5}-year warranty.",
        "Weighs {n:300-2000} g."],
       ["- 18V", "- 32-Piece", "- 5m", "- Yellow", "- Kit"],
       ["Ironbark", "Forgewell", "Torque & Co", "Benchmark Tools"],
       48.0),
    _F("generic", "General",
       ["Gift Set", "Storage Box", "Travel Kit", "Multi-Tool", "Tote Bag", "Desk Organiser",
        "Water Bottle", "Umbrella", "Backpack", "Keyring", "Lunch Box", "Phone Case",
        "Picnic Blanket", "Wallet"],
       ["Compact", "Classic", "Everyday", "Durable", "Lightweight", "Foldable", "Premium"],
       ["recycled materials", "canvas", "stainless steel", "vegan leather", "bamboo"],
       ["a zip closure", "a lifetime guarantee", "a carry strap", "a gift box",
        "a water-resistant finish", "an inner pocket"],
       ["everyday use", "travel", "gifts", "the office", "weekends away"],
       ["Wipe clean.", "Measures {n:10-60} cm.", "Ships in plastic-free packaging."],
       ["- Black", "- Navy", "- Sand", "- Large", "- Small"],
       ["Wayfarer", "Common Thread", "Meridian Goods", "Harbor & Pine"],
       25.0),
]}

# Category label -> family, first match wins. Ordered so "kids clothing"
# reads as clothing and "kitchen appliances" as kitchen, not electronics.
_FAMILY_KEYWORDS = [
    ("footwear", ("shoe", "footwear", "sneaker", "boot")),
    ("clothing", ("cloth", "apparel", "fashion", "garment", "wear", "dress", "outfit")),
    ("kitchen", ("kitchen", "cook", "dining", "tableware", "appliance")),
    ("furniture", ("furniture", "furnishing")),
    ("baby", ("baby", "infant", "nursery", "toddler", "maternity")),
    ("toys", ("toy", "game", "kids", "puzzle", "hobby", "hobbies")),
    ("beauty", ("beauty", "cosmetic", "skin", "makeup", "make-up", "hair", "fragrance",
                "personal care", "grooming")),
    ("health", ("health", "wellness", "supplement", "pharma", "vitamin", "medical")),
    ("sports", ("sport", "fitness", "outdoor", "gym", "athlet", "camping", "cycling")),
    ("books", ("book", "literature", "reading", "media")),
    ("grocery", ("food", "grocer", "snack", "beverage", "drink", "pantry", "coffee",
                 "tea", "gourmet")),
    ("pets", ("pet", "dog", "cat", "animal")),
    ("garden", ("garden", "lawn", "patio", "plant")),
    ("office", ("office", "stationery", "school", "supplies", "paper")),
    ("automotive", ("auto", "car ", "cars", "vehicle", "motor")),
    ("jewelry", ("jewel", "watch", "accessor")),
    ("tools", ("tool", "hardware", "diy", "industrial", "improvement")),
    ("electronics", ("electronic", "tech", "gadget", "computer", "audio", "phone",
                     "camera", "gaming", "tv", "laptop")),
    ("home", ("home", "decor", "bedding", "bath", "household", "living", "house")),
]

# When the table has no category to follow: roughly how a general retailer's
# catalogue splits across departments.
_DEFAULT_FAMILY_MIX = [("electronics", 0.18), ("clothing", 0.2), ("home", 0.14),
                       ("kitchen", 0.08), ("beauty", 0.1), ("sports", 0.09),
                       ("toys", 0.05), ("footwear", 0.05), ("books", 0.04),
                       ("grocery", 0.04), ("pets", 0.03)]


def family_for_category(category) -> Optional[str]:
    """The product family a category label names, or None."""
    c = f" {str(category).strip().lower()} "
    if c.strip() in ("", "nan", "none"):
        return None
    for key, words in _FAMILY_KEYWORDS:
        if any(w in c for w in words):
            return key
    return None


@dataclass
class ProductFrame:
    family: List[str]
    brand: List[str]
    noun: List[str]
    attr: List[str]
    material: List[str]
    name: List[str]


def draw_products(rng: np.random.Generator, size: int,
                  categories: Optional[Sequence] = None,
                  key: str = "products") -> ProductFrame:
    """One latent product per row, following ``categories`` when given."""
    cats = _clean_values(categories, size)
    if cats is not None:
        fam = [family_for_category(c) or "generic" for c in cats]
    else:
        keys, w = zip(*_DEFAULT_FAMILY_MIX)
        w = np.array(w) / sum(w)
        fam = [keys[i] for i in rng.choice(len(keys), size=size, p=w)]
    fam_arr = np.array(fam, dtype=object)
    brand = np.empty(size, dtype=object)
    noun = np.empty(size, dtype=object)
    attr = np.empty(size, dtype=object)
    material = np.empty(size, dtype=object)
    names = np.empty(size, dtype=object)
    for f in sorted(set(fam)):
        idx = np.flatnonzero(fam_arr == f)
        F = PRODUCT_FAMILIES[f]
        n = len(idx)
        # Brands are concentrated, product types less so.
        brand[idx] = zipf_choice(rng, F.brands, n, s=1.1, key=f"{key}|{f}|brand")
        noun[idx] = zipf_choice(rng, F.nouns, n, s=0.7, q=4, key=f"{key}|{f}|noun")
        for i in idx:
            a_pool = eligible(F.attrs, noun[i]) or [""]
            m_pool = eligible(F.materials, noun[i]) or [""]
            attr[i] = a_pool[int(rng.integers(len(a_pool)))]
            material[i] = m_pool[int(rng.integers(len(m_pool)))]
        use_brand = rng.random(n) < 0.6
        use_attr = rng.random(n) < 0.55
        use_mat = (rng.random(n) < 0.3) if F.material_in_name else np.zeros(n, bool)
        use_var = rng.random(n) < 0.4
        variants = [eligible(F.variants, noun[i]) for i in idx]
        variants = [v[int(rng.integers(len(v)))] if v else "" for v in variants]
        for j, i in enumerate(idx):
            parts = []
            if use_brand[j]:
                parts.append(brand[i])
            if use_attr[j] and attr[i]:
                parts.append(attr[i])
            if use_mat[j] and material[i] and str(material[i]).lower() not in str(noun[i]).lower():
                parts.append(" ".join(w.capitalize() if w.islower() else w
                                      for w in str(material[i]).split()))
            parts.append(noun[i])
            nm = " ".join(parts)
            if use_var[j] and variants[j]:
                nm = f"{nm} {variants[j]}"
            names[i] = nm
    return ProductFrame(list(fam), list(brand), list(noun), list(attr),
                        list(material), list(names))


# (weight, template, features it consumes)
_DESC_OPENERS = [
    (3, "{A_attr_noun} {made} {material}.", 0),
    (2, "{This} {attr} {noun} {is} {made} {material}.", 0),
    (2, "{Brand}'s {attr} {noun}, in {material}.", 0),
    (2, "The {noun} you reach for every day, now in {material}.", 0),
    (2, "A {attr} take on the classic {noun}.", 0),
    (1, "Our best-selling {noun}, refined.", 0),
    (1, "Meet the {brand} {noun}.", 0),
    (2, "{A_noun} built around {f1}.", 1),
    (1, "{Noun} with {f1} and {f2}.", 2),
]
_DESC_FEATURES = {
    1: [(2, "Comes with {f1}."), (2, "Includes {f1}."), (2, "Features {f1}.")],
    2: [(3, "Features {f1} and {f2}."), (2, "Designed with {f1} and {f2}."),
        (1, "Comes with {f1} and {f2}.")],
    3: [(2, "It has {f1}, {f2} and {f3}."), (1, "{F1}, {f2} and {f3} come as standard."),
        (1, "Features {f1}, {f2} and {f3}.")],
}
_DESC_USES = [
    (3, "Ideal for {use}."), (2, "Made for {use}."), (2, "A good pick for {use}."),
    (1, "Works well for {use}."), (1, "Great for {use} and {use2}."),
    (1, "Perfect for {use}."),
]
_NOT_PLURAL = {"atlas", "canvas", "glass", "dress", "harness", "press", "gloss"}


def is_plural(noun: str) -> bool:
    last = noun.split()[-1].lower() if noun.split() else ""
    return (last.endswith("s") and not last.endswith(("ss", "us", "is"))
            and last not in _NOT_PLURAL)


def _soft_lower(phrase: str) -> str:
    """Lowercase ordinary words, keep acronyms and model names (SPF, USB-C, V-Neck)."""
    return " ".join(w if any(ch.isupper() for ch in w[1:]) or w[:1].isdigit() else w.lower()
                    for w in phrase.split())


_ELIGIBLE: Dict[tuple, List[str]] = {}


def eligible(pool: Sequence[str], noun: str) -> List[str]:
    """Entries of ``pool`` that fit ``noun``.

    An entry may end in ``@Kw1|Kw2``: it then fits only nouns containing one
    of those words ("a washable filter@Vacuum|Purifier"). Untagged entries fit
    every noun in the family. A noun no tagged entry names gets the untagged
    ones, so nothing is left empty while a vacuum never gets a "soft-touch
    weave"."""
    k = (id(pool), noun)
    hit = _ELIGIBLE.get(k)
    if hit is not None:
        return hit
    tagged, plain = [], []
    for e in pool:
        if "@" in e:
            text, kws = e.rsplit("@", 1)
            if any(w.lower() in noun.lower() for w in kws.split("|")):
                tagged.append(text)
        else:
            plain.append(e)
    out = tagged + plain
    _ELIGIBLE[k] = out
    return out


_CUM: Dict[int, List[float]] = {}


def _weighted(rng, options):
    cum = _CUM.get(id(options))
    if cum is None:
        w = np.cumsum([float(o[0]) for o in options])
        cum = (w / w[-1]).tolist()
        _CUM[id(options)] = cum
    import bisect
    return options[min(bisect.bisect_right(cum, rng.random()), len(options) - 1)][1]


def _distinct(rng, pool, k):
    """``k`` distinct entries of ``pool`` (pool order randomised)."""
    n = len(pool)
    k = min(k, n)
    out, seen = [], set()
    while len(out) < k:
        j = int(rng.integers(n))
        if j not in seen:
            seen.add(j)
            out.append(pool[j])
    return out


_KNOWN_NOUNS: Optional[List[str]] = None


def _noun_from_name(name: str) -> str:
    """The product type in a name: a known product noun when the name holds
    one ("Northfold Relaxed Rain Jacket - Navy" is a rain jacket), else the
    last two words before any variant suffix."""
    global _KNOWN_NOUNS
    if _KNOWN_NOUNS is None:
        _KNOWN_NOUNS = sorted({n for f in PRODUCT_FAMILIES.values() for n in f.nouns},
                              key=len, reverse=True)
    text = str(name)
    for n in _KNOWN_NOUNS:
        if n in text:
            return n if n.isupper() or n[:2].isupper() else n.lower()
    base = re.split(r"\s[-–(]\s?|\(", text)[0].strip()
    words = base.split()
    return " ".join(words[-2:]).lower() if words else "product"


def render_product_descriptions(rng: np.random.Generator, frame: ProductFrame,
                                names: Optional[Sequence] = None) -> np.ndarray:
    """Descriptions of the products in ``frame``.

    ``names``: the row's actual product names when they did not come from the
    frame (a user's own vocabulary). The description then talks about the
    product type in that name rather than about a different product."""
    size = len(frame.family)
    names = _clean_values(names, size)
    out = np.empty(size, dtype=object)
    n_sent = rng.choice([1, 2, 3, 4], size=size, p=[0.12, 0.33, 0.38, 0.17])
    bullet = rng.random(size) < 0.1
    for i in range(size):
        F = PRODUCT_FAMILIES[frame.family[i]]
        key_noun = frame.noun[i]
        if names is not None and names[i] and names[i] != frame.name[i]:
            key_noun = _noun_from_name(names[i]).title()
        noun = _soft_lower(key_noun)
        plural = is_plural(noun)
        feats = [_fill(f, rng, {}) for f in _distinct(rng, eligible(F.features, key_noun), 3)]
        uses = _distinct(rng, eligible(F.uses, key_noun), 2) or ["everyday use"]
        uses += uses[:1]
        specs = eligible(F.specs, key_noun)
        attr = (frame.attr[i] or "classic").lower()
        material = frame.material[i] or "durable materials"
        made = "made with" if frame.family[i] in ("beauty", "health", "grocery", "pets") else "made from"
        slots = {
            "noun": noun, "Noun": _cap(noun), "attr": attr, "brand": frame.brand[i],
            "Brand": frame.brand[i], "material": material,
            "made": made, "A_attr_noun": _cap(f"{attr} {noun}" if plural else f"{_article(attr)} {attr} {noun}"),
            "A_noun": _cap(noun if plural else f"{_article(noun)} {noun}"),
            "This": "These" if plural else "This", "is": "are" if plural else "is",
            "use": uses[0], "use2": uses[1],
        }
        for k, f in enumerate(feats, 1):
            slots[f"f{k}"] = f
        if feats:
            slots["F1"] = _cap(feats[0])

        def spec():
            if specs:
                return _fill(specs[int(rng.integers(len(specs)))], rng, {})
            return _fill(_weighted(rng, _DESC_USES), rng, slots)

        if bullet[i] and len(feats) >= 2:
            out[i] = (f"{_cap(noun)} in {material}. Key features: " + "; ".join(feats)
                      + f". {spec()}")
            continue
        openers = [o for o in _DESC_OPENERS if o[2] <= len(feats)]
        w = np.array([o[0] for o in openers], dtype=float)
        tmpl, used = openers[int(rng.choice(len(openers), p=w / w.sum()))][1:]
        sentences = [_fill(tmpl, rng, slots)]
        rest = feats[used:]
        tail = []
        if rest:
            fs = dict(slots, **{f"f{k}": f for k, f in enumerate(rest, 1)}, F1=_cap(rest[0]))
            tail.append(lambda fs=fs, n=len(rest): _fill(_weighted(rng, _DESC_FEATURES[n]), rng, fs))
        tail.append(lambda: _fill(_weighted(rng, _DESC_USES), rng, slots))
        tail.append(spec)
        order = rng.permutation(len(tail)) if rng.random() < 0.3 else np.arange(len(tail))
        for k in order[: int(n_sent[i]) - 1]:
            sentences.append(tail[k]())
        out[i] = " ".join(sentences)
    return out


# ── support tickets ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Issue:
    key: str
    family: str                 # billing, shipping, returns, technical, account, product
    subjects: Sequence[str]     # short subject lines
    symptoms: Sequence[str]     # first sentence of the description
    details: Sequence[str]
    causes: Sequence[str]       # resolution: what was wrong
    fixes: Sequence[str]        # resolution: what the agent did
    severity: float = 1.0       # >1 leans high/urgent, <1 leans low


_I = Issue
ISSUES: List[Issue] = [
    # billing
    _I("double_charge", "billing",
       ["Charged twice for order #{order}", "Double charge on my card", "Duplicate payment",
        "Billed twice this month", "Two charges for one order"],
       ["I was charged twice for order #{order}.", "My card shows two identical charges of ${amount}.",
        "I see a duplicate payment on my statement for the same order."],
       ["Both charges posted on the same day.", "My bank says both are settled, not pending.",
        "I only placed one order."],
       ["The payment provider retried after a timeout and both attempts were captured.",
        "A duplicate authorisation was captured instead of voided."],
       ["Refunded the duplicate charge of ${amount}; it should appear in 3-5 business days.",
        "Voided the second payment and emailed the customer the refund reference."],
       1.3),
    _I("subscription_cancelled_billed", "billing",
       ["Still billed after cancelling", "Charged after cancellation", "Cancelled but charged again",
        "Subscription not cancelled?"],
       ["I cancelled my subscription last month but was charged again on the {day}.",
        "I was billed ${amount} even though I cancelled my plan."],
       ["I have the cancellation confirmation email.", "The account still shows as active.",
        "This is the second month this has happened."],
       ["The cancellation was scheduled for the end of the term but the renewal ran first.",
        "The cancellation request did not sync to the billing system."],
       ["Cancelled the subscription, refunded the last charge and confirmed by email.",
        "Issued a full refund of ${amount} and closed the subscription."],
       1.2),
    _I("invoice_request", "billing",
       ["Need a VAT invoice", "Invoice for order #{order}", "Request for invoice copy",
        "Company details on invoice", "Receipt missing"],
       ["Could you send me an invoice for order #{order}?",
        "I need a VAT invoice with my company name for my last payment.",
        "I can't find the receipt for my payment of ${amount}."],
       ["Our finance team needs it by the end of the month.",
        "The billing email went to an old address."],
       ["The invoice email was sent to an outdated address.",
        "Company details were not on the account when the invoice was generated."],
       ["Regenerated the invoice with the company details and sent it to the customer.",
        "Updated the billing email and resent all invoices from this year."],
       0.6),
    _I("payment_declined", "billing",
       ["Payment declined at checkout", "Card keeps getting declined", "Can't complete payment",
        "Checkout payment error"],
       ["My payment keeps failing at checkout, I've tried {n} different cards.",
        "Checkout says my card was declined but my bank says there's no block."],
       ["The error just says 'payment could not be processed'.",
        "It worked fine last week."],
       ["The card issuer rejected the transaction under 3-D Secure.",
        "The billing postcode did not match the card."],
       ["Asked the customer to complete 3-D Secure in their banking app; payment went through.",
        "Corrected the billing postcode on the account and the payment succeeded."],
       1.1),
    _I("price_dispute", "billing",
       ["Wrong price charged", "Discount code not applied", "Promo code didn't work",
        "Charged full price"],
       ["I was charged full price even though I used the code {code}.",
        "The price at checkout was higher than the price on the product page."],
       ["The code was still valid according to your email.", "The difference is ${amount}."],
       ["The promo code excluded sale items.", "The product page showed a cached price."],
       ["Refunded the ${amount} difference as a goodwill gesture.",
        "Applied the discount retroactively and refunded the difference."],
       0.8),
    # shipping
    _I("not_received", "shipping",
       ["Order #{order} not received", "Package marked delivered but not here",
        "Where is my order?", "Missing parcel", "Delivered to the wrong address?"],
       ["My order #{order} shows as delivered but I haven't received anything.",
        "Tracking says my parcel was delivered on the {day} but it isn't here.",
        "I ordered {n} weeks ago and the parcel still hasn't arrived."],
       ["I've checked with neighbours and the building's mailroom.",
        "There's no photo of the delivery in the tracking.",
        "I was home all day."],
       ["The courier left the parcel at a neighbouring address.",
        "The parcel was lost in transit at the regional depot."],
       ["Shipped a replacement with express delivery at no charge.",
        "Opened a claim with the courier and refunded the order in full."],
       1.2),
    _I("delayed", "shipping",
       ["Delivery delayed", "Order still processing", "Shipping taking too long",
        "When will order #{order} ship?", "Tracking not updating"],
       ["My order #{order} has been 'processing' for {n} days.",
        "Tracking hasn't updated since the {day}.",
        "The estimated delivery date has passed."],
       ["I need it before the weekend.", "I paid for express shipping."],
       ["The item was backordered at the warehouse.",
        "The parcel was held at customs pending paperwork."],
       ["Upgraded shipping to express and refunded the shipping fee.",
        "Confirmed a new dispatch date with the warehouse and updated the customer."],
       1.0),
    _I("damaged", "shipping",
       ["Item arrived damaged", "Broken on arrival", "Damaged packaging, item cracked",
        "Order #{order} arrived broken"],
       ["My order arrived today and the {item} is cracked.",
        "The box was crushed and the {item} inside is damaged."],
       ["I've attached photos of the box and the item.", "The outer packaging was torn."],
       ["Insufficient packaging for a fragile item.",
        "The parcel was damaged in transit by the courier."],
       ["Sent a replacement and asked the customer to keep the damaged item.",
        "Refunded the item and flagged the SKU for better packaging."],
       1.0),
    _I("change_address", "shipping",
       ["Change delivery address", "Wrong shipping address", "Update address on order #{order}"],
       ["I entered the wrong delivery address for order #{order}.",
        "Can I change the shipping address on my order? I've moved."],
       ["The order hasn't shipped yet.", "The new address is in the same city."],
       ["The customer selected an old saved address at checkout."],
       ["Updated the address before dispatch and confirmed with the customer.",
        "Redirected the parcel through the courier's portal."],
       0.7),
    # returns
    _I("return_request", "returns",
       ["Return request for order #{order}", "How do I return an item?", "Return label please",
        "Want to return {item}", "Exchange for a different size"],
       ["I'd like to return the {item} from order #{order}.",
        "The {item} doesn't fit, can I exchange it for a different size?",
        "How do I start a return? I can't find the option in my account."],
       ["It's unworn with the tags still on.", "I ordered it {n} days ago."],
       ["The return window was still open.", "The return option was hidden for marketplace items."],
       ["Emailed a prepaid return label; refund will be issued on receipt.",
        "Arranged an exchange and dispatched the new size."],
       0.6),
    _I("refund_not_received", "returns",
       ["Refund not received", "Where is my refund?", "Returned item, no refund yet",
        "Refund for order #{order}"],
       ["I returned my order {n} weeks ago and still haven't had a refund.",
        "The courier confirmed my return was delivered but no refund has been issued."],
       ["The tracking shows it arrived at your warehouse on the {day}.",
        "The refund amount should be ${amount}."],
       ["The return was received but not scanned into the returns system.",
        "The refund was issued to an expired card."],
       ["Processed the refund of ${amount} manually and sent confirmation.",
        "Reissued the refund as store credit at the customer's request."],
       1.0),
    _I("wrong_item", "returns",
       ["Wrong item received", "Received the wrong size", "Not what I ordered",
        "Wrong colour sent"],
       ["I ordered the {item} but received something completely different.",
        "I received the wrong size in order #{order}."],
       ["The packing slip shows the right item.", "I've attached a photo of what arrived."],
       ["A picking error at the warehouse.", "Two orders were swapped at packing."],
       ["Sent the correct item by express and a return label for the wrong one.",
        "Refunded the order and arranged a courier collection."],
       0.9),
    # technical
    _I("app_crash", "technical",
       ["App crashes on startup", "App keeps crashing", "Crash when opening settings",
        "iOS app closes immediately", "Android app freezing"],
       ["The app crashes every time I open the settings page.",
        "Since the latest update the app closes as soon as I open it.",
        "The app freezes on the loading screen."],
       ["I'm on version {version}.", "I've reinstalled it twice.",
        "It works fine on my tablet but not my phone."],
       ["A null pointer in the settings screen for accounts without a profile photo.",
        "A bug in release {version} on older OS versions."],
       ["Fix shipped in version {version}; confirmed with the customer after updating.",
        "Shared a workaround (clear app data) and linked the bug to the next release."],
       1.2),
    _I("login_error", "technical",
       ["Error 500 on login", "Login page not loading", "Stuck on loading screen",
        "Can't log in on mobile"],
       ["I get an error 500 every time I try to log in.",
        "The login page just spins and never loads."],
       ["It happens on Chrome and Safari.", "Started this morning."],
       ["A failed deploy on the authentication service.",
        "An expired TLS certificate on the login subdomain."],
       ["Rolled back the deploy; logins are working again.",
        "Renewed the certificate and confirmed login with the customer."],
       1.6),
    _I("export_broken", "technical",
       ["CSV export is empty", "Export not working", "Report download fails",
        "Data export stuck at 0%"],
       ["The export feature produces an empty CSV file.",
        "When I download the monthly report the file is blank."],
       ["It worked last month with the same filters.", "The table on screen has data."],
       ["The export timed out for date ranges over a year.",
        "A filter on the archived field excluded every row."],
       ["Increased the export timeout; export works for the full date range.",
        "Fixed the filter and re-ran the export for the customer."],
       0.9),
    _I("api_key", "technical",
       ["API key not working", "401 errors from the API", "Regenerated key rejected",
        "API authentication failing"],
       ["My API key returns 401 since I regenerated it.",
        "Every API call fails with 'invalid credentials' since this morning."],
       ["The key is copied exactly from the dashboard.", "The old key also stopped working."],
       ["The new key had not been granted the write scope.",
        "The key was created in the sandbox environment, not production."],
       ["Added the missing scope to the key; calls succeed now.",
        "Pointed the customer to the production key and confirmed a successful call."],
       1.2),
    _I("integration", "technical",
       ["Integration stopped syncing", "Webhook not firing", "Slack notifications stopped",
        "Sync error with calendar", "Zapier integration broken"],
       ["The integration stopped sending notifications on the {day}.",
        "Webhooks are no longer reaching our endpoint.",
        "Our calendar sync has been failing since the {day}."],
       ["Nothing changed on our side.", "The status page shows everything green."],
       ["The OAuth token expired and the refresh failed silently.",
        "The webhook endpoint was disabled after repeated timeouts."],
       ["Reconnected the integration and replayed the missed events.",
        "Re-enabled the webhook and advised the customer to respond within 10 seconds."],
       1.1),
    _I("slow", "technical",
       ["Dashboard very slow", "Pages taking ages to load", "Performance issues",
        "Site is really slow today"],
       ["The dashboard takes over {n} seconds to load.",
        "Everything has been very slow since this morning."],
       ["It's affecting the whole team.", "Other websites are fine."],
       ["A slow database query on accounts with large histories.",
        "A regional CDN outage."],
       ["Added an index for the slow query; load time is back under two seconds.",
        "CDN provider resolved the outage; confirmed performance is normal."],
       1.3),
    # account
    _I("password_reset", "account",
       ["Password reset email not arriving", "Can't reset password", "Reset link expired",
        "Locked out of my account"],
       ["I've requested a password reset {n} times but the email never arrives.",
        "The password reset link says it has expired as soon as I click it.",
        "I'm locked out after too many login attempts."],
       ["I've checked my spam folder.", "I'm using the email address on my account."],
       ["Reset emails to this domain were being blocked by the recipient's mail server.",
        "The account was locked after repeated failed attempts."],
       ["Unlocked the account and sent a reset link to a verified backup address.",
        "Reset the password manually and enabled two-factor authentication."],
       1.0),
    _I("two_factor_sms", "account",
       ["2FA code not arriving", "Verification code never comes", "No SMS code",
        "Two-factor problem"],
       ["Two-factor authentication isn't sending the verification code.",
        "I never receive the SMS code when I try to log in."],
       ["My phone number hasn't changed.", "I've waited over ten minutes for it."],
       ["SMS delivery to the customer's carrier was delayed.",
        "The phone number on file was missing the country code."],
       ["Switched the customer to app-based codes after SMS delays.",
        "Corrected the phone number format; codes arrive now."],
       1.1),
    _I("two_factor_device", "account",
       ["Lost access to authenticator", "New phone, can't log in", "Reset 2FA please"],
       ["I got a new phone and lost my authenticator app.",
        "My old phone broke and I can't get past the two-factor step."],
       ["I still have access to my email.", "I don't have the backup codes."],
       ["The authenticator was tied to the old device."],
       ["Verified identity and reset two-factor; customer re-enrolled successfully.",
        "Reset two-factor after ID check and sent new backup codes."],
       1.1),
    _I("change_email", "account",
       ["Change email address", "Update my email", "New email for my account",
        "Can't update email"],
       ["How do I change the email address on my account?",
        "I'm trying to update my email but the field is greyed out."],
       ["The old email is no longer active.", "I signed up with Google originally."],
       ["The email field is read-only for accounts created via social login.",
        "The new address was already linked to a second account."],
       ["Updated the email after verifying the customer's identity.",
        "Released the address from the unused account and updated the email."],
       0.5),
    _I("merge_accounts", "account",
       ["Merge two accounts", "Duplicate accounts", "Two accounts, same person"],
       ["I have two accounts and would like to merge them.",
        "I accidentally created a second account and my orders are split between them."],
       ["Both accounts are in my name.", "One uses my work email."],
       ["A second account was created at guest checkout."],
       ["Merged the accounts and kept the full order history.",
        "Moved the orders to the main account and closed the duplicate."],
       0.5),
    _I("delete_account", "account",
       ["Delete my account", "Close account request", "Remove my data", "GDPR deletion request"],
       ["Please delete my account and all my personal data.",
        "I'd like to close my account permanently."],
       ["I no longer use the service.", "Please confirm by email once it's done."],
       ["Customer requested erasure under data-protection rules."],
       ["Deleted the account and confirmed under the data-protection request.",
        "Closed the account and scheduled data deletion within 30 days."],
       0.6),
    _I("unsubscribe", "account",
       ["Unsubscribe from emails", "Too many marketing emails", "Still getting newsletters",
        "Stop sending me emails"],
       ["I keep getting marketing emails even though I unsubscribed.",
        "I unsubscribed {n} times and the newsletters keep coming."],
       ["I only want order updates.", "The unsubscribe link says I'm already removed."],
       ["Marketing preferences were stored per list, not per account."],
       ["Removed the customer from all marketing lists; order emails still on.",
        "Turned off every marketing list and confirmed the preference change."],
       0.4),
    # product
    _I("dark_mode", "product",
       ["Feature request: dark mode", "Dark mode please", "Any plans for dark mode?"],
       ["It would be great to have a dark mode in the app.",
        "Is a dark theme on the roadmap? The white screen is harsh at night."],
       ["Our team uses this every day.", "A competitor already offers this."],
       ["Not a defect; logged as a feature request."],
       ["Logged the request with the product team and shared the public roadmap.",
        "Added the customer's vote to the existing dark-mode request."],
       0.4),
    _I("bulk_edit", "product",
       ["Feature request: bulk edit", "Edit several items at once?", "Bulk update option"],
       ["Could you add a way to edit several items at once?",
        "Updating {n}00 records one by one takes hours. Is there a bulk edit?"],
       ["We have to do this every month."],
       ["Not a defect; bulk edit exists for admins only."],
       ["Explained the admin bulk-edit tool and raised the request for other roles.",
        "Shared the CSV import as a workaround and logged the request."],
       0.4),
    _I("pdf_export", "product",
       ["Can you add export to PDF?", "PDF export for reports", "Request: printable reports"],
       ["We'd love an export to PDF option for reports.",
        "Is there a way to save reports as PDF for our board pack?"],
       ["Right now we screenshot the dashboard."],
       ["Not a defect; logged as a feature request."],
       ["Shared the print-to-PDF workaround and logged the request.",
        "Added the customer to the beta for scheduled PDF reports."],
       0.4),
    _I("product_question", "product",
       ["Question about {item}", "Is this compatible?", "Sizing question", "Product information"],
       ["Is the {item} compatible with my current setup?",
        "What are the exact dimensions of the {item}?",
        "Does the {item} come with a warranty?"],
       ["I couldn't find it on the product page."],
       ["Information missing from the product page."],
       ["Answered with the specifications and updated the product page.",
        "Confirmed compatibility and sent the setup guide."],
       0.4),
]

_ISSUE_FAMILY_KEYWORDS = [
    ("returns", ("return", "refund", "exchange")),
    ("billing", ("bill", "payment", "invoice", "charge", "pay", "finance", "subscription")),
    ("shipping", ("ship", "deliver", "logistic", "order", "fulfil", "fulfill", "courier")),
    ("account", ("account", "login", "access", "auth", "security", "password", "privacy")),
    ("technical", ("tech", "bug", "software", "app", "it ", "outage", "integration", "api",
                   "incident", "error", "performance", "support")),
    ("product", ("product", "feature", "request", "feedback", "question", "inquiry",
                 "enquiry", "general", "sales", "other")),
]

_PRIORITY_RANK = {"lowest": 0, "trivial": 0, "low": 0, "minor": 0, "p4": 0, "p3": 1,
                  "normal": 1, "medium": 1, "moderate": 1, "major": 2, "high": 2, "p2": 2,
                  "urgent": 3, "critical": 3, "blocker": 3, "highest": 3, "p1": 3,
                  "p0": 3, "emergency": 3}

_OPEN_STATUSES = {"open", "new", "pending", "in progress", "in_progress", "waiting",
                  "awaiting", "on hold", "on_hold", "assigned", "escalated", "triage",
                  "reopened", "awaiting customer", "awaiting_customer", "todo", "backlog",
                  "active", "investigating", "acknowledged"}


def issue_family_for(category) -> Optional[str]:
    c = f" {str(category).strip().lower()} "
    if c.strip() in ("", "nan", "none"):
        return None
    for fam, words in _ISSUE_FAMILY_KEYWORDS:
        if any(w in c for w in words):
            return fam
    return None


def is_open_status(status) -> bool:
    s = str(status).strip().lower()
    return s in _OPEN_STATUSES or s.startswith(("open", "pending", "waiting", "in prog"))


@dataclass
class TicketFrame:
    issue: List[Issue]
    slots: List[Dict[str, str]]
    urgency: List[int]          # 0 low .. 3 urgent
    open_: List[bool]
    subject: List[str]


_FAMILY_ISSUES: Dict[str, List[Issue]] = {}
for _iss in ISSUES:
    _FAMILY_ISSUES.setdefault(_iss.family, []).append(_iss)


_DEVICES = ["iPhone 15", "iPhone 13", "Pixel 8", "Galaxy S23", "Galaxy A54", "iPad",
            "MacBook Air", "Windows laptop", "Chromebook", "work PC", "Android tablet"]
_BROWSERS = ["Chrome", "Safari", "Firefox", "Edge", "the app", "Brave"]
_WHENS = ["yesterday", "this morning", "last night", "Monday", "Tuesday", "the weekend",
          "Friday", "last week", "two days ago", "the last update", "about an hour ago"]
_PLANS = ["Basic", "Plus", "Pro", "Premium", "Business", "Family", "Starter", "Team"]


def _ticket_slots(rng, item: Optional[str]) -> Dict[str, str]:
    return _ticket_slot_rows(rng, 1, [item])[0]


def _ticket_slot_rows(rng, size: int, items: Optional[Sequence] = None) -> List[Dict[str, str]]:
    """Per-row slot values, drawn as whole columns."""
    def pick(pool):
        idx = rng.integers(0, len(pool), size)
        return [pool[i] for i in idx]
    devices, browsers, whens, plans = pick(_DEVICES), pick(_BROWSERS), pick(_WHENS), pick(_PLANS)
    codes = pick(["SPRING20", "WELCOME10", "SAVE15", "FREESHIP", "VIP25"])
    mins = rng.integers(5, 90, size)
    orders = rng.integers(100000, 999999, size)
    amounts = np.exp(rng.normal(3.6, 0.8, size))
    ns = rng.integers(2, 6, size)
    days = rng.integers(1, 29, size)
    va, vb, vc = rng.integers(3, 9, size), rng.integers(0, 15, size), rng.integers(0, 6, size)
    out = []
    for i in range(size):
        it = items[i] if items is not None and i < len(items) and items[i] else None
        out.append({
            "device": devices[i], "browser": browsers[i], "when": whens[i], "plan": plans[i],
            "mins": str(int(mins[i])), "order": str(int(orders[i])),
            "amount": f"{float(amounts[i]):.2f}", "n": str(int(ns[i])), "day": _day(int(days[i])),
            "code": codes[i], "version": f"{va[i]}.{vb[i]}.{vc[i]}", "item": it or "item",
        })
    return out


def _day(d: int) -> str:
    return f"{d}{'th' if 11 <= d <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"


def draw_tickets(rng: np.random.Generator, size: int, *,
                 categories: Optional[Sequence] = None,
                 priorities: Optional[Sequence] = None,
                 statuses: Optional[Sequence] = None,
                 items: Optional[Sequence] = None) -> TicketFrame:
    """One latent support issue per row, conditioned on what the row says.

    The declared category picks the issue family; the declared priority
    leans the draw toward severe issues (an urgent ticket is a login outage
    more often than a feature request); the status decides whether the
    ticket has a resolution yet. None of those columns are changed."""
    cats = _clean_values(categories, size)
    pris = _clean_values(priorities, size)
    stats = _clean_values(statuses, size)
    items = _clean_values(items, size)
    import bisect
    issues: List[Issue] = []
    urg: List[int] = []
    fam_cache: Dict[str, Optional[str]] = {}
    cum_cache: Dict[tuple, List[float]] = {}
    draws = rng.random(size)
    noise = rng.normal(0, 0.6, size)
    for i in range(size):
        c = cats[i] if cats else ""
        if c not in fam_cache:
            fam_cache[c] = issue_family_for(c) if c else None
        fam = fam_cache[c]
        pool = (_FAMILY_ISSUES.get(fam) if fam else None) or ISSUES
        u = _PRIORITY_RANK.get(pris[i].lower(), None) if pris else None
        key = (fam, u)
        cum = cum_cache.get(key)
        if cum is None:
            # severity^(2(u-1.5)): urgent rows favour severe issues, low rows mild ones
            w = np.array([1.0 if u is None else iss.severity ** (2.0 * (u - 1.5)) for iss in pool])
            cum = np.cumsum(w / w.sum()).tolist()
            cum_cache[key] = cum
        issues.append(pool[min(bisect.bisect_right(cum, draws[i]), len(pool) - 1)])
        if u is None:
            sev = issues[-1].severity
            u = int(np.clip(round(1 + (sev - 1) * 2.5 + noise[i]), 0, 3))
        urg.append(int(u))
    open_ = [is_open_status(s) for s in stats] if stats else [False] * size
    slots = _ticket_slot_rows(rng, size, [(_noun_from_name(x) if x else None) for x in items]
                              if items else None)
    subjects = []
    for i, iss in enumerate(issues):
        s = _fill(str(_pick(rng, iss.subjects)), rng, slots[i])
        q = rng.random()
        if q < 0.12 and "#" not in s and iss.family in ("billing", "shipping", "returns"):
            s = f"{s} (order #{slots[i]['order']})"
        elif q < 0.2 and iss.family == "technical":
            s = f"{s} since {slots[i]['when']}"
        elif q < 0.26 and iss.family in ("technical",):
            s = f"{s} on {slots[i]['device']}"
        elif q < 0.32 and items and items[i]:
            s = f"{s} - {items[i] if len(items[i]) < 40 else slots[i]['item']}"
        elif q < 0.36:
            s = f"{_pick(rng, ['Help: ', 'Question: ', 'Issue: ', 'Problem: '])}{s[0].lower() + s[1:]}"
        r = rng.random()
        if r < 0.06:
            s = f"Re: {s}"
        elif r < 0.10:
            s = s.lower()
        elif r < 0.13 and urg[i] >= 2:
            s = f"URGENT: {s}"
        elif r < 0.16:
            s = f"{s}!"
        subjects.append(s)
    return TicketFrame(issues, slots, urg, open_, subjects)


_GREETINGS = ["Hi,", "Hello,", "Hi there,", "Hello team,", "Hey,", "Good morning,", "Dear support,",
              "Hi support,", "Hello there,", "Good afternoon,", "To whom it may concern,", "Hiya,"]
_LEADS = ["", "", "", "", "", "", "So, ", "Quick one: ", "For the second time, ", "Hoping you can help. ",
          "Not sure who to ask, but ", "Following up on my chat: ", "As the title says, "]
_CONTEXT_LINES = {
    "any": ["This started {when}.", "I first noticed it {when}.", "I'm on the {plan} plan.",
            "My account email is the one on this ticket.", "I've been a customer for {n} years.",
            "I've waited {mins} minutes on chat already.", "This is my {n}th time asking."],
    "technical": ["I'm using {browser} on a {device}.", "Same thing on my {device}.",
                  "It started after the {version} update.", "Screenshot attached.",
                  "Our whole team on the {plan} plan sees it.", "Happens every time, not just once."],
    "account": ["I'm on a {device}.", "I tried from {browser} as well.",
                "I'm travelling, so I can't get to my usual laptop."],
    "billing": ["The charge was ${amount}.", "My order number is #{order}.",
                "Statement screenshot attached.", "I'm on the {plan} plan."],
    "shipping": ["My order number is #{order}.", "I paid ${amount} for it.",
                 "The tracking link just says 'in transit'.", "It was a gift, so timing matters."],
    "returns": ["Order #{order}.", "It cost ${amount}.", "I still have the original packaging.",
                "I sent it back {when}."],
    "product": ["We're on the {plan} plan.", "I'd happily pay more for this.",
                "Using it on a {device} mostly."],
}
_TRIED = ["I've tried logging out and back in.", "I've already cleared my cache and cookies.",
          "I tried a different browser with the same result.", "Restarting didn't help.",
          "I checked the help centre but couldn't find an answer.",
          "I contacted you on chat but the conversation dropped."]
_IMPACT = {
    0: ["No rush, just wanted to flag it.", "No hurry on this one.", "Whenever you get a chance.", ""],
    1: ["Could you look into this?", "Please let me know what to do.", "Thanks for your help.", ""],
    2: ["This is affecting my work, please help.", "I need this sorted this week.",
        "Please look into this as soon as possible."],
    3: ["This is urgent, we can't operate.", "Please escalate, this is blocking our whole team.",
        "We need this fixed today.", "I need this resolved today."],
}
_SIGNOFFS = ["Thanks", "Thanks,", "Thank you", "Cheers", "Regards", "Best", "Many thanks"]
_PREFACES = ["Spoke to the customer by phone. ", "Per chat with the customer: ",
             "Checked the logs. ", "Escalated to tier 2. ", "Reproduced the issue. ",
             "Verified the account. ", "Investigated with engineering. ",
             "Customer called back. ", "Reviewed the order history. ", "Confirmed with billing. ",
             "Looked into this with the warehouse. ", "Followed up by email. ",
             "Checked with the courier. ", "Pulled the payment logs. "]
_RESOLUTION_NOTES_EXTRA = [
    "Customer is on the {plan} plan.", "Handled in {mins} minutes.", "Order #{order}.",
    "Amount involved: ${amount}.", "Reported on {device}.", "First reported {when}.",
    "Second contact about this.", "Customer was polite and patient.",
    "Customer was frustrated; offered a {n}0% voucher.", "No further action needed.",
    "Checked for similar tickets: {n} others this week.", "Added an internal note to the account.",
]
_FOLLOWUPS = ["", "", "Customer confirmed the issue is resolved.",
              "Customer thanked us; marking solved.", "Will monitor for a week.",
              "Macro sent.", "Linked to the known-issue article.", "Tagged for the weekly review.",
              "Closing; customer can reply to reopen.", "Sent a follow-up survey.",
              "No reply from customer after 48 hours; closing.",
              "Added a note to the account for future reference.",
              "Shared the help-centre article for next time."]


def render_ticket_descriptions(rng: np.random.Generator, frame: TicketFrame) -> np.ndarray:
    size = len(frame.issue)
    out = np.empty(size, dtype=object)
    for i in range(size):
        iss, sl, u = frame.issue[i], frame.slots[i], frame.urgency[i]
        parts = []
        greeting = str(_pick(rng, _GREETINGS)) if rng.random() < 0.4 else ""
        sym = _fill(str(_pick(rng, iss.symptoms)), rng, sl)
        lead = str(_pick(rng, _LEADS))
        if lead and not lead.endswith(". ") and not lead.endswith(": "):
            sym = sym[0].lower() + sym[1:] if not sym.startswith("I ") and not sym.startswith("I'") else sym
        parts.append(lead + sym)
        if rng.random() < 0.7:
            parts.append(_fill(str(_pick(rng, iss.details)), rng, sl))
        if rng.random() < 0.6:
            pool = _CONTEXT_LINES.get(iss.family, []) + _CONTEXT_LINES["any"]
            parts.append(_fill(str(_pick(rng, pool)), rng, sl))
        if iss.family in ("technical", "account") and rng.random() < 0.5:
            parts.append(str(_pick(rng, _TRIED)))
        impact = str(_pick(rng, _IMPACT[u]))
        if impact:
            parts.append(impact)
        text = " ".join(parts)
        if greeting:
            text = f"{greeting}\n\n{text}"
        if rng.random() < 0.25:
            text += f"\n\n{_pick(rng, _SIGNOFFS)}"
        out[i] = text
    return out


def render_ticket_resolutions(rng: np.random.Generator, frame: TicketFrame) -> np.ndarray:
    """What was wrong and what was done; None while the ticket is still open."""
    size = len(frame.issue)
    out = np.empty(size, dtype=object)
    for i in range(size):
        if frame.open_[i]:
            out[i] = None
            continue
        iss, sl = frame.issue[i], frame.slots[i]
        # Causes and fixes are written in pairs: the fix answers the cause.
        k = int(rng.integers(len(iss.fixes)))
        fix = _fill(iss.fixes[k], rng, sl)
        cause = _fill(iss.causes[k % len(iss.causes)], rng, sl)
        r = rng.random()
        if r < 0.45:
            text = f"{cause} {fix}"
        elif r < 0.6:
            text = f"Root cause: {cause[0].lower() + cause[1:]}".rstrip(".") + f". {fix}"
        else:
            text = fix
        if rng.random() < 0.5:
            text = f"{text} {_fill(str(_pick(rng, _RESOLUTION_NOTES_EXTRA)), rng, sl)}"
        follow = str(_pick(rng, _FOLLOWUPS))
        pre = str(_pick(rng, _PREFACES)) if rng.random() < 0.3 else ""
        if pre and r >= 0.45:
            text = text[0].lower() + text[1:] if pre.endswith(": ") else text
        out[i] = f"{pre}{text} {follow}".strip()
    return out


# ── short labels and profiles ────────────────────────────────────────────────

_JOB_FUNCTIONS = [
    "Software Engineer", "Product Manager", "Data Analyst", "Account Executive",
    "Marketing Manager", "Operations Manager", "Customer Success Manager", "Accountant",
    "UX Designer", "Project Manager", "Sales Representative", "Business Analyst",
    "HR Business Partner", "Registered Nurse", "Teacher", "Financial Analyst",
    "Data Scientist", "DevOps Engineer", "Graphic Designer", "Content Strategist",
    "Recruiter", "Office Manager", "Mechanical Engineer", "Civil Engineer",
    "Pharmacist", "Paralegal", "Solicitor", "Consultant", "Store Manager",
    "Logistics Coordinator", "Supply Chain Analyst", "Electrician", "Architect",
    "Copywriter", "Customer Service Representative", "Executive Assistant",
    "Research Scientist", "Security Analyst", "QA Engineer", "Social Media Manager",
    "Procurement Specialist", "Payroll Specialist", "Physiotherapist", "Chef",
    "Event Coordinator", "Real Estate Agent", "Insurance Underwriter", "Lab Technician",
    "Frontend Developer", "Backend Developer", "Solutions Architect", "Brand Manager",
    "Technical Writer", "Site Reliability Engineer", "Bookkeeper", "Teaching Assistant",
]
_SENIORITY = [("", 0.58), ("Senior ", 0.2), ("Junior ", 0.06), ("Lead ", 0.06),
              ("Principal ", 0.03), ("Associate ", 0.05), ("Head of ", 0.02)]
_HEAD_OF_OK = {"Marketing", "Operations", "Product", "Sales", "Design", "Engineering"}


def job_titles(rng: np.random.Generator, size: int, key: str = "job_title") -> np.ndarray:
    """Job titles with a realistic skew and seniority mix (several hundred
    distinct titles, a few common ones), not 22 titles in equal shares."""
    base = zipf_choice(rng, _JOB_FUNCTIONS, size, s=0.9, q=3, key=key)
    prefixes, w = zip(*_SENIORITY)
    pre = rng.choice(len(prefixes), size=size, p=np.array(w) / sum(w))
    out = np.empty(size, dtype=object)
    for i in range(size):
        p = prefixes[pre[i]]
        b = base[i]
        if p == "Head of ":
            fn = b.split()[0]
            out[i] = f"Head of {fn}" if fn in _HEAD_OF_OK else b
        elif p and b.split()[0] in ("Registered", "Executive", "Teaching"):
            out[i] = b
        else:
            out[i] = f"{p}{b}"
    return out


_INTERESTS = ["trail running", "sourdough", "film photography", "board games", "cycling",
              "jazz", "climbing", "gardening", "chess", "travel", "podcasts", "yoga",
              "cooking", "reading sci-fi", "football", "open source", "wild swimming",
              "pottery", "birdwatching", "live music", "baking", "hiking", "coffee",
              "vintage cars", "running", "tennis", "DIY", "knitting", "surfing", "writing"]
_FIELDS = ["product", "design", "data", "marketing", "operations", "finance", "healthcare",
           "education", "retail", "engineering", "logistics", "sales", "consulting",
           "customer success", "hospitality", "research"]


def bios(rng: np.random.Generator, size: int, *, jobs: Optional[Sequence] = None,
         cities: Optional[Sequence] = None, companies: Optional[Sequence] = None) -> np.ndarray:
    """Short profile bios in several structures, using the row's own job,
    city and employer when the table has them."""
    jobs = _clean_values(jobs, size)
    cities = _clean_values(cities, size)
    companies = _clean_values(companies, size)
    out = np.empty(size, dtype=object)
    for i in range(size):
        job = (jobs[i] if jobs and jobs[i] else None)
        city = (cities[i] if cities and cities[i] else None)
        comp = (companies[i] if companies and companies[i] else None)
        a, b = _distinct(rng, _INTERESTS, 2)
        field = str(_pick(rng, _FIELDS))
        opts = [(2, f"{a.capitalize()} and {b}.")]
        if job:
            opts += [(3, f"{job}{f' at {comp}' if comp else ''}. Into {a} and {b}."),
                     (2, f"{job} | {a} | {b}"),
                     (2, f"{job}{f' based in {city}' if city else ''}."),
                     (1, f"{job} by day, {a} by night.")]
        if city:
            opts += [(2, f"{city}-based. {a.capitalize()}, {b} and good coffee."),
                     (1, f"Living in {city}. Big on {a}.")]
        opts += [(2, f"{int(rng.integers(2, 25))} years in {field}. Weekends are for {a}."),
                 (1, f"I work in {field} and spend my free time on {a}."),
                 (1, f"Curious about {field}, {a} and {b}."),
                 (1, f"{field.capitalize()} person. {a.capitalize()} fan.")]
        text = _weighted(rng, opts)
        if rng.random() < 0.08:
            text += " " + str(_pick(rng, ["🌍", "☕", "🚴", "📚", "🎧", "🌱", "📷"]))
        out[i] = text
    return out


# Street address formats by country: the street line only; city and postcode
# live in their own columns when the table has them.
_STREETS_EN = ["Main", "High", "Church", "Station", "Park", "Mill", "Victoria", "King",
               "Queen", "Oak", "Maple", "Cedar", "Elm", "Pine", "Lake", "Hill", "River",
               "Spring", "Market", "Bridge", "North", "South", "Washington", "Lincoln",
               "Franklin", "Highland", "Sunset", "Grove", "Meadow", "Chestnut"]
_SUFFIX_US = ["St", "Ave", "Rd", "Blvd", "Dr", "Ln", "Way", "Ct", "Pl"]
_SUFFIX_UK = ["Street", "Road", "Lane", "Avenue", "Close", "Gardens", "Way", "Crescent", "Terrace"]
_STREETS_DE = ["Haupt", "Bahnhof", "Schul", "Garten", "Berg", "Wald", "Linden", "Kirch",
               "Dorf", "Ring", "Wiesen", "Birken"]
_SUFFIX_DE = ["straße", "weg", "allee", "gasse", "platz"]
_STREETS_FR = ["de la République", "Victor Hugo", "de la Paix", "Jean Jaurès", "du Moulin",
               "des Écoles", "de la Gare", "Pasteur", "du Château", "des Lilas"]
_TYPE_FR = ["rue", "avenue", "boulevard", "place", "impasse", "chemin"]
_STREETS_NL = ["Kerk", "Dorps", "Molen", "School", "Linden", "Prins Hendrik", "Wilhelmina",
               "Beatrix", "Juliana", "Stations"]
_SUFFIX_NL = ["straat", "weg", "laan", "plein", "gracht"]
_STREETS_BR = ["das Flores", "Sete de Setembro", "São João", "XV de Novembro", "da Paz",
               "Santos Dumont", "Tiradentes", "Brasil", "Getúlio Vargas"]
_TYPE_BR = ["Rua", "Avenida", "Travessa", "Alameda"]
_STREETS_IN = ["MG Road", "Station Road", "Gandhi Nagar", "Nehru Street", "Park Street",
               "Civil Lines", "Laxmi Nagar", "Sector 14", "Brigade Road", "Anna Salai"]
_JP_WARDS = ["Chuo", "Minato", "Shibuya", "Shinjuku", "Kita", "Naka", "Higashi", "Nishi"]


def street_lines(rng: np.random.Generator, size: int,
                 countries: Optional[Sequence] = None) -> np.ndarray:
    """Street lines in each row's country's own format."""
    cs = _clean_values(countries, size) or ["United States"] * size
    out = np.empty(size, dtype=object)
    for i in range(size):
        c = cs[i].lower()
        num = 1 + int(rng.exponential(1800 if c in ("united states", "usa", "us") else 60))
        if c in ("united kingdom", "uk", "great britain", "england", "australia", "ireland",
                 "new zealand"):
            line = f"{num} {_pick(rng, _STREETS_EN)} {_pick(rng, _SUFFIX_UK)}"
            if rng.random() < 0.15:
                line = f"Flat {int(rng.integers(1, 30))}, {line}"
        elif c in ("germany", "deutschland", "austria", "switzerland"):
            line = f"{_pick(rng, _STREETS_DE)}{_pick(rng, _SUFFIX_DE)} {num}"
        elif c in ("france", "belgium"):
            line = f"{num} {_pick(rng, _TYPE_FR)} {_pick(rng, _STREETS_FR)}"
        elif c in ("netherlands", "holland"):
            line = f"{_pick(rng, _STREETS_NL)}{_pick(rng, _SUFFIX_NL)} {num}"
        elif c in ("brazil", "brasil", "portugal"):
            line = f"{_pick(rng, _TYPE_BR)} {_pick(rng, _STREETS_BR)}, {num}"
        elif c == "india":
            line = f"{num}, {_pick(rng, _STREETS_IN)}"
        elif c == "japan":
            line = (f"{int(rng.integers(1, 9))}-{int(rng.integers(1, 30))}-"
                    f"{int(rng.integers(1, 20))} {_pick(rng, _JP_WARDS)}")
        else:
            line = f"{num} {_pick(rng, _STREETS_EN)} {_pick(rng, _SUFFIX_US)}"
            if rng.random() < 0.25:
                line += str(rng.choice([f", Apt {int(rng.integers(1, 40))}{rng.choice(list('ABCDEF'))}",
                                        f", Unit {int(rng.integers(1, 300))}",
                                        f", Suite {int(rng.integers(1, 9)) * 100}"]))
        out[i] = line
    return out


def brands(rng: np.random.Generator, size: int, families: Optional[Sequence] = None,
           key: str = "brand") -> np.ndarray:
    """Brand names, concentrated the way real catalogues are."""
    if families is None:
        pool = sorted({b for f in PRODUCT_FAMILIES.values() for b in f.brands})
        return zipf_choice(rng, pool, size, s=1.05, key=key)
    fam = np.array([f or "generic" for f in families], dtype=object)
    out = np.empty(size, dtype=object)
    for f in sorted(set(fam)):
        idx = np.flatnonzero(fam == f)
        out[idx] = zipf_choice(rng, PRODUCT_FAMILIES[f].brands, len(idx), s=1.1,
                               key=f"{key}|{f}")
    return out


def subject_from_name(name: str) -> str:
    """How a review refers to a product: the type, not the full listing title
    ("Northfold Relaxed Rain Jacket - Navy" is "rain jacket")."""
    return _noun_from_name(name)
