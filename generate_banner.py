import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageDraw, ImageChops
import scipy.ndimage as ndimage
from scipy.optimize import linear_sum_assignment

# --- Configuration & Palette ---
WIDTH, HEIGHT = 1180, 610
PORTRAIT_COLS, PORTRAIT_ROWS = 300, 340
N_TRAVELLERS = 900
N_BANDS = 94

# Dark Mode Colors
DARK_BG = "#0A101F"
DARK_CHROME = "#22D3EE"
DARK_DOTS = "#A78BFA"
DARK_ACCENT = "#10B981"
DARK_TEXT = "#F1F5F9"
DARK_LEADER = "#475569"

# Light Mode Colors
LIGHT_BG = "#F8FAFC"
LIGHT_CHROME = "#0891B2"
LIGHT_DOTS = "#7C3AED"
LIGHT_ACCENT = "#10B981"
LIGHT_TEXT = "#0F172A"
LIGHT_LEADER = "#CBD5E1"

# Target Coordinates for Portrait inside Left Pane
# Pane: x inside [20, 430] (width 410), y inside [55, 590] (height 535)
CX_LEFT, CY_LEFT = 225, 322.5
PORTRAIT_DX, PORTRAIT_DY = 1.0, 1.0
PORTRAIT_W = PORTRAIT_COLS * PORTRAIT_DX
PORTRAIT_H = PORTRAIT_ROWS * PORTRAIT_DY
X_START_PORTRAIT = int(CX_LEFT - PORTRAIT_W / 2)
Y_START_PORTRAIT = int(CY_LEFT - PORTRAIT_H / 2)

# Traced Logos Centered at (CX_LEFT, CY_LEFT)
LOGO_CENTER = (CX_LEFT, CY_LEFT)

# --- Info Panel Details ---
INFO_DETAILS = [
    # Label, Value, Section
    ("Subject", "Bhuvan Raj Guguloth", 1),
    ("Role", "AI Product Researcher", 1),
    ("Origin", "Warangal, India", 1),
    ("Education", "B.Tech + M.Tech Dual Degree, IIT Kharagpur", 1),
    ("Status", "Delusional", 1),
    ("ToolChain", "VS Code, Git, Flutter, Unity, Android Studio", 1),
    ("", "", 0), # Spacer
    ("Core.Lang", "Python, C, C++, JS, Dart, Kotlin", 2),
    ("Core.Frontend", "React, HTML, CSS, JS, GSAP, Kotlin, Flutter, React Native", 2),
    ("Core.Backend", "Python, Node.js, C++", 2),
    ("Core.Database", "MongoDB, Firebase", 2),
    ("Core.Infra", "Git, Vercel, GitHub Actions", 2),
    ("", "", 0), # Spacer
    ("Grid.Mail", "bhuvanrajnaik@gmail.com", 3),
    ("Grid.Portfolio", "coming soon", 3),
    ("Grid.LinkedIn", "linkedin.com/in/bhuvanrajguguloth05", 3),
    ("Grid.GitHub", "github.com/GugulothBhuvan", 3),
    ("Grid.Facebook", "n/a", 3)
]

# --- 1. Background Segmentation ---
def segment_background(img_path):
    print("Loading image and segmenting background...")
    img = Image.open(img_path)
    img_resized = img.resize((PORTRAIT_COLS, PORTRAIT_ROWS), Image.Resampling.LANCZOS)
    data_rgb = np.array(img_resized)
    data_gray = np.array(img_resized.convert('L'))

    # Peach background reference color
    bg_color = np.array([250, 159, 140])
    dist = np.sqrt(np.sum((data_rgb - bg_color) ** 2, axis=2))

    # Texture calculation using local standard deviation
    local_mean = ndimage.uniform_filter(data_gray.astype(float), size=5)
    local_sq_mean = ndimage.uniform_filter(data_gray.astype(float)**2, size=5)
    local_std = np.sqrt(np.maximum(0.0, local_sq_mean - local_mean**2))

    # Background is flat (low std) AND close to peach color
    bg_mask = (dist < 45) & (local_std < 1.5)
    mask = ~bg_mask

    # Morphological clean up
    mask_closed = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    mask_filled = ndimage.binary_fill_holes(mask_closed)
    
    labeled, num = ndimage.label(mask_filled)
    if num > 0:
        sizes = ndimage.sum(mask_filled, labeled, range(1, num + 1))
        largest_label = np.argmax(sizes) + 1
        mask_final = labeled == largest_label
    else:
        mask_final = mask_filled

    # Trim boundaries to prevent edge artifacts
    mask_final[:3, :] = False
    mask_final[:, :3] = False
    mask_final[:, -3:] = False
    
    return img_resized, mask_final

# --- 2. Image Preprocessing & Dithering ---
def preprocess_image(img):
    # Contrast 1.3x
    img_contrast = ImageEnhance.Contrast(img).enhance(1.3)
    # Autocontrast (cutoff=1)
    img_auto = ImageOps.autocontrast(img_contrast, cutoff=1)
    # UnsharpMask (radius=3, percent=140)
    img_sharp = img_auto.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=3))
    return np.array(img_sharp.convert('L')).astype(float)

def dither_serpentine(gray_img, mask):
    h, w = gray_img.shape
    dithered = np.zeros((h, w), dtype=np.uint8)
    err_buf = gray_img.copy()
    
    for y in range(h):
        # Serpentine order: left-to-right on even, right-to-left on odd
        if y % 2 == 0:
            x_range = range(w)
            direction = 1
        else:
            x_range = range(w - 1, -1, -1)
            direction = -1
            
        for x in x_range:
            if not mask[y, x]:
                dithered[y, x] = 0
                err_buf[y, x] = 0
                continue
                
            old_px = err_buf[y, x]
            new_px = 255 if old_px > 127 else 0
            dithered[y, x] = new_px
            err = old_px - new_px
            
            # Floyd-Steinberg error distribution
            if 0 <= x + direction < w:
                err_buf[y, x + direction] += err * 7/16
            if y + 1 < h:
                if 0 <= x - direction < w:
                    err_buf[y + 1, x - direction] += err * 3/16
                err_buf[y + 1, x] += err * 5/16
                if 0 <= x + direction < w:
                    err_buf[y + 1, x + direction] += err * 1/16
    return dithered

def fit_dither_density(gray, mask, target_dots=17000):
    print("Fitting image intensity to target dot count...")
    low, high = 0.05, 1.5
    best_factor = 1.0
    best_dither = None
    best_dots = 0
    
    for i in range(10):
        factor = (low + high) / 2
        d = dither_serpentine(gray * factor, mask)
        dots = np.sum(d == 255)
        
        if abs(dots - target_dots) < abs(best_dots - target_dots) or best_dots == 0:
            best_dots = dots
            best_factor = factor
            best_dither = d
            
        if dots > target_dots:
            high = factor
        else:
            low = factor
    print(f"Optimal factor: {best_factor:.3f} yields {best_dots} dots.")
    return best_dither, best_dots

# --- 3. Logo Generation & Alignment ---
def get_flutter_points(n_points):
    img = Image.open('c:/Users/bhuva/OneDrive/Desktop/4x/New folder/GugulothBhuvan/icons8-flutter-48.png')
    img = img.resize((200, 200), Image.Resampling.LANCZOS)
    data = np.array(img)
    mask = data[:, :, 3] > 50
    coords = np.argwhere(mask)
    
    cy, cx = np.mean(coords, axis=0)
    coords = coords - [cy, cx]
    max_val = np.max(np.abs(coords))
    coords = coords / max_val * 90
    
    np.random.seed(42)
    idx = np.random.choice(len(coords), n_points, replace=False)
    pts = coords[idx]
    pts = pts[:, [1, 0]]  # swap x, y
    return pts + LOGO_CENTER

def get_code_glyph_points(n_points):
    img = Image.new('L', (300, 300), 0)
    draw = ImageDraw.Draw(img)
    draw.line([(100, 80), (50, 150), (100, 220)], fill=255, width=25)
    draw.line([(200, 80), (250, 150), (200, 220)], fill=255, width=25)
    draw.line([(170, 70), (130, 230)], fill=255, width=25)
    
    data = np.array(img)
    coords = np.argwhere(data > 127)
    cy, cx = np.mean(coords, axis=0)
    coords = coords - [cy, cx]
    max_val = np.max(np.abs(coords))
    coords = coords / max_val * 90
    
    np.random.seed(42)
    idx = np.random.choice(len(coords), n_points, replace=False)
    pts = coords[idx]
    pts = pts[:, [1, 0]]
    return pts + LOGO_CENTER

def get_react_points(n_points):
    img = Image.new('L', (300, 300), 0)
    draw = ImageDraw.Draw(img)
    for angle in [0, 60, 120]:
        ell = Image.new('L', (300, 300), 0)
        d = ImageDraw.Draw(ell)
        d.ellipse([50, 110, 250, 190], outline=255, width=15)
        ell_rot = ell.rotate(angle, resample=Image.Resampling.BICUBIC)
        img = ImageChops.lighter(img, ell_rot)
        
    draw.ellipse([130, 130, 170, 170], fill=255)
    
    data = np.array(img)
    coords = np.argwhere(data > 127)
    cy, cx = np.mean(coords, axis=0)
    coords = coords - [cy, cx]
    max_val = np.max(np.abs(coords))
    coords = coords / max_val * 90
    
    np.random.seed(42)
    idx = np.random.choice(len(coords), n_points, replace=False)
    pts = coords[idx]
    pts = pts[:, [1, 0]]
    return pts + LOGO_CENTER

def generate_traced_logos():
    print("Tracing logos and solving optimal transport assignments...")
    pts_flutter = get_flutter_points(N_TRAVELLERS)
    pts_code = get_code_glyph_points(N_TRAVELLERS)
    pts_react = get_react_points(N_TRAVELLERS)
    
    # Align Code to Flutter
    d1 = np.sum((pts_flutter[:, None, :] - pts_code[None, :, :])**2, axis=2)
    _, col1 = linear_sum_assignment(d1)
    pts_code_aligned = pts_code[col1]
    
    # Align React to Code
    d2 = np.sum((pts_code_aligned[:, None, :] - pts_react[None, :, :])**2, axis=2)
    _, col2 = linear_sum_assignment(d2)
    pts_react_aligned = pts_react[col2]
    
    return pts_flutter, pts_code_aligned, pts_react_aligned

# --- 4. Dotted Leaders ---
def get_monospaced_leader(label, value, total_chars=65):
    if not label:
        return ""
    space_for_dots = total_chars - len(label) - len(value) - 2
    dots_count = max(0, space_for_dots)
    dots_str = "." * dots_count
    return f"{label} {dots_str} {value}"

# --- 5. Metrics & Verification ---
def calculate_evenness_metric(dots, groups):
    # Centroid deviation of intro groups
    cx_overall = np.mean(dots[:, 0])
    cy_overall = np.mean(dots[:, 1])
    distances = []
    for g in range(60):
        g_dots = dots[groups == g]
        if len(g_dots) == 0:
            continue
        cx_g = np.mean(g_dots[:, 0])
        cy_g = np.mean(g_dots[:, 1])
        distances.append(np.sqrt((cx_g - cx_overall)**2 + (cy_g - cy_overall)**2))
    return np.mean(distances) / PORTRAIT_ROWS

def calculate_boundary_metric(dots, bands):
    # R-squared of predicting group index from x + y
    proj = dots[:, 0] + dots[:, 1]
    band_means = np.array([np.mean(proj[bands == g]) for g in range(N_BANDS)])
    residuals = proj - band_means[bands]
    metric = np.std(residuals) / np.std(proj)
    return metric

# --- 6. SVG Output Generation ---
def make_svg_path_run(dots_coords):
    """Compact RLE encoding: group dots by row, merge consecutive x into runs."""
    if len(dots_coords) == 0:
        return ""
    # Group by y (row)
    rows = {}
    for x, y in dots_coords:
        ix = int(X_START_PORTRAIT + x)
        iy = int(Y_START_PORTRAIT + y)
        rows.setdefault(iy, []).append(ix)
    
    parts = []
    for iy in sorted(rows):
        xs = sorted(rows[iy])
        # Merge consecutive x values into runs
        i = 0
        while i < len(xs):
            start = xs[i]
            while i + 1 < len(xs) and xs[i + 1] == xs[i] + 1:
                i += 1
            run_len = xs[i] - start + 1
            parts.append(f"M{start},{iy}h{run_len}v1h-{run_len}z")
            i += 1
    return "".join(parts)

def make_travellers_path_data(coords):
    parts = []
    for x, y in coords:
        parts.append(f"M{x:.0f},{y:.0f}h2v2h-2z")
    return "".join(parts)

def write_svg_file(filename, dithered_dots, is_dark):
    bg_color = DARK_BG if is_dark else LIGHT_BG
    chrome_color = DARK_CHROME if is_dark else LIGHT_CHROME
    dots_color = DARK_DOTS if is_dark else LIGHT_DOTS
    accent_color = DARK_ACCENT if is_dark else LIGHT_ACCENT
    text_color = DARK_TEXT if is_dark else LIGHT_TEXT
    leader_color = DARK_LEADER if is_dark else LIGHT_LEADER
    
    # Extract dot coordinates
    dots = np.argwhere(dithered_dots == 255) # shape (M, 2), contains (y, x)
    dots = dots[:, [1, 0]] # swap to (x, y)
    
    # 1. Intro Groups (60 groups)
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(dots))
    intro_groups = shuffled_idx % 60
    
    # 2. Drift Bands (94 bands) with noise (sigma=4)
    noise = np.random.normal(0, 4.0, len(dots))
    proj = dots[:, 0] + dots[:, 1] + noise
    sort_idx = np.argsort(proj)
    sorted_dots = dots[sort_idx]
    
    # Assign to 94 bands
    band_size = len(dots) // N_BANDS
    drift_bands = np.zeros(len(dots), dtype=int)
    for g in range(N_BANDS):
        drift_bands[g*band_size : (g+1)*band_size] = g
    drift_bands[N_BANDS*band_size:] = N_BANDS - 1
    
    # Traced Logos for Travellers
    pts_f, pts_c, pts_r = generate_traced_logos()
    d_flutter = make_travellers_path_data(pts_f)
    d_code = make_travellers_path_data(pts_c)
    d_react = make_travellers_path_data(pts_r)
    
    # Centroid of Flutter logo for drift direction
    cx_f, cy_f = np.mean(pts_f, axis=0)
    
    # Build Intro SVG Content
    intro_svg_elements = []
    for g in range(60):
        g_dots = dots[intro_groups == g]
        d_path = make_svg_path_run(g_dots)
        
        # Timing calculation
        delay = 1.2 * (g / 59)
        k1 = delay / 3.2
        k2 = (delay + 0.8) / 3.2
        
        element = f"""    <path d="{d_path}" fill="{dots_color}" shape-rendering="crispEdges" opacity="0">
      <animate attributeName="opacity" dur="3.2s" fill="freeze" values="0;0;1;1" keyTimes="0;{k1:.4f};{k2:.4f};1" begin="0s" />
    </path>"""
        intro_svg_elements.append(element)
        
    intro_layer = "\n".join(intro_svg_elements)
    
    # Build Loop Portrait SVG Content
    loop_svg_elements = []
    for g in range(N_BANDS):
        g_dots = sorted_dots[drift_bands == g]
        d_path = make_svg_path_run(g_dots)
        
        # Centroid of this band
        if len(g_dots) > 0:
            cx_g = X_START_PORTRAIT + np.mean(g_dots[:, 0]) * PORTRAIT_DX
            cy_g = Y_START_PORTRAIT + np.mean(g_dots[:, 1]) * PORTRAIT_DY
        else:
            cx_g, cy_g = CX_LEFT, CY_LEFT
            
        # Drift vector: 42% towards Flutter centroid
        dx = 0.42 * (cx_f - cx_g)
        dy = 0.42 * (cy_f - cy_g)
        
        # Timing
        delay = 0.7 * (g / 93)
        t1 = (3.0 + delay) / 14.2
        t2 = (3.0 + delay + 0.6) / 14.2
        t3 = (12.9 + 0.7 * (1.0 - g / 93)) / 14.2
        t4 = (12.9 + 0.7 * (1.0 - g / 93) + 0.6) / 14.2
        
        element = f"""    <g opacity="0">
      <animate attributeName="opacity" dur="14.2s" repeatCount="indefinite" begin="3.2s"
               values="1;1;0;0;1;1" keyTimes="0;{t1:.4f};{t2:.4f};{t3:.4f};{t4:.4f};1" />
      <path d="{d_path}" fill="{dots_color}" shape-rendering="crispEdges">
        <animateTransform attributeName="transform" type="translate" dur="14.2s" repeatCount="indefinite" begin="3.2s"
                          values="0,0;0,0;{dx:.1f},{dy:.1f};{dx:.1f},{dy:.1f};0,0;0,0"
                          keyTimes="0;{t1:.4f};{t2:.4f};{t3:.4f};{t4:.4f};1" />
      </path>
    </g>"""
        loop_svg_elements.append(element)
        
    loop_layer = "\n".join(loop_svg_elements)
    
    # Info Panel Rows
    info_rows = []
    y_pos = 135
    for label, val, section in INFO_DETAILS:
        if not label:
            y_pos += 15
            continue
            
        row = f"""    <text x="480" y="{y_pos}" textLength="650" lengthAdjust="spacingAndGlyphs" 
          fill="{text_color}" font-family="ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace" font-size="14">
      {label} <tspan fill="{leader_color}">{"." * (65 - len(label) - len(val) - 2)}</tspan> {val}
    </text>"""
        info_rows.append(row)
        y_pos += 23

    rows_layer = "\n".join(info_rows)
    
    # Assemble final SVG
    svg_content = f"""<svg width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <style>
    .terminal-title {{
      font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
      font-size: 14px;
      font-weight: bold;
    }}
    .panel-label {{
      font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
      font-size: 13px;
      font-weight: bold;
    }}
    .live-badge {{
      font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
      font-size: 12px;
      font-weight: bold;
    }}
    .pill-text {{
      font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
      font-size: 13px;
      font-weight: bold;
    }}
  </style>

  <!-- Background -->
  <rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="{bg_color}" />

  <!-- Terminal Window Top Chrome -->
  <rect x="10" y="10" width="1160" height="35" rx="6" ry="6" fill="{"#1E293B" if is_dark else "#E2E8F0"}" />
  <circle cx="30" cy="27" r="6" fill="#EF4444" />
  <circle cx="50" cy="27" r="6" fill="#F59E0B" />
  <circle cx="70" cy="27" r="6" fill="#10B981" />
  <text x="95" y="32" fill="{chrome_color}" class="terminal-title">profile.sh --live</text>

  <!-- Left Pane (VISUAL.MAP) -->
  <rect x="20" y="55" width="410" height="535" rx="5" ry="5" stroke="{chrome_color}" stroke-width="2" fill="none" />
  <rect x="35" y="47" width="105" height="16" fill="{bg_color}" />
  <text x="42" y="59" fill="{chrome_color}" class="panel-label">VISUAL.MAP</text>

  <!-- Right Pane (SYSTEM.INFO) -->
  <rect x="450" y="55" width="710" height="535" rx="5" ry="5" stroke="{chrome_color}" stroke-width="2" fill="none" />
  <rect x="465" y="47" width="110" height="16" fill="{bg_color}" />
  <text x="472" y="59" fill="{chrome_color}" class="panel-label">SYSTEM.INFO</text>

  <!-- Handle Pill & LIVE Badge -->
  <rect x="480" y="80" width="165" height="24" rx="12" fill="{accent_color}" />
  <text x="562.5" y="96" fill="{"#0A101F" if is_dark else "#FFFFFF"}" text-anchor="middle" class="pill-text">@GugulothBhuvan</text>
  
  <circle cx="675" cy="92" r="5" fill="#EF4444">
    <animate attributeName="opacity" values="1;0.2;1" dur="1.5s" repeatCount="indefinite" />
  </circle>
  <text x="687" y="96" fill="#EF4444" class="live-badge">LIVE</text>

  <!-- Info Rows Readout -->
{rows_layer}

  <!-- PORTRAIT LAYER 1 (Intro Shimmer, 0s to 3.2s) -->
  <g id="intro-portrait">
    <animate attributeName="opacity" dur="3.2s" fill="freeze" values="1;1;0" keyTimes="0;0.99;1" begin="0s" />
{intro_layer}
  </g>

  <!-- PORTRAIT LAYER 2 (Loop Portrait, begin 3.2s) -->
  <g id="loop-portrait">
{loop_layer}
  </g>

  <!-- TRAVELLERS LAYER (Logo Morphing, begin 3.2s) -->
  <g id="travellers">
    <path fill="{dots_color}" shape-rendering="crispEdges" opacity="0">
      <animate attributeName="opacity" dur="14.2s" repeatCount="indefinite" begin="3.2s"
               values="0;0;1;1;1;1;1;1;0"
               keyTimes="0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1" />
      <animate attributeName="d" dur="14.2s" repeatCount="indefinite" begin="3.2s"
               values="{d_react};{d_react};{d_flutter};{d_flutter};{d_code};{d_code};{d_react};{d_react};{d_react}"
               keyTimes="0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1" />
    </path>
  </g>

</svg>
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg_content)
        
    print(f"Generated {filename} ({os.path.getsize(filename)/1024:.1f} KB)")
    
    # Calculate and report metrics
    evenness = calculate_evenness_metric(dots, intro_groups)
    boundary = calculate_boundary_metric(sorted_dots, drift_bands)
    print(f"Validation metrics for {filename}:")
    print(f"  Evenness metric (intro): {evenness:.3f} (target ~0.05 good)")
    print(f"  Straight-boundary metric (drift): {boundary:.3f} (target ~0.01 organic)")

# --- Main Pipeline Execution ---
if __name__ == "__main__":
    photo_path = 'c:/Users/bhuva/OneDrive/Desktop/4x/New folder/GugulothBhuvan/me_edit.jpeg'
    
    # 1. Segment
    img_resized, mask_final = segment_background(photo_path)
    
    # 2. Preprocess
    gray = preprocess_image(img_resized)
    
    # 3. Fit density & dither
    best_dither, dot_count = fit_dither_density(gray, mask_final, target_dots=17000)
    
    # 4. Generate Dark Mode SVG
    write_svg_file("c:/Users/bhuva/OneDrive/Desktop/4x/New folder/GugulothBhuvan/dark.svg", best_dither, is_dark=True)
    
    # 5. Generate Light Mode SVG
    write_svg_file("c:/Users/bhuva/OneDrive/Desktop/4x/New folder/GugulothBhuvan/light.svg", best_dither, is_dark=False)
    
    print("Done! Banners generated successfully.")
