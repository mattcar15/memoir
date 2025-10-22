# live_visible_windows.py
import time, json, sys, os, subprocess, argparse
from datetime import datetime
from pathlib import Path

from Quartz import (
    # CoreGraphics window listing
    CGWindowListCopyWindowInfo,
    kCGNullWindowID,
    kCGWindowListOptionOnScreenOnly,
    kCGWindowListExcludeDesktopElements,
    kCGWindowBounds,
    kCGWindowLayer,
    kCGWindowOwnerName,
    kCGWindowOwnerPID,
    kCGWindowName,
    kCGWindowNumber,
    kCGWindowAlpha,
    kCGWindowSharingState,
    kCGWindowMemoryUsage,
)

from ApplicationServices import (
    AXIsProcessTrustedWithOptions,
    AXUIElementCreateApplication,
    AXUIElementCopyAttributeValue,
    AXValueGetType,
    AXValueGetValue,
    kAXWindowsAttribute,
    kAXRoleAttribute,
    kAXTitleAttribute,
    kAXDescriptionAttribute,
    kAXChildrenAttribute,
    kAXPositionAttribute,
    kAXSizeAttribute,
    kAXValueCGPointType,
    kAXValueCGSizeType,
    kAXTrustedCheckOptionPrompt,  # optional: to show the permission prompt
)

import objc  # for AXIsProcessTrustedWithOptions(dict) prompt

AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})

from AppKit import NSWorkspace
from PIL import Image, ImageDraw, ImageFont

ONSCREEN_OPTS = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements

# Create snapshots directories
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True)

MARKED_SNAPSHOT_DIR = Path(__file__).parent / "marked_snapshots"

# Debug log file
DEBUG_LOG = Path(__file__).parent / "window_debug.log"


def debug_log(msg):
    """Write debug message to log file with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with open(DEBUG_LOG, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")


# ---------- AX Helpers ----------


def ax_copy(element, attr):
    """Return (ok, value) for an AX attribute, where ok==True if err==0 and value is not None."""
    try:
        err, value = AXUIElementCopyAttributeValue(element, attr, None)
        return (err == 0 and value is not None), value
    except Exception as e:
        debug_log(f"ax_copy error for attr {attr}: {e}")
        return (False, None)


def ax_point(axval):
    """Unpack AX CGPoint value -> (x, y) ints, or None."""
    try:
        if axval is None or AXValueGetType(axval) != kAXValueCGPointType:
            return None
        # PyObjC automatically handles the buffer - pass None and get the struct back
        success, point = AXValueGetValue(axval, kAXValueCGPointType, None)
        if success and point:
            # point is a CGPoint struct with x and y attributes
            return int(point.x), int(point.y)
        return None
    except Exception as e:
        debug_log(f"ax_point error: {e}")
        return None


def ax_size(axval):
    """Unpack AX CGSize value -> (w, h) ints, or None."""
    try:
        if axval is None or AXValueGetType(axval) != kAXValueCGSizeType:
            return None
        # PyObjC automatically handles the buffer - pass None and get the struct back
        success, size = AXValueGetValue(axval, kAXValueCGSizeType, None)
        if success and size:
            # size is a CGSize struct with width and height attributes
            return int(size.width), int(size.height)
        return None
    except Exception as e:
        debug_log(f"ax_size error: {e}")
        return None


def get_accessibility_elements(pid, window_bounds):
    """Extract accessibility elements for a given application PID."""
    elements = []
    try:
        debug_log(f"Getting accessibility elements for PID {pid}")
        debug_log(f"Window bounds: {window_bounds}")
        app_ref = AXUIElementCreateApplication(pid)
        if not app_ref:
            debug_log(f"Failed to create app ref for PID {pid}")
            return elements

        ok, windows = ax_copy(app_ref, kAXWindowsAttribute)
        if not ok or not windows:
            debug_log(f"No windows found (ok={ok}, windows={windows})")
            return elements

        debug_log(f"Found {len(windows)} windows")
        # Get the first window (usually the focused one)
        window = windows[0]
        elements.extend(_extract_elements_recursive(window, window_bounds, 0))
        debug_log(f"Extracted {len(elements)} elements total")
    except Exception as e:
        debug_log(f"Error getting accessibility elements: {e}")
        import traceback
        import io

        s = io.StringIO()
        traceback.print_exc(file=s)
        debug_log(s.getvalue())

    return elements


def _extract_elements_recursive(element, window_bounds, depth, max_depth=10):
    """Recursively extract UI elements with their positions and roles."""
    if depth > max_depth:
        return []

    elements = []
    elements_at_depth = 0
    try:
        ok_pos, pos_val = ax_copy(element, kAXPositionAttribute)
        ok_size, size_val = ax_copy(element, kAXSizeAttribute)
        ok_role, role = ax_copy(element, kAXRoleAttribute)
        ok_desc, description = ax_copy(element, kAXDescriptionAttribute)
        ok_title, title = ax_copy(element, kAXTitleAttribute)

        pt = ax_point(pos_val) if ok_pos else None
        sz = ax_size(size_val) if ok_size else None

        if pt and sz:
            x = pt[0] - int(window_bounds["x"])
            y = pt[1] - int(window_bounds["y"])
            w, h = sz

            # Only include elements within the window bounds and with reasonable size
            # Allow smaller elements (5px instead of 10px) to catch more UI components
            if x >= 0 and y >= 0 and w > 5 and h > 5 and w < 4000 and h < 4000:
                label = ""
                if role:
                    label = str(role).replace("AX", "")
                if title and str(title).strip():
                    label += f": {str(title)[:30]}"
                elif description and str(description).strip():
                    label += f": {str(description)[:30]}"

                elements.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "role": str(role) if role else "Unknown",
                        "label": label or "UI",
                    }
                )
                elements_at_depth += 1
                if depth <= 2 and elements_at_depth <= 3:
                    debug_log(
                        f"Depth {depth}: Added element {role} at ({x},{y}) size {w}x{h}"
                    )

        ok_children, children = ax_copy(element, kAXChildrenAttribute)
        if ok_children and children:
            if depth <= 2:
                debug_log(f"Depth {depth}: Found {len(children)} children")
            for child in children:
                elements.extend(
                    _extract_elements_recursive(
                        child, window_bounds, depth + 1, max_depth
                    )
                )

    except Exception as e:
        if depth == 0:
            debug_log(f"Exception at depth {depth}: {e}")

    return elements


# ---------- Imaging ----------


def markup_image(image_path, elements):
    """Add bounding boxes and labels to an image based on accessibility elements."""
    try:
        debug_log(f"\nmarkup_image called with {len(elements)} elements")
        img = Image.open(image_path)
        debug_log(f"Image size: {img.size}, mode: {img.mode}")

        # Convert to RGBA to support transparency in overlays
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Create a transparent overlay for drawing
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Try to use a system font, fallback to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except Exception:
            font = ImageFont.load_default()

        colors = {
            "AXButton": (255, 0, 0, 255),  # Red
            "AXTextField": (0, 255, 0, 255),  # Green
            "AXStaticText": (0, 0, 255, 255),  # Blue
            "AXImage": (255, 165, 0, 255),  # Orange
            "AXGroup": (128, 128, 128, 255),  # Gray
        }
        default_color = (255, 255, 0, 255)  # Yellow

        drawn_count = 0
        for i, elem in enumerate(elements):
            role = elem.get("role", "Unknown")
            color = colors.get(role, default_color)

            if i < 10:  # Log first 10 elements
                debug_log(
                    f"Element {i}: role={role}, x={elem['x']}, y={elem['y']}, w={elem['w']}, h={elem['h']}, label={elem.get('label', '')[:30]}"
                )

            # Rectangle
            draw.rectangle(
                [elem["x"], elem["y"], elem["x"] + elem["w"], elem["y"] + elem["h"]],
                outline=color,
                width=2,
            )
            drawn_count += 1

            # Label
            label = elem.get("label", "")
            if label:
                # draw text background
                text_bbox = draw.textbbox(
                    (elem["x"], max(0, elem["y"] - 16)), label, font=font
                )
                draw.rectangle(text_bbox, fill=(0, 0, 0, 200))
                draw.text(
                    (text_bbox[0] + 2, text_bbox[1] + 1),
                    label,
                    fill=(255, 255, 255, 255),
                    font=font,
                )

        debug_log(f"Drew {drawn_count} rectangles on overlay")

        # Composite the overlay onto the original image
        img = Image.alpha_composite(img, overlay)
        debug_log(f"Composite complete")

        return img
    except Exception as e:
        debug_log(f"Error marking up image: {e}")
        import traceback
        import io

        s = io.StringIO()
        traceback.print_exc(file=s)
        debug_log(s.getvalue())
        return None


def take_screenshot(window_info, markup_regions=False):
    """Take a screenshot of the specified window and save it to snapshots directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    app_name = window_info["app"].replace("/", "_").replace(" ", "_")
    title = window_info["title"].replace("/", "_").replace(" ", "_")[:50]
    filename = f"{timestamp}_{app_name}_{title}.png"

    base_dir = SNAPSHOT_DIR
    filepath = base_dir / filename

    window_id = window_info["id"]
    try:
        subprocess.run(
            ["screencapture", "-o", "-l", str(window_id), "-x", str(filepath)],
            check=True,
            capture_output=True,
        )

        if markup_regions:
            MARKED_SNAPSHOT_DIR.mkdir(exist_ok=True)

            # Load the screenshot to get actual image dimensions
            from PIL import Image as PILImage

            img = PILImage.open(filepath)
            img_width, img_height = img.size
            img.close()

            window_bounds = {
                "x": window_info["x"],
                "y": window_info["y"],
                "w": window_info["w"],
                "h": window_info["h"],
            }

            # Calculate scale factor (for retina displays)
            scale_x = img_width / window_info["w"] if window_info["w"] > 0 else 1.0
            scale_y = img_height / window_info["h"] if window_info["h"] > 0 else 1.0
            debug_log(
                f"Image size: {img_width}x{img_height}, Window size: {window_info['w']}x{window_info['h']}, Scale: {scale_x}x{scale_y}"
            )

            elements = get_accessibility_elements(window_info["pid"], window_bounds)

            # Scale element coordinates to match image size
            for elem in elements:
                elem["x"] = int(elem["x"] * scale_x)
                elem["y"] = int(elem["y"] * scale_y)
                elem["w"] = int(elem["w"] * scale_x)
                elem["h"] = int(elem["h"] * scale_y)

            marked_img = markup_image(filepath, elements)
            if marked_img:
                marked_filepath = MARKED_SNAPSHOT_DIR / filename
                marked_img.save(marked_filepath)
                print(f"  → Saved marked version with {len(elements)} regions")

        return filepath
    except subprocess.CalledProcessError as e:
        print(f"Error capturing screenshot: {e}")
        return None


# ---------- Window listing ----------


def take_snapshot(include_nonzero_layers=True):
    raw = CGWindowListCopyWindowInfo(ONSCREEN_OPTS, kCGNullWindowID) or []
    snap = []
    for order, w in enumerate(raw):
        layer = w.get(kCGWindowLayer, 0)
        if not include_nonzero_layers and layer != 0:
            continue
        b = w.get(kCGWindowBounds, {})
        snap.append(
            {
                "order": order,  # 0 = topmost on screen
                "id": w.get(kCGWindowNumber),
                "pid": w.get(kCGWindowOwnerPID),
                "app": w.get(kCGWindowOwnerName, "") or "",
                "title": w.get(kCGWindowName, "") or "",
                "x": int(b.get("X", 0)),
                "y": int(b.get("Y", 0)),
                "w": int(b.get("Width", 0)),
                "h": int(b.get("Height", 0)),
                "layer": layer,
                "alpha": w.get(kCGWindowAlpha, 1.0),
                "sharing": w.get(
                    kCGWindowSharingState, 0
                ),  # 1=cannot share, 2=can share
                "mem": int((w.get(kCGWindowMemoryUsage, 0) or 0) / 1024),
            }
        )
    return snap


def index_by_id(snap):
    return {w["id"]: w for w in snap if w.get("id") is not None}


def diff(prev_idx, curr_idx):
    prev_ids, curr_ids = set(prev_idx), set(curr_idx)
    opened = curr_ids - prev_ids
    closed = prev_ids - curr_ids
    changed = []
    for wid in prev_ids & curr_ids:
        a, b = prev_idx[wid], curr_idx[wid]
        if (a["x"], a["y"], a["w"], a["h"]) != (b["x"], b["y"], b["w"], b["h"]) or a[
            "title"
        ] != b["title"]:
            changed.append((wid, a, b))
    return opened, closed, changed


def clear():
    sys.stdout.write("\x1b[2J\x1b[H")


def print_table(rows):
    hdr = f"{'TOP':<3} {'ORD':>3} {'LAYER':>5} {'APP':<26} {'TITLE':<46} {'PID':>6} {'WINID':>8}   {'(x,y) w×h':<20} {'α':>2} {'share':>5} {'memKB':>7}"
    print(hdr)
    print("-" * len(hdr))
    top_order = min((r["order"] for r in rows), default=9999)
    for r in rows[:20]:
        top = "✓" if (r["layer"] == 0 and r["order"] == top_order) else ""
        app = (r["app"][:23] + "…") if len(r["app"]) > 24 else r["app"]
        title = (r["title"][:43] + "…") if len(r["title"]) > 44 else r["title"]
        bounds = f"({r['x']},{r['y']}) {r['w']}×{r['h']}"
        print(
            f"{top:<3} {r['order']:>3} {r['layer']:>5} {app:<26} {title:<46} {r['pid']:>6} {r['id']:>8}   {bounds:<20} {int(r['alpha']):>2} {r['sharing']:>5} {r['mem']:>7}"
        )


def main():
    # Clear/initialize debug log file
    with open(DEBUG_LOG, "w") as f:
        f.write(
            f"=== Window Monitor Debug Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        )

    # Ensure AX trust / prompt if needed
    AXIsProcessTrustedWithOptions({objc._C_ID: True})

    parser = argparse.ArgumentParser(
        description="Monitor visible windows and optionally capture screenshots with accessibility markup"
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Enable screenshot capture on window/title changes",
    )
    parser.add_argument(
        "--markup-regions",
        action="store_true",
        help="Create marked-up versions of screenshots with accessibility regions (implies --capture)",
    )
    args = parser.parse_args()

    capture_enabled = args.capture or args.markup_regions
    markup_enabled = args.markup_regions

    prev = take_snapshot(include_nonzero_layers=True)
    prev_idx = index_by_id(prev)
    last_event_lines = []

    prev_top_level = [w for w in prev if w["layer"] == 0]
    prev_topmost = (
        min(prev_top_level, key=lambda r: r["order"]) if prev_top_level else None
    )

    try:
        while True:
            curr = take_snapshot(include_nonzero_layers=True)
            curr_idx = index_by_id(curr)

            opened, closed, changed = diff(prev_idx, curr_idx)

            top_level = [w for w in curr if w["layer"] == 0]
            curr_topmost = (
                min(top_level, key=lambda r: r["order"]) if top_level else None
            )

            if prev_topmost and curr_topmost:
                if (
                    prev_topmost["app"] != curr_topmost["app"]
                    or prev_topmost["id"] != curr_topmost["id"]
                ):
                    if capture_enabled:
                        filepath = take_screenshot(
                            curr_topmost, markup_regions=markup_enabled
                        )
                        if filepath:
                            last_event_lines.append(
                                f"[SCREENSHOT] Topmost app changed to {curr_topmost['app']} → {filepath.name}"
                            )

            event_lines = []
            for wid in opened:
                w = curr_idx[wid]
                event_lines.append(
                    f"[OPEN]  {w['app']} — '{w['title']}' id={w['id']} ({w['x']},{w['y']} {w['w']}x{w['h']}) layer={w['layer']} ord={w['order']}"
                )
            for wid in closed:
                a = prev_idx[wid]
                event_lines.append(f"[CLOSE] {a['app']} — '{a['title']}' id={wid}")
            for wid, a, b in changed:
                if (a["x"], a["y"], a["w"], a["h"]) != (b["x"], b["y"], b["w"], b["h"]):
                    event_lines.append(
                        f"[MOVE/RESIZE] id={wid} ({a['x']},{a['y']} {a['w']}x{a['h']}) → ({b['x']},{b['y']} {b['w']}x{b['h']})"
                    )
                if a["title"] != b["title"]:
                    event_lines.append(
                        f"[TITLE] id={wid} '{a['title']}' → '{b['title']}'"
                    )
                    if capture_enabled and curr_topmost and wid == curr_topmost["id"]:
                        filepath = take_screenshot(
                            curr_topmost, markup_regions=markup_enabled
                        )
                        if filepath:
                            event_lines.append(
                                f"[SCREENSHOT] Title change for topmost app → {filepath.name}"
                            )

            if event_lines:
                last_event_lines = (last_event_lines + event_lines)[-6:]

            clear()
            status = "Monitor only"
            if capture_enabled:
                status = "Capture enabled"
                if markup_enabled:
                    status += " + Accessibility markup"
            print(
                f"Visible top-level windows (active Space/display) — refresh 4 Hz — {status}"
            )
            print(f"Screenshots saved to: {SNAPSHOT_DIR}")
            if markup_enabled:
                print(f"Marked screenshots saved to: {MARKED_SNAPSHOT_DIR}")
                print(f"Debug log: {DEBUG_LOG}")
            if top_level:
                ttl = min(top_level, key=lambda r: r["order"])
                print(
                    f"Topmost top-level: {ttl['app']} — '{ttl['title']}' id={ttl['id']} order={ttl['order']}\n"
                )
            print_table(top_level)
            if last_event_lines:
                print("\nRecent events:")
                for line in last_event_lines:
                    print("  " + line)

            prev, prev_idx = curr, curr_idx
            prev_topmost = curr_topmost
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
