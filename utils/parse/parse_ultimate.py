from bs4 import BeautifulSoup


def _text_of(node):
    return " ".join(node.get_text(" ", strip=True).split())


def _parse_table_anchors(table):
    """Yield child nodes for <tr> rows that contain <a id="..."> inside the left <td>."""
    children = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        left = tds[0]
        a = left.find("a", id=True)
        if not a:
            continue
        rule_id = a["id"]
        a.extract()  # remove anchor so left td text is clean
        title = _text_of(left)
        # Optional note/value in right td (often a number or phrase)
        note = _text_of(tds[1]) if len(tds) > 1 else None
        node = {
            "id": rule_id,
            "label": None,
            "title": title,
            "tooltips": [],
            "annotations": [],
            "children": [],
        }
        if note:
            node["note"] = note
        children.append(node)
    return children


def _parse_li(li):
    """Parse one <li> into a node dict and its nested children."""
    # Find the first <a id="..."> in this li (the rule's anchor)
    anchor = None
    for child in li.contents:
        if getattr(child, "name", None) == "a" and child.has_attr("id"):
            anchor = child
            break

    label = anchor.get_text(" ", strip=True) if anchor else None
    rule_id = anchor["id"] if anchor else None

    # Build the visible title from content before the first nested structure
    # But exclude annotation and tooltip spans from title text
    title_parts, children, hit_nested = [], [], False
    for child in li.contents:
        tag = getattr(child, "name", None)
        if tag in ("ul", "ol"):
            hit_nested = True
            for sub_li in child.find_all("li", recursive=False):
                children.append(_parse_li(sub_li))
        elif tag == "table":
            hit_nested = True
            children.extend(_parse_table_anchors(child))
        elif not hit_nested:
            if child is anchor:
                continue
            if hasattr(child, "get_text"):
                # Skip annotation and tooltip spans when building title
                if tag == "span" and child.get("class"):
                    classes = (
                        child.get("class")
                        if isinstance(child.get("class"), list)
                        else [child.get("class")]
                    )
                    if "annotation" in classes or "tooltip" in classes:
                        continue
                title_parts.append(child.get_text(" ", strip=True))
            else:
                title_parts.append(str(child).strip())
        # ignore tail after nested content for the title
    title = " ".join(p for p in title_parts if p).strip()

    # Collect extras anywhere within this <li>
    tooltips = [
        {"text": t.get_text(strip=True), "title": t.get("title")}
        for t in li.find_all("span", class_="tooltip")
    ]
    annotations = [_text_of(a) for a in li.find_all("span", class_="annotation")]

    return {
        "id": rule_id,
        "label": label,  # e.g., "15.A.5.b."
        "title": title,  # human-readable text
        "tooltips": tooltips,  # [{"text": "...", "title": "..."}]
        "annotations": annotations,  # ["...", "..."]
        "children": children,  # recursive list of child nodes
    }


def load_rules(html_path=None, html_string=None):
    """
    Returns a Python list[dict] of rule nodes (top-level sections), fully nested.
    Use either html_path or html_string.
    """
    if (html_path is None) == (html_string is None):
        raise ValueError("Provide exactly one of html_path or html_string.")
    if html_path:
        with open(html_path, "rb") as f:
            soup = BeautifulSoup(f, "html.parser")
    else:
        soup = BeautifulSoup(html_string, "html.parser")

    # Your file has multiple top-level <li> elements (no outer <ul>/<ol>), so parse them directly.
    roots = [_parse_li(li) for li in soup.find_all("li", recursive=False)]
    return roots


def flatten(tree):
    """Flatten nested nodes into a list."""
    out = []

    def walk(nodes):
        for n in nodes:
            out.append(n)
            walk(n["children"])

    walk(tree)
    return out


def build_index(tree):
    """Map rule id -> node (for quick lookups)."""
    idx = {}
    for n in flatten(tree):
        if n["id"]:
            idx[n["id"]] = n
    return idx


# --- minimal usage example ---
if __name__ == "__main__":
    rules = load_rules(html_path="../rules/ultimate.html")
    index = build_index(rules)
    print(f"Top-level sections: {len(rules)}")
    print(f"Total rules (unique ids): {len(index)}")
    # Example lookup:
    sample_id = "15.A.5.b.3.a"
    node = index.get(sample_id)
    if node:
        print(sample_id, "→", node["title"], node.get("note"))
