from bs4 import BeautifulSoup
import re

"""
Parser for PDGA Disc Golf rules HTML structure.

Key differences from Ultimate frisbee rules:
- Uses nested <div class="section-X"> structure instead of <li> lists
- Rules are numbered like "801.01" instead of "15.A.5.b"
- Includes Q&A sections with IDs like "QA-APP-1"
- Has both rule sections and appendix/manual sections
"""


def _text_of(node):
    return " ".join(node.get_text(" ", strip=True).split())


def _extract_rule_id_from_heading(heading_text):
    """Extract rule ID from heading text like '801.01 Fairness' -> '801.01'"""
    match = re.match(r"^(\d+(?:\.\d+)*)", heading_text.strip())
    return match.group(1) if match else None


def _parse_qa_section(details_element):
    """Parse Q&A details section into a node structure"""
    # Extract ID from details element
    qa_id = details_element.get("id")
    if not qa_id:
        return None

    # Get question from h3 element
    question_h3 = details_element.find("h3", class_="question")
    question_text = _text_of(question_h3) if question_h3 else ""

    # Get answer from p element
    answer_p = details_element.find("p", class_="answer")
    answer_text = _text_of(answer_p) if answer_p else ""

    # Combine question and answer for title
    title = question_text
    if answer_text:
        title += f" Answer: {answer_text}"

    return {
        "id": qa_id,
        "label": qa_id,
        "title": title,
        "tooltips": [],
        "annotations": [],
        "children": [],
    }


def _parse_section_div(div):
    """Parse a section div into a node with its children"""
    # Get the h1 heading
    heading = div.find("h1", class_="book-heading", recursive=False)
    if not heading:
        return None

    heading_text = _text_of(heading)
    rule_id = _extract_rule_id_from_heading(heading_text)

    # Create the node
    node = {
        "id": rule_id,
        "label": rule_id,
        "title": heading_text,
        "tooltips": [],
        "annotations": [],
        "children": [],
    }

    # Find child section divs - look for immediate children only
    child_divs = div.find_all("div", class_=re.compile(r"section-\d+"), recursive=False)

    for child_div in child_divs:
        child_node = _parse_section_div(child_div)
        if child_node:
            node["children"].append(child_node)

    # Also look for Q&A details within this section (but not recursively in child sections)
    qa_details = div.find_all("details", id=True, recursive=True)
    for qa_detail in qa_details:
        # Check if this detail is actually in a child section - if so, skip it
        parent_section = qa_detail.find_parent("div", class_=re.compile(r"section-\d+"))
        if parent_section and parent_section != div:
            continue

        qa_node = _parse_qa_section(qa_detail)
        if qa_node:
            node["children"].append(qa_node)

    return node


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

    # Find all top-level section divs (section-3 seems to be the main level based on the samples)
    main_sections = soup.find_all("div", class_=re.compile(r"section-3"))

    roots = []
    for section in main_sections:
        node = _parse_section_div(section)
        if node and node["title"]:  # Only include sections with titles
            roots.append(node)

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
    rules = load_rules(html_path="../rules/discgolf.html")
    index = build_index(rules)
    print(f"Top-level sections: {len(rules)}")
    print(f"Total rules (unique ids): {len(index)}")

    # Show some example rules
    print("\nExample rules found:")
    for rule_id, node in list(index.items())[:10]:
        print(
            f"{rule_id} → {node['title'][:100]}{'...' if len(node['title']) > 100 else ''}"
        )

    # Test specific rule lookup
    print("\nTesting specific rule lookups:")
    test_ids = ["801.01", "802.01", "QA-APP-1"]
    for test_id in test_ids:
        node = index.get(test_id)
        if node:
            print(f"{test_id} → {node['title']}")
        else:
            print(f"{test_id} → NOT FOUND")

    # Show hierarchy structure
    print("\nTop-level sections with child counts:")
    for rule in rules:
        child_count = len(rule["children"])
        print(f"{rule['id']} ({child_count} children): {rule['title']}")
