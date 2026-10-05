#!/usr/bin/env python3
"""
Build the chunking benchmark: four PDFs of different shapes plus questions
with exact answer keys.

Facts are invented (made-up products, people, stars, functions), so a model
cannot answer from memory. It has to retrieve. Each question has a `key`,
the exact string a correct answer must contain, and a `fact`, the sentence
the key comes from. That makes scoring mechanical: no LLM judge decides
whether an answer is right.

    document        shape                          why it is here
    manual          numbered headings + prose      structure should win
    projects        continuous prose, no headings  recursive should hold its own
    fieldnotes      topic after topic, no headings semantic's best case
    codebase        Python source                  code-aware's best case

Each question appears in two forms: `direct` reuses the document's wording,
`paraphrase` swaps it for synonyms, which is where dense retrieval and
query rewriting have to earn their keep.

    python make_bench.py          # writes eval/bench/
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parent / "bench"
DOCS = OUT / "docs"
rng = random.Random(7)


def pick(seq):
    return seq[rng.randrange(len(seq))]


# ---------------------------------------------------------------- PDF writing

class Doc(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(20, 18, 20)
        self.add_page()

    def heading(self, text: str, size: int = 15):
        self.ln(3)
        self.set_font("Helvetica", "B", size)
        self.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def para(self, text: str):
        self.set_font("Helvetica", "", 10.5)
        self.multi_cell(0, 5.2, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2.5)

    def code(self, lines: list[str]):
        self.set_font("Courier", "", 9)
        for line in lines:
            self.multi_cell(0, 4.2, line if line else " ", new_x="LMARGIN", new_y="NEXT")


questions: list[dict] = []


def q(doc: str, fact: str, key: str, direct: str, para: str, kind: str):
    questions.append(
        {"doc": doc, "fact": fact, "key": key, "direct": direct, "paraphrase": para, "kind": kind}
    )


# ------------------------------------------------------------------- manual

MODELS = [f"{a} {b}" for a, b in zip(
    ["Zephyr", "Quillon", "Marlow", "Tessel", "Orrin", "Halvard", "Brindle", "Caspian",
     "Dovrin", "Eskel", "Fennick", "Garrow", "Hollis", "Ivarra", "Jessup", "Kestrel"],
    [f"X{n}" for n in range(200, 1000, 50)])]
CAUSES = ["a blocked intake fan", "a failed thermal fuse", "a loose antenna lead",
          "a worn drive belt", "a corrupted settings chip", "a depleted backup cell",
          "a jammed paper guide", "a cracked sensor lens", "an overheated relay",
          "a misaligned encoder", "a dry pump seal", "a shorted ribbon cable",
          "a stuck exhaust valve", "a faulty door switch", "a clogged coolant line",
          "an unseated memory card", "a bent drive spindle", "a tripped surge guard",
          "a frozen heating coil", "a warped bearing plate"]
FIXES = ["clear the housing and cycle power", "replace the fuse and reseat the cover",
         "tighten the lead and recalibrate", "swap the belt and reset the counter",
         "reflash the chip from the service card", "install a fresh cell and recharge for an hour",
         "open the guide and remove the sheet", "clean the lens with the supplied cloth",
         "let the unit rest for twenty minutes", "re-run the alignment routine twice",
         "add seal oil and bleed the line", "replace the cable and lock the latch",
         "free the valve and test the vent", "re-seat the door and press it firmly",
         "flush the line with the cleaning kit", "push the card in until it clicks",
         "install the spare spindle", "reset the guard and check the outlet",
         "thaw the coil and check the thermostat", "replace the plate and torque the screws"]
BUTTONS = ["Reset", "Mode", "Sync", "Eject", "Link", "Boost", "Purge", "Pair", "Lock", "Dim"]
ACTIONS = ["restore factory settings", "enter service mode", "clear the job queue",
           "start a deep clean", "pair a new remote", "unlock the panel", "run a self-test",
           "export the log", "flush the cache", "calibrate the sensor"]
PARTS = ["intake fan", "main battery", "drive belt", "display panel", "power adapter", "antenna",
         "cooling pump", "sensor array", "hinge assembly", "filter cartridge", "exhaust valve",
         "control board", "heating coil", "speaker unit", "latch mechanism"]
ATTRS = [("maximum load", "kg"), ("operating temperature", "degrees Celsius"),
         ("rated voltage", "volts"), ("noise level", "decibels"),
         ("service interval", "hours"), ("peak current", "amperes")]
SYN = {"hold": "held down", "seconds": "seconds", "restore factory settings": "wipe the unit back to its defaults",
       "enter service mode": "switch into maintenance mode", "clear the job queue": "empty the pending jobs",
       "start a deep clean": "begin an intensive cleaning cycle", "pair a new remote": "link another handheld remote",
       "unlock the panel": "release the control lock", "run a self-test": "launch the built-in diagnostics",
       "export the log": "save the event history", "flush the cache": "purge stored temporary data",
       "calibrate the sensor": "tune the detector"}


def build_manual():
    d = Doc()
    d.heading("Operator Manual: Zephyr Product Line", 20)
    used_codes: set[int] = set()
    for ci, model in enumerate(MODELS, start=1):
        d.heading(f"Chapter {ci}: {model}", 16)
        # error codes
        d.heading(f"{ci}.1 Error Codes", 13)
        facts = []
        for _ in range(8):
            code = rng.randrange(100, 500)
            while code in used_codes:
                code = rng.randrange(100, 500)
            used_codes.add(code)
            cause, fix = pick(CAUSES), pick(FIXES)
            sent = f"On the {model}, error E-{code} indicates {cause}; to clear it, {fix}."
            facts.append(sent)
            q("manual", sent, cause,
              f"What does error E-{code} mean on the {model}?",
              f"The {model} is showing code E-{code}. What has gone wrong inside it?", "error")
        for i in range(0, 8, 4):
            d.para(" ".join(facts[i:i + 4]))
        # controls
        d.heading(f"{ci}.2 Controls", 13)
        facts = []
        combos = [(b, a) for b in BUTTONS for a in ACTIONS]
        rng.shuffle(combos)
        for button, action in combos[:6]:
            n = rng.randrange(3, 30)
            sent = f"On the {model}, hold the {button} button for {n} seconds to {action}."
            facts.append(sent)
            q("manual", sent, f"{n} seconds",
              f"How many seconds do you hold the {button} button on the {model} to {action}?",
              f"On the {model}, for how long must the {button} control be {SYN['hold']} in order to {SYN[action]}?",
              "control")
        for i in range(0, 6, 3):
            d.para(" ".join(facts[i:i + 3]))
        # specs
        d.heading(f"{ci}.3 Specifications", 13)
        facts = []
        combos = [(p, a) for p in PARTS for a in ATTRS]
        rng.shuffle(combos)
        for part, (attr, unit) in combos[:6]:
            v = rng.randrange(5, 990)
            sent = f"The {part} of the {model} has a {attr} of {v} {unit}."
            facts.append(sent)
            q("manual", sent, f"{v} {unit}",
              f"What is the {attr} of the {part} on the {model}?",
              f"For the {model}, which figure is given for the {part} regarding its {attr.replace('maximum', 'upper').replace('rated', 'nominal')}?",
              "spec")
        for i in range(0, 6, 3):
            d.para(" ".join(facts[i:i + 3]))
    d.output(str(DOCS / "manual.pdf"))


# ----------------------------------------------------------------- projects

SYL = ["al", "dor", "vex", "mir", "tan", "ro", "kel", "zun", "pri", "os", "ba", "len", "qui", "dra", "fon"]
FIRST = ["Maren", "Odile", "Tobias", "Ingrid", "Casimir", "Leona", "Rurik", "Selma", "Anselm", "Greta",
         "Ivo", "Thea", "Bram", "Yvette", "Corin", "Dagny"]
LAST = ["Halloran", "Vestergaard", "Okonkwo", "Brandt", "Pellegrino", "Takahashi", "Lindqvist", "Moreau",
        "Castellan", "Draycott", "Fairweather", "Gutierrez", "Hargreave", "Ilves", "Jansen", "Kowalczyk"]
CITIES = ["Tromso", "Valparaiso", "Krakow", "Dunedin", "Mombasa", "Reykjavik", "Bergen", "Porto",
          "Hobart", "Tallinn", "Cusco", "Lyon", "Gdansk", "Cork", "Split", "Turku"]


def build_projects():
    d = Doc()
    d.heading("A History of the Northfield Laboratories", 20)
    names: set[str] = set()
    for _ in range(110):
        name = "".join(pick(SYL) for _ in range(3)).capitalize()
        while name in names:
            name = "".join(pick(SYL) for _ in range(3)).capitalize()
        names.add(name)
        lead = f"{pick(FIRST)} {pick(LAST)}"
        year = rng.randrange(1962, 2021)
        budget = rng.randrange(2, 480)
        city = pick(CITIES)
        staff = rng.randrange(12, 900)
        fl = f"Project {name} was led by {lead}."
        fy = f"It began in {year} and was funded with a budget of {budget} million dollars."
        fc = f"The work took place in {city}, where {staff} staff were assigned to it."
        d.para(f"{fl} {fy} {fc} Later reviews described the effort as steady rather than spectacular, "
               f"and its records were archived without ceremony once the final report was filed.")
        q("projects", fl, lead, f"Who led Project {name}?",
          f"Which person was in charge of the {name} initiative at Northfield?", "lead")
        q("projects", fy, f"{budget} million", f"What budget did Project {name} receive?",
          f"How much money was the {name} programme given?", "budget")
        q("projects", fc, f"{staff} staff", f"How many staff worked on Project {name}?",
          f"What was the headcount assigned to the {name} effort?", "staff")
    d.output(str(DOCS / "projects.pdf"))


# --------------------------------------------------------------- fieldnotes

def build_fieldnotes():
    """Six unrelated domains, one after another, twice round, no headings."""
    d = Doc()
    d.heading("Collected Field Notes", 20)
    domains = {
        "star": ("The star {n} lies {a} light-years from the Sun and shines at magnitude {b}.",
                 "light-years", "How far from the Sun is the star {n}?", "What is the distance to the star {n}?",
                 "magnitude", "What is the apparent magnitude of the star {n}?", "How bright does the star {n} appear?"),
        "boat": ("The sloop {n} measures {a} feet on deck and carries {b} square metres of sail.",
                 "feet", "How long is the sloop {n}?", "What is the deck length of the sloop {n}?",
                 "square metres", "How much sail area does the sloop {n} carry?", "What canvas does the sloop {n} spread?"),
        "recipe": ("The {n} stew calls for {a} grams of barley and simmers for {b} minutes.",
                   "grams", "How much barley goes into the {n} stew?", "What weight of barley does the {n} stew need?",
                   "minutes", "How long does the {n} stew simmer?", "For what duration is the {n} stew cooked?"),
        "mineral": ("The mineral {n} has a hardness of {a} on the Mohs scale and a density of {b} grams per cubic centimetre.",
                    "on the Mohs", "How hard is the mineral {n}?", "What Mohs rating does the mineral {n} have?",
                    "grams per cubic", "What is the density of the mineral {n}?", "How heavy is the mineral {n} per unit volume?"),
        "plant": ("The shrub {n} grows to {a} centimetres and blooms for {b} weeks each spring.",
                  "centimetres", "How tall does the shrub {n} grow?", "What height does the shrub {n} reach?",
                  "weeks", "How many weeks does the shrub {n} bloom?", "For how long does the shrub {n} flower?"),
        "bridge": ("The {n} bridge spans {a} metres and was opened in {b}.",
                   "metres", "What is the span of the {n} bridge?", "How wide is the gap crossed by the {n} bridge?",
                   "opened", "In which year did the {n} bridge open?", "When was the {n} bridge inaugurated?"),
    }
    names = set()
    def uname():
        n = "".join(pick(SYL) for _ in range(2)).capitalize()
        while n in names:
            n = "".join(pick(SYL) for _ in range(2)).capitalize() + pick(["a", "is", "or", "an", "us"])
        names.add(n)
        return n
    for round_ in range(2):
        for dom in list(domains):
            tpl, u1, q1, p1, u2, q2, p2 = domains[dom]
            for _ in range(20):
                n = uname()
                a, b = rng.randrange(11, 990), rng.randrange(3, 99)
                if dom == "bridge":
                    b = rng.randrange(1840, 2015)
                fact = tpl.format(n=n, a=a, b=b)
                d.para(fact + " " + pick([
                    "Notes from the visit were filed alongside the others.",
                    "Observers recorded little else of interest that season.",
                    "The entry was copied from an older ledger."]))
                if dom == "mineral":
                    q("fieldnotes", fact, f"{a}", q1.format(n=n), p1.format(n=n), "a")
                    q("fieldnotes", fact, f"{b} grams per cubic", q2.format(n=n), p2.format(n=n), "b")
                elif dom == "bridge":
                    q("fieldnotes", fact, f"{a} metres", q1.format(n=n), p1.format(n=n), "a")
                    q("fieldnotes", fact, f"{b}", q2.format(n=n), p2.format(n=n), "b")
                else:
                    q("fieldnotes", fact, f"{a} {u1}", q1.format(n=n), p1.format(n=n), "a")
                    q("fieldnotes", fact, f"{b} {u2}", q2.format(n=n), p2.format(n=n), "b")
    d.output(str(DOCS / "fieldnotes.pdf"))


# ----------------------------------------------------------------- codebase

VERBS = ["fetch", "merge", "split", "audit", "rotate", "encode", "resolve", "compact", "verify", "stage",
         "export", "retire"]
NOUNS = ["orders", "tokens", "shards", "ledgers", "sessions", "invoices", "profiles", "manifests",
         "queues", "reports", "snapshots", "tickets"]


def build_codebase():
    d = Doc()
    lines: list[str] = ['"""Order pipeline utilities."""', "", "import time", ""]
    seen: set[str] = set()
    combos = [(v, n) for v in VERBS for n in NOUNS]
    rng.shuffle(combos)
    for v, n in combos[:90]:
        fname = f"{v}_{n}"
        retries = rng.randrange(2, 19)
        cap = rng.randrange(20, 990)
        timeout = rng.randrange(3, 120)
        facts = (f"{fname} retries {retries} times before giving up.",
                 f"{fname} returns at most {cap} rows.",
                 f"The default timeout of {fname} is {timeout} seconds.")
        lines += [f"def {fname}(source, limit={cap}, timeout={timeout}):",
                  f'    """{facts[0]}',
                  f"    {facts[1]}",
                  f"    {facts[2]}",
                  '    """',
                  f"    for attempt in range({retries}):",
                  "        rows = source.read(limit)",
                  "        if rows:",
                  "            return rows[:limit]",
                  "        time.sleep(1)",
                  "    return []", ""]
        q("codebase", facts[0], f"{retries} times", f"How many times does {fname} retry?",
          f"How many attempts does the function {fname} make before it stops trying?", "retry")
        q("codebase", facts[1], f"{cap} rows", f"What is the maximum number of rows {fname} returns?",
          f"What is the upper bound on the rows handed back by {fname}?", "cap")
        q("codebase", facts[2], f"{timeout} seconds", f"What is the default timeout of {fname}?",
          f"How long does {fname} wait by default before timing out?", "timeout")
    d.code(lines)
    d.output(str(DOCS / "codebase.pdf"))


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    build_manual()
    build_projects()
    build_fieldnotes()
    build_codebase()
    per_doc: dict[str, int] = {}
    for item in questions:
        per_doc[item["doc"]] = per_doc.get(item["doc"], 0) + 1
    # Keep a balanced, reproducible sample: 40 per document.
    sample: list[dict] = []
    for doc in per_doc:
        pool = [x for x in questions if x["doc"] == doc]
        rng.shuffle(pool)
        sample.extend(pool[:40])
    with open(OUT / "questions.jsonl", "w") as f:
        for i, item in enumerate(sample):
            item["id"] = f"b{i:03d}"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print({k: v for k, v in per_doc.items()}, "-> sampled", len(sample))
