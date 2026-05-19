import pathlib

BS = chr(92)
NL = chr(10)
DQ = chr(34)
BT2 = chr(96) * 2
SQ2 = chr(39) * 2
LSMART = chr(8220)
RSMART = chr(8221)
EMDASH = chr(8212)
ENDASH = chr(8211)
AGRAVE = chr(224)

target = pathlib.Path("C:/Users/mahau/OneDrive/Desktop/projects/Law-as-code-aif/rebuild_docx.py")

L = []
a = L.append

# -- Header --
a("#!/usr/bin/env python3")
a("import re, os")
a("from pathlib import Path")
a("from docx import Document")
a("from docx.shared import Pt, Inches, Cm, RGBColor")
a("from docx.enum.text import WD_ALIGN_PARAGRAPH")
a("")
a("TEX = Path(" + DQ + "C:/Users/mahau/OneDrive/Desktop/projects/Law-as-code-aif/paper/main.tex" + DQ + ")")
a("OUTD = Path(" + DQ + "c:/Users/mahau/Downloads/Law as code" + DQ + ")")
a("OUTP = OUTD / " + DQ + "lawascode.docx" + DQ)
a("FIGD = Path(" + DQ + "c:/Users/mahau/Downloads/Law as code/fig_images" + DQ + ")")
a("FM = {")
for f in ["fig1_minimization","fig2_geofence","fig3_emergency","fig4_summary","fig5_overhead"]:
    a("    " + DQ + f + DQ + ": FIGD / " + DQ + f + ".png" + DQ + ",")
a("}")

target.write_text(NL.join(L), encoding="utf-8")
print("Wrote", len(L), "lines to", target)