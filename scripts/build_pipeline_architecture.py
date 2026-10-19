"""Render the documented PayPal pipeline as standalone HTML, SVG and PNG.

Run from any directory: python3 scripts/build_pipeline_architecture.py
Requires Pillow only for PNG rendering. No credentials or network calls.
Diagram content reflects paypal_sec_10k.ipynb; update it when the pipeline changes.
"""
from html import escape
from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
STEM = "paypal_sec_rag_pipeline_architecture"
W, H = 1400, 2290
SCALE = 2
canvas = Image.new("RGB", (W * SCALE, H * SCALE), "#ffffff")
draw = ImageDraw.Draw(canvas)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
       '<title id="title">PayPal SEC RAG pipeline architecture</title>',
       '<desc id="desc">SEC ingestion, fixed and semantic chunking, MPNet embeddings, Pinecone storage, three retrieval experiments, Nebius generation, and RAGAS evaluation. A separate callout describes the interactive chain.</desc>']
PALETTE = {
    "blue": ("#eff6ff", "#93c5fd", "#193e72"),
    "orange": ("#fff7ed", "#fdba74", "#7c3715"),
    "purple": ("#f5f3ff", "#c4b5fd", "#50328a"),
    "teal": ("#f0fdfa", "#5eead4", "#125c56"),
    "green": ("#f0fdf4", "#86cf9a", "#205a35"),
    "gray": ("#f8fafc", "#cbd5e1", "#334155"),
}


def font(size, bold=False):
    candidates = [
        Path('/System/Library/Fonts/Supplemental') / ('Arial Bold.ttf' if bold else 'Arial.ttf'),
        Path('/usr/share/fonts/truetype/dejavu') / ('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'),
        Path('C:/Windows/Fonts') / ('arialbd.ttf' if bold else 'arial.ttf'),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size * SCALE)
    raise RuntimeError('Install Arial or DejaVu Sans to render the PNG.')


def text(x, y, value, size=18, color="#475569", bold=False):
    svg.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}">{escape(value)}</text>')
    draw.text((x * SCALE, y * SCALE), value, font=font(size, bold), fill=color, anchor="ls")


def rect(x, y, w, h, fill, stroke, radius=14):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    draw.rounded_rectangle((x*SCALE, y*SCALE, (x+w)*SCALE, (y+h)*SCALE), radius*SCALE, fill=fill, outline=stroke, width=3)


def arrow(points, color="#94a3b8"):
    svg.append('<polyline points="' + ' '.join(f'{x},{y}' for x,y in points) + f'" fill="none" stroke="{color}" stroke-width="2"/>')
    draw.line([(x*SCALE,y*SCALE) for x,y in points], fill=color, width=4)
    x,y = points[-1]
    px,py = points[-2]
    a = math.atan2(y-py,x-px)
    head = [(x,y), (x-9*math.cos(a-.5),y-9*math.sin(a-.5)), (x-9*math.cos(a+.5),y-9*math.sin(a+.5))]
    svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in head)+f'" fill="{color}"/>')
    draw.polygon([(x*SCALE,y*SCALE) for x,y in head],fill=color)


def box(x, y, w, h, title, lines, theme="gray"):
    fill, stroke, ink = PALETTE[theme]
    rect(x,y,w,h,fill,stroke)
    title_size = 22
    while draw.textlength(title, font=font(title_size, True)) / SCALE > w-40:
        title_size -= 1
    assert title_size >= 16, title
    text(x+20,y+32,title,title_size,ink,True)
    for i,line in enumerate(lines):
        # Fail on overflowing copy rather than silently clipping exported diagrams.
        size = 18
        while draw.textlength(line, font=font(size)) / SCALE > w-40:
            size -= 1
        assert size >= 16, line
        text(x+20,y+62+i*25,line,size)


def phase(y, n, title, steps):
    text(48,y,f"0{n}  /  {title.upper()}",18,"#1e293b",True)
    text(1110,y,steps,16,"#64748b")


rect(0,0,W,H,"#ffffff","#ffffff",0)
rect(0,0,W,150,"#102c4f","#102c4f",0)
text(48,43,"PAYPAL  /  FINANCIAL RAG",16,"#93c5fd",True)
text(48,88,"SEC filings to evidence-backed answers",38,"#ffffff",True)
text(48,123,"Implementation architecture  •  paypal_sec_10k.ipynb  •  Current local reports: FY 2023–2025",18,"#cbd5e1")

phase(189,1,"Ingest & prepare filing evidence","Notebook steps 1–6")
box(48,210,400,145,"SEC EDGAR · PYPL",[
    "PayPal Holdings, Inc. · CIK 0001633917",
    "sec_edgar_downloader · latest 3 Form 10-Ks",
    "Raw full-submission.txt + manifest.json"],"blue")
box(500,210,380,145,"Load & preserve provenance",[
    "LangChain TextLoader + filing metadata",
    "Source path, SEC URL, report / filing date",
    "Accession number retained downstream"],"blue")
box(932,210,420,145,"Extract & clean",[
    "Select main 10-K; exclude separate exhibits",
    "Strip tags, decode entities, tidy whitespace",
    "Persist cleaned text → documents.json"],"blue")
arrow([(448,282),(500,282)])
arrow([(880,282),(932,282)])
arrow([(1142,355),(1142,380),(640,380),(640,425)])
arrow([(1142,380),(1030,380),(1030,425)])

phase(410,2,"Split the same cleaned reports","Notebook steps 7–13")
box(48,425,640,162,"A  /  Fixed character chunks",[
    "RecursiveCharacterTextSplitter",
    "Maximum 512 characters · target overlap 64 characters",
    "Preserve filing metadata + start_index",
    "Stable SHA-256 IDs for repeatable uploads"],"orange")
box(712,425,640,162,"B  /  Semantic chunks",[
    "SemanticChunker uses MPNet · buffer 1 · percentile 95",
    "Secondary split: at most 8,000 characters, no overlap",
    "Keep parent / piece offsets · stable SHA-256 IDs",
    "Validate text + metadata payload ≤ 40,000 bytes"],"purple")
arrow([(370,587),(370,637)])
arrow([(1030,587),(1030,637)])

phase(622,3,"Embed & store","Notebook steps 8–14")
box(48,637,1304,112,"Shared local embedding model · normalized 768-dimensional vectors",[
    "sentence-transformers/multi-qa-mpnet-base-cos-v1 via HuggingFaceEmbeddings",
    "Used for semantic boundaries, document embeddings, and question embeddings"],"teal")
arrow([(370,749),(370,775)])
arrow([(1030,749),(1030,775)])
box(48,775,640,112,'Pinecone · namespace="fixed"',[
    "Cosine index · vectors + chunk text + source metadata",
    "Replace fixed namespace; upload batches of 100"],"orange")
box(712,775,640,112,'Pinecone · namespace="semantic"',[
    "Same index and filings · independent chunk collection",
    "Replace semantic namespace; upload batches of 100"],"purple")
text(48,916,"Missing index: create serverless on AWS us-east-1. Existing index: validate 768 dimensions and cosine metric.",17)

phase(961,4,"Retrieve & compare three experiments","Notebook step 21")
box(48,982,1304,88,"Question → same MPNet model → cosine similarity search",[
    "10 evaluation questions · no date filter in experiments · semantic candidate pool shared by B and C"],"gray")
arrow([(267,1070),(267,1100)])
arrow([(700,1070),(700,1100)])
arrow([(1133,1070),(1133,1100)])
box(48,1100,416,184,"A  /  fixed_only",[
    "Search fixed namespace → top 3",
    "Keep cosine retrieval order",
    "No reranking",
    "3 passages → generation"],"orange")
box(492,1100,416,184,"B  /  semantic_only",[
    "Search semantic namespace → top 10",
    "Keep first 3 in cosine retrieval order",
    "No reranking",
    "3 passages → generation"],"purple")
box(936,1100,416,184,"C  /  semantic_rerank",[
    "Reuse B's 10 semantic candidates",
    "Qwen on Nebius scores usefulness 0–4",
    "Stable score sort → retain top 3",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "LLM scoring via /v1/chat/completions"],"green")
arrow([(267,1284),(267,1337),(700,1337),(700,1383)])
arrow([(700,1284),(700,1337)])
arrow([(1133,1284),(1133,1337),(700,1337)])

phase(1368,5,"Generate a cited answer for each experiment","Notebook steps 18, 21")
box(48,1383,400,142,"Assemble evidence prompt",[
    "Question + top 3 passages + metadata",
    "Assign source IDs [S1], [S2], [S3]",
    "Same prompt across A, B and C"],"blue")
box(500,1383,400,142,"Nebius answer generation",[
    "Default: openai/gpt-oss-120b",
    "NEBIUS_GENERATION_MODEL override",
    "Temperature 0 · bounded 429 retries"],"purple")
box(952,1383,400,142,"Answer + evidence trace",[
    "Prompt requires citations and units",
    "Identify periods; abstain if unsupported",
    "Save contexts, metadata, usage, errors"],"green")
arrow([(448,1453),(500,1453)])
arrow([(900,1453),(952,1453)])
arrow([(1152,1525),(1152,1552),(700,1552),(700,1598)])

phase(1583,6,"Evaluate saved answers & compare","Notebook steps 20–23")
box(48,1598,400,164,"Reference evaluation set",[
    "10 question / ground_truth pairs",
    "data/paypal_eval_dataset.json",
    "Evidence: paypal_eval_sources.json",
    "Ground truth used only for evaluation"],"blue")
box(500,1598,400,164,"RAGAS · Nebius judge",[
    "Faithfulness · context precision · recall",
    "Default Qwen3-235B-A22B-Instruct-2507",
    "Full model prefix: Qwen/",
    "NEBIUS_EVAL_MODEL override"],"teal")
box(952,1598,400,164,"Compare A / B / C",[
    "Per-question scores + means",
    "Valid / missing score coverage",
    "Paired changes: B − A and C − B",
    "CSV / JSON + analysis_report.txt"],"green")
arrow([(448,1680),(500,1680)])
arrow([(900,1680),(952,1680)])
text(48,1792,"Results: data/experiments/<run>/results_*.json, traces.json, errors.json → ragas_<run>/ score and analysis files",17)
text(48,1820,"Ground truth is attached after generation. The small development set and model judgments do not establish a winning strategy.",17)

phase(1871,7,"Interactive chain & inspection","Notebook steps 15–19")
box(48,1892,1304,140,"Notebook question → semantic retrieval (20 candidates) → optional Qwen reranking → top 3 → Nebius answer",[
    "LangChain RunnableLambda: retrieve_rag_context | generate_rag_answer; use_reranker defaults to False.",
    "Optional metadata_filter (for example, report_date). The example and four-question end-to-end check enable reranking.",
    "Inspect passages, source dates and citation IDs in the notebook; end-to-end results saved to data/rag_e2e_results.json."],"gray")

rect(48,2062,1304,132,"#f8fafc","#cbd5e1")
text(68,2094,"IMPLEMENTATION NOTES",16,"#334155",True)
text(68,2125,"scripts/run_rag_experiments.py runs the three-experiment collector against existing namespaces; it does not write the index.",17)
text(68,2152,"Cleanup is basic; hidden inline XBRL may remain. Long semantic pieces may exceed MPNet's input token limit.",17)
text(68,2179,"The diagram documents implemented flow and defaults; citation checks validate structure, while factual accuracy needs review.",17)
text(48,2231,"STACK   LangChain  /  MPNet  /  Pinecone  /  Nebius  /  Qwen  /  GPT-OSS  /  RAGAS",17,"#193e72",True)
text(48,2265,"Source: paypal_sec_10k.ipynb + scripts/run_rag_experiments.py  •  Architecture snapshot: 18 October 2026",16,"#64748b")

svg.append('</svg>')
svg_text = '\n'.join(svg)
(ROOT / f'{STEM}.svg').write_text(svg_text, encoding='utf-8')
canvas.save(ROOT / f'{STEM}.png', optimize=True)
html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PayPal SEC RAG · Pipeline architecture</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#e9eef4;color:#24364b;font-family:Arial,sans-serif}
nav{max-width:1400px;margin:auto;display:flex;gap:18px;align-items:center;padding:18px 24px;flex-wrap:wrap}
nav strong{margin-right:auto}a{color:#174c85;text-underline-offset:4px}button{border:1px solid #94a3b8;border-radius:6px;background:white;padding:8px 14px;cursor:pointer}
main{max-width:1400px;margin:0 auto;background:white;box-shadow:0 8px 40px #102c4f12}svg{display:block;width:100%;height:auto}
footer{max-width:1400px;margin:auto;padding:22px 24px;line-height:1.6;font-size:14px}
@media(max-width:700px){main{overflow-x:auto}svg{min-width:1000px}nav{font-size:14px}}
@media print{@page{size:A3 portrait;margin:8mm}body{background:white}nav,footer{display:none}main{box-shadow:none}svg{width:100%;min-width:0}}
</style></head><body>
<nav aria-label="Diagram exports"><strong>PayPal SEC RAG / Architecture</strong>
<a href="STEM.png" download>Download PNG</a><a href="STEM.svg" download>Download SVG</a>
<button type="button" onclick="window.print()">Print / Save PDF</button></nav>
<main>DIAGRAM</main>
<footer>Implementation source: <a href="paypal_sec_10k.ipynb">paypal_sec_10k.ipynb</a> and
<a href="scripts/run_rag_experiments.py">run_rag_experiments.py</a>.
Layout inspired by <a href="https://github.com/The-Gen-Academy/2B-Financial-RAG-Jev/blob/main/financial_rag_pipeline_architecture.html">The Gen Academy reference architecture</a>.
The diagram is self-contained and works offline. On narrow screens, scroll horizontally to inspect details.</footer>
</body></html>'''.replace('STEM', STEM).replace('DIAGRAM', svg_text)
(ROOT / f'{STEM}.html').write_text(html, encoding='utf-8')
print(f'Created {STEM}.html, .svg, .png ({canvas.width} × {canvas.height})')
