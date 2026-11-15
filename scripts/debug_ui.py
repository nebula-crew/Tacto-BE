"""로컬 디버그 화면: 사진 한 장을 올리면 판정 결과와 함께 마스크·중심선·판정 위치를 겹쳐 보여 준다.

서버(app/)와 따로 도는 개발용 도구라 배포 이미지에는 들어가지 않는다 (.dockerignore가 scripts/를 뺌).
판정은 서비스와 똑같이 pipeline.run.run_photo로 하고, 그림만 따로 그린다. 저장은 하지 않는다.

    .venv/bin/python scripts/debug_ui.py            # → http://localhost:8001
    PORT=8002 .venv/bin/python scripts/debug_ui.py

모델은 화면에서 weights/*.pt 중에 고른다. 기본값은 배포와 같은 Dockerfile 기본값(v11 + scene4_v1).
블록 칸에서 v11 같은 블록 종류 모델을 고르면 서버와 똑같이 SceneTypeSegmenter(장면 tactile 영역 → 선형·점형)로 돌린다.
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.dual_segment import DualSegmenter  # noqa: E402
from pipeline.scene_type_segment import SceneTypeSegmenter, is_type_checkpoint  # noqa: E402
from pipeline.mask_refine import refine_mask  # noqa: E402
from pipeline.run import analyze_label, run_photo  # noqa: E402
from pipeline.schema import MaskClass, PhotoKind  # noqa: E402

WEIGHTS = ROOT / "weights"
DEFAULT_BLOCK, DEFAULT_SCENE = "v11_best.pt", "scene4_v1_best.pt"
COLORS = {  # BGR
    MaskClass.LINEAR: (0, 200, 255),
    MaskClass.DOT: (0, 120, 255),
    MaskClass.CROSSWALK: (255, 0, 255),
    MaskClass.ROAD: (255, 140, 0),
}
VERDICT_COLOR = {"suspected": (40, 40, 230), "hold": (120, 120, 120), "normal": (60, 170, 60)}

app = FastAPI(title="TactileCheck 로컬 디버그")
_segmenters: dict = {}


class _Recorder:
    """segmenter를 감싸 run_photo가 받은 라벨을 그림용으로 기록한다 (판정 로직은 건드리지 않음)."""

    def __init__(self, inner):
        self.inner, self.label = inner, None
        self.available_classes = getattr(inner, "available_classes", None)

    def predict(self, image, photo):
        label, seg = self.inner.predict(image, photo)
        self.label = label.copy()
        return label, seg


def _segmenter(block: str, scene: str):
    key = (block, scene)
    if key not in _segmenters:
        b, s = WEIGHTS / block, WEIGHTS / scene
        if not b.is_file() or not s.is_file():
            raise HTTPException(400, f"가중치 파일이 없음: {block}, {scene}")
        _segmenters[key] = SceneTypeSegmenter(b, s) if is_type_checkpoint(b) else DualSegmenter(b, s)
    return _segmenters[key]


def _png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _overlay(image: np.ndarray, label: np.ndarray, result) -> np.ndarray:
    vis = image.copy()
    for cls, color in COLORS.items():
        m = label == int(cls)
        vis[m] = (0.45 * vis[m] + 0.55 * np.array(color)).astype(np.uint8)
    path = analyze_label(label, PhotoKind.WIDE).path
    t = max(2, round(max(image.shape[:2]) / 400))
    for e in path.edges:
        pts = np.array([[p.x, p.y] for p in e.polyline], np.int32)
        if len(pts) > 1:
            cv2.polylines(vis, [pts], False, (0, 255, 0), t)
    for n in path.nodes:
        cv2.circle(vis, (int(n.pt.x), int(n.pt.y)), 2 * t, (0, 0, 255), -1)
    for i, f in enumerate(result.findings, 1):
        if f.location is None:
            continue
        x, y = int(f.location.x), int(f.location.y)
        cv2.circle(vis, (x, y), 9 * t, VERDICT_COLOR.get(f.verdict.value, (0, 0, 0)), -1)
        cv2.putText(vis, str(i), (x - 4 * t, y + 4 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, (255, 255, 255), t)
    return vis


@app.get("/api/weights")
def weights() -> dict:
    names = sorted(p.name for p in WEIGHTS.glob("*.pt"))
    return {"block": [n for n in names if not n.startswith("scene")], "scene": [n for n in names if n.startswith("scene")],
            "default": {"block": DEFAULT_BLOCK, "scene": DEFAULT_SCENE}}


@app.post("/api/inspect")
async def inspect(photo: UploadFile = File(...), block: str = Form(DEFAULT_BLOCK), scene: str = Form(DEFAULT_SCENE),
                  max_side: int = Form(1400)) -> dict:
    data = np.frombuffer(await photo.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "이미지를 읽을 수 없음")
    s = max_side / max(image.shape[:2])
    if s < 1:
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rec = _Recorder(_segmenter(block, scene))
    t0 = time.perf_counter()
    result = run_photo(rec, image, "local-debug")
    ms = round((time.perf_counter() - t0) * 1000)
    label, _ = refine_mask(rec.label)  # run_photo 안에서 하는 후처리와 같음
    seg, rect = result.photo.segmentation, result.photo.rectification
    return {
        "ms": ms, "size": [image.shape[1], image.shape[0]], "models": [block, scene],
        "overall": result.overall.value, "scope": result.scope,
        "findings": [{"n": i, "code": f.code, "verdict": f.verdict.value, "message": f.message, "primary": f.primary,
                      "clause": f.clause, "suggestion": f.suggestion, "confidence": f.confidence}
                     for i, f in enumerate(result.findings, 1)],
        "areas": {MaskClass(int(k)).name.lower() if str(k).isdigit() else str(k): v for k, v in (seg.class_area_px or {}).items()},
        "mean_confidence": seg.mean_confidence, "unknown_block_area_px": seg.unknown_block_area_px,
        "rectification": None if rect is None else {"ok": rect.ok, "reason": rect.reason},
        "original": _png(image), "overlay": _png(_overlay(image, label, result)),
    }


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TactileCheck 로컬 디버그</title><style>
body{font-family:-apple-system,'Apple SD Gothic Neo',sans-serif;margin:0;background:#f3f4f6;color:#14181f}
header{padding:14px 20px;background:#fff;border-bottom:1px solid #e5e7eb;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0 12px 0 0}select,button{font-size:14px;padding:6px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff}
button{background:#14181f;color:#fff;border:0;cursor:pointer}#drop{flex:1;min-width:220px;border:2px dashed #cbd5e1;border-radius:10px;padding:8px 12px;color:#64748b;cursor:pointer}
main{display:grid;grid-template-columns:minmax(0,2fr) minmax(320px,1fr);gap:16px;padding:16px}@media(max-width:900px){main{grid-template-columns:1fr}}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:8px}.imgs img{width:100%;border-radius:8px;background:#fff}
.card{background:#fff;border-radius:12px;padding:14px 16px;margin-bottom:12px}.badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;font-weight:600}
.suspected{background:#fde8e8;color:#b42318}.hold{background:#eef0f3;color:#475467}.normal{background:#e6f4ea;color:#1e7b34}
.f{border-top:1px solid #eef0f3;padding:8px 0}.f:first-child{border:0}.muted{color:#667085;font-size:13px}code{font-size:12px}
.legend span{display:inline-block;margin-right:10px;font-size:12px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
</style></head><body><header><h1>TactileCheck 로컬 디버그</h1>
블록 <select id="block"></select> 장면 <select id="scene"></select>
<label id="drop">사진을 끌어다 놓거나 눌러서 선택<input id="file" type="file" accept="image/*" hidden></label>
<button id="go">다시 판정</button><span id="status" class="muted"></span></header>
<main><section><div class="imgs"><img id="orig"><img id="ov"></div>
<p class="legend muted"><span><i class="sw" style="background:#ffc800"></i>선형</span><span><i class="sw" style="background:#ff7800"></i>점형</span>
<span><i class="sw" style="background:#ff00ff"></i>횡단보도</span><span><i class="sw" style="background:#008cff"></i>차도</span>
<span><i class="sw" style="background:#00ff00"></i>중심선</span><span><i class="sw" style="background:#ff0000"></i>노드</span><span>번호 원 = 판정 위치</span></p></section>
<section id="out"></section></main>
<script>
const $=id=>document.getElementById(id);let file=null;
fetch('/api/weights').then(r=>r.json()).then(w=>{for(const[k,list]of[['block',w.block],['scene',w.scene]]){$(k).innerHTML=list.map(n=>`<option ${n===w.default[k]?'selected':''}>${n}</option>`).join('')}});
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function run(){if(!file)return;$('status').textContent='판정 중…';const fd=new FormData();fd.append('photo',file);fd.append('block',$('block').value);fd.append('scene',$('scene').value);
const r=await fetch('/api/inspect',{method:'POST',body:fd});if(!r.ok){$('status').textContent='실패: '+(await r.text());return}const d=await r.json();
$('orig').src=d.original;$('ov').src=d.overlay;$('status').textContent=`${d.ms}ms · ${d.size.join('×')} · ${d.models.join(' + ')}`;
$('out').innerHTML=`<div class="card"><b>종합</b> <span class="badge ${d.overall}">${d.overall}</span>
<p class="muted">scope: ${esc(JSON.stringify(d.scope))}</p></div>
<div class="card"><b>판정 (${d.findings.length})</b>${d.findings.map(f=>`<div class="f"><b>${f.n}. ${f.code}</b> <span class="badge ${f.verdict}">${f.verdict}</span>${f.primary?'':' <span class="muted">보조</span>'} <span class="muted">conf ${f.confidence}</span>
<div>${esc(f.message)}</div>${f.suggestion?`<div class="muted">개선: ${esc(f.suggestion)}</div>`:''}${f.clause?`<div class="muted">근거: ${esc(f.clause)}</div>`:''}</div>`).join('')}</div>
<div class="card"><b>분석 값</b><p class="muted">시점 보정: ${d.rectification?(d.rectification.ok?'성공':'실패 – '+esc(d.rectification.reason)):'없음'}<br>
블록 평균 신뢰도: ${d.mean_confidence==null?'-':d.mean_confidence.toFixed(3)} · 종류 모름 블록: ${d.unknown_block_area_px??'-'}px</p>
<code>${esc(JSON.stringify(d.areas))}</code></div>`}
$('file').onchange=e=>{file=e.target.files[0];$('drop').textContent=file.name;run()};
$('drop').ondragover=e=>e.preventDefault();$('drop').ondrop=e=>{e.preventDefault();file=e.dataTransfer.files[0];$('drop').textContent=file.name;run()};
$('go').onclick=run;$('block').onchange=run;$('scene').onchange=run;
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return PAGE


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8001")))
