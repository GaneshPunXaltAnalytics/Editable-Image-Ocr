import React, { useRef, useState, useEffect } from "react";
import ReactQuill from "react-quill";
import "react-quill/dist/quill.snow.css";

const quillModules = {
  toolbar: [
    [{ header: [1, 2, 3, false] }],
    ["bold", "italic", "underline"],
    [{ color: [] }, { background: [] }],
    [{ size: ["small", false, "large", "huge"] }],
    [{ 'font': ['monospace', 'serif'] }],
    [{ list: "ordered" }, { list: "bullet" }],
    [{ align: [] }],
  ],
};

// Use HTML for overlay only if it contains real text (avoids showing blank when Quill saved empty <p><br></p>).
function useHtmlForDisplay(html, plainText) {
  if (!html || typeof html !== "string") return false;
  const stripped = (html.replace(/<[^>]+>/g, "").trim() || "").replace(/\s+/g, " ").trim();
  return stripped.length > 0;
}

// Get the font actually applied in the editor: prefer any element with Quill font class (ql-font-serif, ql-font-monospace, etc.).
function getEffectiveFontFamily(editorRoot) {
  if (!editorRoot || typeof window.getComputedStyle !== "function") return "";
  function findElWithFontClass(el) {
    if (!el || el.nodeType !== 1) return null;
    const cls = el.className && typeof el.className === "string" ? el.className : "";
    if (/\bql-font-(?!sans-serif\b)\w+/.test(cls)) return el;
    for (let i = 0; i < el.childNodes.length; i++) {
      const r = findElWithFontClass(el.childNodes[i]);
      if (r) return r;
    }
    return null;
  }
  const fontEl = findElWithFontClass(editorRoot);
  if (fontEl) {
    const font = (window.getComputedStyle(fontEl).fontFamily || "").trim();
    if (font) return font;
  }
  function firstTextNode(el) {
    if (!el) return null;
    if (el.nodeType === 3 && (el.textContent || "").trim().length > 0) return el;
    for (let i = 0; i < el.childNodes.length; i++) {
      const r = firstTextNode(el.childNodes[i]);
      if (r) return r;
    }
    return null;
  }
  const textNode = firstTextNode(editorRoot);
  const parentEl = textNode && textNode.parentElement;
  if (parentEl) {
    const font = (window.getComputedStyle(parentEl).fontFamily || "").trim();
    if (font) return font;
  }
  return (window.getComputedStyle(editorRoot).fontFamily || "").trim();
}

// Uncontrolled ReactQuill: defaultValue only, read content on Done. Enables toolbar (color, size, alignment, etc.).
function CompositeTextEditor({ region, posLeft, posTop, compositeDisplaySize, compositeImageSize, boxFromMeasuredText, onDone, onDragStart }) {
  const quillRef = useRef(null);
  const initialContent = region.html ?? region.text ?? "";

  return (
    <div
      className="card"
      style={{
        position: "absolute",
        left: posLeft,
        top: posTop,
        width: 360,
        maxWidth: "90%",
        zIndex: 20,
        boxShadow: "0 4px 12px rgba(0,0,0,0.25)",
        marginTop: 0,
      }}
    >
      <h3
        style={{ marginTop: 0, cursor: "move", userSelect: "none" }}
        onMouseDown={onDragStart}
        title="Drag to move"
      >
        Edit text — Composite
      </h3>
      <div className="final-composite-editor" style={{ marginBottom: 8 }}>
        <ReactQuill
          ref={quillRef}
          theme="snow"
          defaultValue={initialContent}
          modules={quillModules}
          style={{ minHeight: 140 }}
        />
      </div>
      <button
        type="button"
        onClick={() => {
          let text = "";
          let html = "";
          let fontFamily = "";
          try {
            const editor = typeof quillRef.current?.getEditor === "function" ? quillRef.current.getEditor() : null;
            const rootEl = editor?.root ?? document.querySelector(".final-composite-editor .ql-editor");
            if (editor) {
              text = (editor.getText().replace(/\n+$/, "") ?? "").trim();
              html = editor.root?.innerHTML ?? "";
            } else if (rootEl) {
              text = (rootEl.innerText ?? rootEl.textContent ?? "").replace(/\n+$/, "").trim();
              html = rootEl.innerHTML ?? "";
            }
            if (rootEl) fontFamily = getEffectiveFontFamily(rootEl);
          } catch (_) {
            const el = document.querySelector(".final-composite-editor .ql-editor");
            if (el) {
              text = (el.innerText ?? el.textContent ?? "").replace(/\n+$/, "").trim();
              html = el.innerHTML ?? "";
              fontFamily = getEffectiveFontFamily(el);
            }
          }
          const stripped = (html.replace(/<[^>]+>/g, "").trim() || "").replace(/\s+/g, " ").trim();
          if (stripped.length === 0) html = "";
          onDone({ text, html, fontFamily });
        }}
      >
        Done editing
      </button>
    </div>
  );
}

export default function App() {
  const [thumbs, setThumbs] = useState([]);
  const [cropped, setCropped] = useState([]);
  const [processed, setProcessed] = useState([]); // images returned from backend
  const [processing, setProcessing] = useState(false);
  const [lastCrop, setLastCrop] = useState(null); // { nx, ny, nWidth, nHeight, dataUrl }
  const [finalImage, setFinalImage] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fullFinal, setFullFinal] = useState(null); // full-image inpainted from /process_roi
  const [fullFinalImageSize, setFullFinalImageSize] = useState(null); // { width, height } natural size
  const [textRegions, setTextRegions] = useState([]); // [{ id, text, score, box }] box = [[x,y],...] in image coords
  const [fullFinalDisplaySize, setFullFinalDisplaySize] = useState(null); // { width, height } displayed pixels
  const [cropTextRegions, setCropTextRegions] = useState([]); // from /process, in crop coords [{ id, text, score, box }]
  const [cropSize, setCropSize] = useState(null); // { width, height } of crop from /process
  const [compositeTextRegions, setCompositeTextRegions] = useState([]); // in full image coords, set when Apply to Original
  const [compositeImageSize, setCompositeImageSize] = useState(null); // full composite dimensions
  const [compositeDisplaySize, setCompositeDisplaySize] = useState(null); // displayed size of composite img
  const [polygonMode, setPolygonMode] = useState(true); // Default to polygon mode
  const [polygonPoints, setPolygonPoints] = useState([]); // array of {dx,dy,nx,ny}
  const [polygons, setPolygons] = useState([]); // saved polygons (multiple)
  const [polygonColors, setPolygonColors] = useState({}); // { polygonId: { color, background_color } }
  const CLOSE_POLYGON_THRESHOLD_PX = 18; // click within this many px of first point to close polygon
  const [maskDataUrl, setMaskDataUrl] = useState(null);
  const [selected, setSelected] = useState(null); // url
  const [selectedFullFinalTextId, setSelectedFullFinalTextId] = useState(null); // show border only when this region is selected
  const [selectedCompositeTextId, setSelectedCompositeTextId] = useState(null);
  const [compositeEditorPosition, setCompositeEditorPosition] = useState(null); // { left, top } when user has dragged the panel
  const [openAiMeta, setOpenAiMeta] = useState(null); // { enabled, model, cost_usd, error }

  const rightImgRef = useRef(null);
  const compositeEditorDragRef = useRef(null); // { clientX, clientY, startLeft, startTop } while dragging
  const overlayRef = useRef(null);
  const selectionRef = useRef(null);
  const drawingRef = useRef(false);
  const startRef = useRef({ x: 0, y: 0 });
  const fullFinalImgRef = useRef(null);
  const textDragRef = useRef({ regionId: null, startX: 0, startY: 0, startBox: null });
  const finalCompositeImgRef = useRef(null);
  const compositeDragRef = useRef({ regionId: null, startX: 0, startY: 0, startBox: null });
  const fullFinalQuillRef = useRef(null); // ReactQuill for Full Final editor (uncontrolled)

  useEffect(() => {
    return () => {
      // revoke object URLs on unmount
      thumbs.forEach(t => URL.revokeObjectURL(t.url));
      cropped.forEach(c => URL.revokeObjectURL(c));
    };
  }, []); // eslint-disable-line

  // Track displayed size of full final image for text overlay scaling
  useEffect(() => {
    if (!fullFinal || !fullFinalImgRef.current) return;
    const img = fullFinalImgRef.current;
    const updateSize = () => {
      if (img && img.getBoundingClientRect) {
        const rect = img.getBoundingClientRect();
        setFullFinalDisplaySize({ width: rect.width, height: rect.height });
      }
    };
    updateSize();
    const ro = new ResizeObserver(updateSize);
    ro.observe(img);
    return () => ro.disconnect();
  }, [fullFinal, textRegions]);

  // Drag text regions: window mousemove/mouseup
  useEffect(() => {
    const onMove = (e) => {
      const drag = textDragRef.current;
      if (!drag || !drag.regionId || !fullFinalImageSize || !fullFinalDisplaySize) return;
      const scaleX = fullFinalImageSize.width / fullFinalDisplaySize.width;
      const scaleY = fullFinalImageSize.height / fullFinalDisplaySize.height;
      const deltaNatX = (e.clientX - drag.startX) * scaleX;
      const deltaNatY = (e.clientY - drag.startY) * scaleY;
      setTextRegions((prev) =>
        prev.map((r) =>
          r.id === drag.regionId
            ? { ...r, box: drag.startBox.map(([x, y]) => [x + deltaNatX, y + deltaNatY]) }
            : r
        )
      );
    };
    const onUp = () => {
      textDragRef.current = { regionId: null, startX: 0, startY: 0, startBox: null };
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [fullFinalImageSize, fullFinalDisplaySize]);

  // Track displayed size of final composite image for text overlay
  useEffect(() => {
    if (!finalImage || !finalCompositeImgRef.current) return;
    const img = finalCompositeImgRef.current;
    const updateSize = () => {
      if (img && img.getBoundingClientRect) {
        const rect = img.getBoundingClientRect();
        setCompositeDisplaySize({ width: rect.width, height: rect.height });
      }
    };
    updateSize();
    const ro = new ResizeObserver(updateSize);
    ro.observe(img);
    return () => ro.disconnect();
  }, [finalImage, compositeTextRegions]);

  // Drag the Composite edit panel (mousemove/mouseup)
  useEffect(() => {
    const onMove = (e) => {
      const ref = compositeEditorDragRef.current;
      if (!ref) return;
      setCompositeEditorPosition({
        left: ref.startLeft + (e.clientX - ref.clientX),
        top: ref.startTop + (e.clientY - ref.clientY),
      });
    };
    const onUp = () => {
      compositeEditorDragRef.current = null;
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  // Drag composite text regions
  useEffect(() => {
    const onMove = (e) => {
      const drag = compositeDragRef.current;
      if (!drag || !drag.regionId || !compositeImageSize || !compositeDisplaySize) return;
      const scaleX = compositeImageSize.width / compositeDisplaySize.width;
      const scaleY = compositeImageSize.height / compositeDisplaySize.height;
      const deltaNatX = (e.clientX - drag.startX) * scaleX;
      const deltaNatY = (e.clientY - drag.startY) * scaleY;
      setCompositeTextRegions((prev) =>
        prev.map((r) =>
          r.id === drag.regionId
            ? { ...r, box: drag.startBox.map(([x, y]) => [x + deltaNatX, y + deltaNatY]) }
            : r
        )
      );
    };
    const onUp = () => {
      compositeDragRef.current = { regionId: null, startX: 0, startY: 0, startBox: null };
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [compositeImageSize, compositeDisplaySize]);

  // Undo functionality: Ctrl+Z removes last polygon point
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Check for Ctrl+Z (or Cmd+Z on Mac)
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey && !e.altKey) {
        // Prevent default browser undo behavior
        e.preventDefault();
        
        // Only undo if we're in polygon mode and have points
        if (polygonMode && polygonPoints.length > 0) {
          setPolygonPoints(prev => prev.slice(0, -1));
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [polygonMode, polygonPoints.length]);

  function handleFiles(e) {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    console.log('handleFiles', files.map(f => f.name));
    const newThumbs = files.map(f => ({
      id: cryptoRandomId(),
      url: URL.createObjectURL(f),
      name: f.name,
      file: f
    }));
    setThumbs(prev => [...prev, ...newThumbs]);
    e.target.value = "";
  }

  function cryptoRandomId() {
    return Math.random().toString(36).slice(2, 9);
  }

  function getRelativePos(clientX, clientY, imgEl) {
    const rect = imgEl.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(rect.width, clientX - rect.left)),
      y: Math.max(0, Math.min(rect.height, clientY - rect.top)),
      rect
    };
  }

  // helper math utilities for polygon warp
  function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function orderQuadPoints(pts) {
    // pts: [{x,y}] in natural coordinate space
    // sort by y, then split top/bottom, then sort by x
    const sorted = [...pts].sort((a, b) => a.y - b.y);
    const top = sorted.slice(0, 2).sort((a, b) => a.x - b.x);
    const bottom = sorted.slice(2, 4).sort((a, b) => a.x - b.x);
    return [top[0], top[1], bottom[1], bottom[0]]; // TL, TR, BR, BL
  }

  function invert3x3(m) {
    // m is 3x3 array
    const a = m[0][0], b = m[0][1], c = m[0][2];
    const d = m[1][0], e = m[1][1], f = m[1][2];
    const g = m[2][0], h = m[2][1], i = m[2][2];
    const A = e * i - f * h;
    const B = c * h - b * i;
    const C = b * f - c * e;
    const D = f * g - d * i;
    const E = a * i - c * g;
    const F = c * d - a * f;
    const G = d * h - e * g;
    const H = b * g - a * h;
    const I = a * e - b * d;
    const det = a * A + b * D + c * G;
    if (Math.abs(det) < 1e-8) return null;
    const invDet = 1 / det;
    return [
      [A * invDet, B * invDet, C * invDet],
      [D * invDet, E * invDet, F * invDet],
      [G * invDet, H * invDet, I * invDet],
    ];
  }

  function multiplyMatVec(m, v) {
    return [
      m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
      m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
      m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ];
  }

  // Map crop-space point (cx, cy) to full-image coords using lastCrop (rect or polygon)
  function mapCropPointToFull(cx, cy, lastCrop) {
    if (!lastCrop) return { x: cx, y: cy };
    if (lastCrop.polygon) {
      const { nWidth: nW, nHeight: nH, polygon } = lastCrop;
      if (nW <= 0 || nH <= 0) return { x: cx, y: cy };
      const s = cx / nW;
      const t = cy / nH;
      const p0 = polygon[0], p1 = polygon[1], p2 = polygon[2], p3 = polygon[3];
      return {
        x: (1 - s) * (1 - t) * p0.x + s * (1 - t) * p1.x + (1 - s) * t * p3.x + s * t * p2.x,
        y: (1 - s) * (1 - t) * p0.y + s * (1 - t) * p1.y + (1 - s) * t * p3.y + s * t * p2.y,
      };
    }
    return { x: cx + lastCrop.nx, y: cy + lastCrop.ny };
  }

  function computeAffine(srcTri, dstTri) {
    // srcTri/dstTri: [{x,y}, {x,y}, {x,y}]
    const X = [
      [srcTri[0].x, srcTri[0].y, 1],
      [srcTri[1].x, srcTri[1].y, 1],
      [srcTri[2].x, srcTri[2].y, 1],
    ];
    const invX = invert3x3(X);
    if (!invX) throw new Error("Singular matrix in affine computation");
    const U = [dstTri[0].x, dstTri[1].x, dstTri[2].x];
    const V = [dstTri[0].y, dstTri[1].y, dstTri[2].y];
    const paramsU = multiplyMatVec(invX, U); // [a, c, e]
    const paramsV = multiplyMatVec(invX, V); // [b, d, f]
    return {
      a: paramsU[0],
      c: paramsU[1],
      e: paramsU[2],
      b: paramsV[0],
      d: paramsV[1],
      f: paramsV[2],
    };
  }

  function drawTriangleToCtx(ctx, srcCanvas, srcTri, dstTri) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(dstTri[0].x, dstTri[0].y);
    ctx.lineTo(dstTri[1].x, dstTri[1].y);
    ctx.lineTo(dstTri[2].x, dstTri[2].y);
    ctx.closePath();
    ctx.clip();
    const m = computeAffine(srcTri, dstTri);
    // setTransform(a, b, c, d, e, f) maps x' = a*x + c*y + e; y' = b*x + d*y + f
    ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
    ctx.drawImage(srcCanvas, 0, 0);
    ctx.restore();
    // reset transform to identity explicitly
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  async function createRectifiedCrop(quadPoints) {
    // quadPoints: [{nx,ny}] in natural pixels, any order
    if (!rightImgRef.current) throw new Error("No image");
    const ordered = orderQuadPoints(quadPoints);
    const p0 = ordered[0], p1 = ordered[1], p2 = ordered[2], p3 = ordered[3];
    const w = Math.max(dist(p0, p1), dist(p2, p3)) | 0;
    const h = Math.max(dist(p1, p2), dist(p3, p0)) | 0;
    if (w <= 0 || h <= 0) throw new Error("Invalid polygon size");

    // source canvas (original full-size)
    const origEl = rightImgRef.current;
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = origEl.naturalWidth;
    srcCanvas.height = origEl.naturalHeight;
    const sctx = srcCanvas.getContext("2d");
    sctx.drawImage(origEl, 0, 0, srcCanvas.width, srcCanvas.height);

    // destination canvas (rectified)
    const dest = document.createElement("canvas");
    dest.width = w;
    dest.height = h;
    const dctx = dest.getContext("2d");

    // triangle 1: p0,p1,p2 -> (0,0),(w,0),(w,h)
    drawTriangleToCtx(
      dctx,
      srcCanvas,
      [p0, p1, p2],
      [{ x: 0, y: 0 }, { x: w, y: 0 }, { x: w, y: h }]
    );

    // triangle 2: p0,p2,p3 -> (0,0),(w,h),(0,h)
    drawTriangleToCtx(
      dctx,
      srcCanvas,
      [p0, p2, p3],
      [{ x: 0, y: 0 }, { x: w, y: h }, { x: 0, y: h }]
    );

    return { dataUrl: dest.toDataURL("image/png"), width: w, height: h, orderedQuad: ordered };
  }

  function generateMaskFromPoints(points) {
    // points: array of {nx,ny} (natural coords)
    if (!rightImgRef.current) throw new Error("No image loaded");
    const origW = rightImgRef.current.naturalWidth;
    const origH = rightImgRef.current.naturalHeight;
    const c = document.createElement("canvas");
    c.width = origW;
    c.height = origH;
    const ctx = c.getContext("2d");
    // black background
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, origW, origH);
    // white polygon
    ctx.beginPath();
    ctx.moveTo(points[0].nx, points[0].ny);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].nx, points[i].ny);
    }
    ctx.closePath();
    ctx.fillStyle = "white";
    ctx.fill();
    return c.toDataURL("image/png");
  }

  function generateMaskFromPolygons(polys) {
    // polys: array of polygons, each polygon is array of {nx,ny}
    if (!rightImgRef.current) throw new Error("No image loaded");
    const origW = rightImgRef.current.naturalWidth;
    const origH = rightImgRef.current.naturalHeight;
    const c = document.createElement("canvas");
    c.width = origW;
    c.height = origH;
    const ctx = c.getContext("2d");
    // black background
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, origW, origH);
    // draw each polygon filled white
    ctx.fillStyle = "white";
    for (const pts of polys) {
      if (!pts || pts.length === 0) continue;
      ctx.beginPath();
      ctx.moveTo(pts[0].nx, pts[0].ny);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].nx, pts[i].ny);
      ctx.closePath();
      ctx.fill();
    }
    return c.toDataURL("image/png");
  }

  function downloadDataUrl(dataUrl, filename) {
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // Measure text in display pixels (same font as overlay) for dynamic box resize
  function measureTextDisplay(text, fontSizePx) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    ctx.font = `${fontSizePx}px sans-serif`;
    const width = ctx.measureText(text || "").width;
    const height = fontSizePx * 1.2;
    return { width, height };
  }

  // Compute new box (image coords): keep width and top-left fixed; expand height only (never shrink)
  function boxFromMeasuredText(region, newText, displaySize, imageSize, fontSizeDisplay) {
    const box = region.box;
    const minX = Math.min(...box.map((p) => p[0]));
    const maxX = Math.max(...box.map((p) => p[0]));
    const minY = Math.min(...box.map((p) => p[1]));
    const maxY = Math.max(...box.map((p) => p[1]));
    const currentHImg = maxY - minY; // current height in image coords
    const measured = measureTextDisplay(newText, fontSizeDisplay);
    const paddingY = 4;
    const hDisp = measured.height + paddingY;
    const hImgNeeded = hDisp * (imageSize.height / displaySize.height);
    const hImg = Math.max(currentHImg, hImgNeeded); // only expand, never shrink
    return [[minX, minY], [maxX, minY], [maxX, minY + hImg], [minX, minY + hImg]];
  }

  // Parse HTML into lines of { text, color } for canvas drawing. Uses a temp div + getComputedStyle.
  // function parseHtmlToColoredLines(html) {
  //   if (!html || typeof html !== "string") return null;
  //   const stripped = html.replace(/<[^>]+>/g, "").trim().replace(/\s+/g, " ").trim();
  //   if (stripped.length === 0) return null;
  //   const div = document.createElement("div");
  //   div.innerHTML = html;
  //   div.style.position = "absolute";
  //   div.style.left = "-9999px";
  //   div.style.visibility = "hidden";
  //   document.body.appendChild(div);
  //   const defaultColor = "#111827";
  //   function getColor(el) {
  //     if (!el || el.nodeType !== 1) return defaultColor;
  //     const style = window.getComputedStyle(el);
  //     const c = style.color;
  //     if (c && c !== "rgba(0, 0, 0, 0)" && c !== "transparent") return c;
  //     return getColor(el.parentElement);
  //   }
  //   const lines = [];
  //   let currentLine = [];
  //   function visit(node) {
  //     if (node.nodeType === 3) {
  //       const text = node.textContent || "";
  //       if (text.length > 0) {
  //         const color = getColor(node.parentElement);
  //         currentLine.push({ text, color });
  //       }
  //       return;
  //     }
  //     if (node.nodeType !== 1) return;
  //     const tag = node.tagName.toLowerCase();
  //     if (tag === "br" || tag === "p" || tag === "div") {
  //       if (currentLine.length > 0) {
  //         lines.push(currentLine);
  //         currentLine = [];
  //       }
  //       if (tag === "br") return;
  //     }
  //     for (let i = 0; i < node.childNodes.length; i++) visit(node.childNodes[i]);
  //     if (tag === "p" || tag === "div") {
  //       if (currentLine.length > 0) {
  //         lines.push(currentLine);
  //         currentLine = [];
  //       }
  //     }
  //   }
  //   visit(div);
  //   if (currentLine.length > 0) lines.push(currentLine);
  //   document.body.removeChild(div);
  //   return lines.length > 0 ? lines : null;
  // }
  function parseHtmlToColoredLines(html) {
    if (!html || typeof html !== 'string') return null;
    const stripped = html.replace(/<[^>]*>/g, '').trim().replace(/\s+/g, ' ').trim();
    if (stripped.length === 0) return null;

    const div = document.createElement('div');
    div.innerHTML = html;
    div.style.position = 'absolute';
    div.style.left = '-9999px';
    div.style.visibility = 'hidden';
    document.body.appendChild(div);

    const defaultColor = '#111827';

    function getStyle(el) {
      if (!el || el.nodeType !== 1) return { color: defaultColor, fontFamily: 'sans-serif', fontSize: '16px' };
      const style = window.getComputedStyle(el);
      const c = style.color;
      return {
        color: (c && c !== 'rgba(0, 0, 0, 0)' && c !== 'transparent') ? c : defaultColor,
        fontFamily: style.fontFamily || 'sans-serif',
        fontSize: style.fontSize || '16px'
      };
    }

    const lines = [];
    let currentLine = [];

    function visit(node) {
      if (node.nodeType === 3) {
        const text = node.textContent;
        if (text.length > 0) {
          const style = getStyle(node.parentElement);
          currentLine.push({ text, color: style.color, fontFamily: style.fontFamily, fontSize: style.fontSize });
        }
        return;
      }
      if (node.nodeType !== 1) return;

      const tag = node.tagName.toLowerCase();
      if (tag === 'br' || tag === 'p' || tag === 'div') {
        if (currentLine.length > 0) lines.push(currentLine);
        currentLine = [];
        if (tag === 'br') return;
      }

      for (let i = 0; i < node.childNodes.length; i++) {
        visit(node.childNodes[i]);
      }

      if (tag === 'p' || tag === 'div') {
        if (currentLine.length > 0) lines.push(currentLine);
        currentLine = [];
      }
    }

    visit(div);
    if (currentLine.length > 0) lines.push(currentLine);
    document.body.removeChild(div);

    return lines.length === 0 ? null : lines;
  }

  // Export image with text regions rendered onto it (for download edited). Preserves text color when region.html is set.
  async function exportImageWithText(imageDataUrl, imageSize, regions) {
    if (!imageDataUrl) return imageDataUrl;
    const img = await new Promise((res, rej) => {
      const i = new Image();
      i.crossOrigin = "anonymous";
      i.onload = () => res(i);
      i.onerror = rej;
      i.src = imageDataUrl;
    });
    const w = (imageSize && imageSize.width) ? imageSize.width : img.naturalWidth;
    const h = (imageSize && imageSize.height) ? imageSize.height : img.naturalHeight;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, w, h);
    if (!regions || regions.length === 0) return canvas.toDataURL("image/png");
    ctx.textBaseline = "top";
    const defaultColor = "#111827";

    // Build canvas font string: quote family only if it contains comma (e.g. "Monaco, Menlo, monospace").
    function toCanvasFont(sizePx, family) {
      const f = (family || "sans-serif").trim();
      // If the family string already contains quotes or is a font stack, don't wrap it in extra quotes.
      // Most browsers return computed fontFamily with quotes around names with spaces.
      return `${sizePx}px ${f}`;
    }

    for (const region of regions) {
      const box = region.box;
      if (!box) continue;
      const minX = Math.min(...box.map(p => p[0]));
      const minY = Math.min(...box.map(p => p[1]));
      const maxY = Math.max(...box.map(p => p[1]));
      const fontSize = Math.max(10, (maxY - minY) * 0.72);
      const lineHeight = fontSize * 1.2;
      const angleDeg = Number(region.angle_deg || 0);
      const angleRad = (angleDeg * Math.PI) / 180;

      const regionFontFamily = (region.fontFamily && region.fontFamily.trim()) ? region.fontFamily.trim() : "sans-serif";

      const coloredLines = parseHtmlToColoredLines(region.html);
      ctx.textBaseline = "top";
      ctx.save();
      ctx.translate(minX, minY);
      ctx.rotate(angleRad);

      if (coloredLines && coloredLines.length > 0) {
        let y = 0;
        for (const line of coloredLines) {
          let x = 0;
          for (const seg of line) {
            // Use detected text color if available, otherwise use HTML color or default
            ctx.fillStyle = region.color || seg.color || defaultColor;

            // Logic: 
            // 1. If the segment has a specific font (e.g. from ql-font- class), use it.
            // 2. Otherwise if the region has a specific font (applied to the whole region), use it.
            // 3. Fallback to generic sans-serif.
            let family = "sans-serif";
            if (seg.fontFamily && seg.fontFamily.trim() && !seg.fontFamily.includes("sans-serif")) {
              family = seg.fontFamily.trim();
            } else if (regionFontFamily && !regionFontFamily.includes("sans-serif")) {
              family = regionFontFamily;
            } else if (seg.fontFamily) {
              family = seg.fontFamily.trim();
            }

            ctx.font = toCanvasFont(fontSize, family);
            ctx.fillText(seg.text, x, y);
            x += ctx.measureText(seg.text).width;
          }
          y += lineHeight;
        }
      } else {
        // Use detected text color if available
        ctx.fillStyle = region.color || defaultColor;
        ctx.font = toCanvasFont(fontSize, regionFontFamily);
        ctx.fillText(region.text || "", 0, 0);
      }
      ctx.restore();
    }
    return canvas.toDataURL("image/png");
  }

  function onMouseDown(e) {
    // if polygon mode active, don't start rectangular selection
    if (polygonMode) return;
    if (!rightImgRef.current) return;
    const pos = getRelativePos(e.clientX, e.clientY, rightImgRef.current);
    drawingRef.current = true;
    // store start coordinates relative to the image (not the overlay)
    startRef.current = { x: pos.x, y: pos.y };
    if (!selectionRef.current) {
      selectionRef.current = document.createElement("div");
      selectionRef.current.className = "selection";
      overlayRef.current.appendChild(selectionRef.current);
    }
    // The selection element is positioned inside the overlay (which covers the whole container),
    // but startRef is relative to the image. Compute the image offset inside the overlay so
    // the selection lines up with the image where the user clicked.
    const overlayRect = overlayRef.current.getBoundingClientRect();
    const imgRect = rightImgRef.current.getBoundingClientRect();
    const imgOffsetX = imgRect.left - overlayRect.left;
    const imgOffsetY = imgRect.top - overlayRect.top;
    selectionRef.current.style.left = imgOffsetX + startRef.current.x + "px";
    selectionRef.current.style.top = imgOffsetY + startRef.current.y + "px";
    selectionRef.current.style.width = "0px";
    selectionRef.current.style.height = "0px";
  }

  function onMouseMove(e) {
    if (!drawingRef.current || !selectionRef.current) return;
    const pos = getRelativePos(e.clientX, e.clientY, rightImgRef.current);
    const x = Math.min(startRef.current.x, pos.x);
    const y = Math.min(startRef.current.y, pos.y);
    const w = Math.abs(pos.x - startRef.current.x);
    const h = Math.abs(pos.y - startRef.current.y);
    // selection is inside overlay, so add image offset within overlay to position correctly
    const overlayRect = overlayRef.current.getBoundingClientRect();
    const imgRect = rightImgRef.current.getBoundingClientRect();
    const imgOffsetX = imgRect.left - overlayRect.left;
    const imgOffsetY = imgRect.top - overlayRect.top;
    selectionRef.current.style.left = imgOffsetX + x + "px";
    selectionRef.current.style.top = imgOffsetY + y + "px";
    selectionRef.current.style.width = w + "px";
    selectionRef.current.style.height = h + "px";
  }

  function onMouseUp() {
    drawingRef.current = false;
  }

  function clearSelection() {
    if (selectionRef.current) {
      selectionRef.current.remove();
      selectionRef.current = null;
    }
    // clear polygon points too
    setPolygonPoints([]);
    setFinalImage(null);
    setMaskDataUrl(null);
  }

  // Fetch color for a single polygon when it's closed
  async function fetchPolygonColor(polygonId, points) {
    if (!selectedFile || !points || points.length < 3) return;
    try {
      const polygonForServer = points.map(p => ({ x: Math.round(p.nx), y: Math.round(p.ny) }));
      const form = new FormData();
      form.append("file", selectedFile, selectedFile.name || "original.png");
      form.append("polygons", JSON.stringify([polygonForServer]));
      const res = await fetch("http://127.0.0.1:8000/process_roi", {
        method: "POST",
        body: form
      });
      if (!res.ok) return;
      const json = await res.json().catch(() => null);
      const dominant = json?.roi_dominant_text_color || null;
      if (dominant) {
        setPolygonColors(prev => ({
          ...prev,
          [polygonId]: {
            color: dominant,
            background_color: null,
          }
        }));
      }
    } catch (err) {
      console.error("Failed to fetch polygon color:", err);
    }
  }

  function cropSelection() {
    if (!selectionRef.current || !rightImgRef.current || !rightImgRef.current.complete) {
      alert("Make a selection first.");
      return;
    }
    const imgRect = rightImgRef.current.getBoundingClientRect();
    const selRect = selectionRef.current.getBoundingClientRect();
    const sx = selRect.left - imgRect.left;
    const sy = selRect.top - imgRect.top;
    const sw = selRect.width;
    const sh = selRect.height;
    const scaleX = rightImgRef.current.naturalWidth / imgRect.width;
    const scaleY = rightImgRef.current.naturalHeight / imgRect.height;
    const nx = Math.round(sx * scaleX);
    const ny = Math.round(sy * scaleY);
    const nWidth = Math.round(sw * scaleX);
    const nHeight = Math.round(sh * scaleY);
    if (nWidth <= 0 || nHeight <= 0) {
      alert("Invalid selection.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = nWidth;
    canvas.height = nHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(rightImgRef.current, nx, ny, nWidth, nHeight, 0, 0, nWidth, nHeight);
    const dataUrl = canvas.toDataURL("image/png");
    setCropped(prev => [...prev, dataUrl]);
    // remember crop coordinates & data for client-side compositing
    setLastCrop({ nx, ny, nWidth, nHeight, dataUrl });

    // send the cropped image to backend
    (async function sendToBackend() {
      try {
        setProcessing(true);
        // convert dataURL to blob
        function dataURLToBlob(dataURL) {
          const parts = dataURL.split(',');
          const meta = parts[0].match(/:(.*?);/);
          const contentType = meta ? meta[1] : 'image/png';
          const byteString = atob(parts[1]);
          const ia = new Uint8Array(byteString.length);
          for (let i = 0; i < byteString.length; i++) ia[i] = byteString.charCodeAt(i);
          return new Blob([ia], { type: contentType });
        }

        const blob = dataURLToBlob(dataUrl);
        const form = new FormData();
        form.append("file", blob, "crop.png");

        const res = await fetch("http://127.0.0.1:8000/process", {
          method: "POST",
          body: form
        });

        if (!res.ok) {
          const text = await res.text();
          throw new Error(`Server error ${res.status}: ${text}`);
        }

        // FastAPI returns JSON with base64 strings:
        // { original, annotated, mask, erased } (each may be null)
        const json = await res.json().catch(() => null);
        console.log("SERVER RESPONSE: ", json);

        if (json) {
          const result = {
            annotated: json.annotated ? "data:image/png;base64," + json.annotated : null,
            mask: json.mask ? "data:image/png;base64," + json.mask : null,
            erased: json.erased ? "data:image/png;base64," + json.erased : null,
            original: json.original ? "data:image/png;base64," + json.original : null,
          };
          if (result.annotated || result.mask || result.erased || result.original) {
            setProcessed(result);
          } else {
            console.warn("No image fields found in /process response", json);
            alert("Server returned no processed images. Check server logs.");
          }
          const regions = (json.text_regions || []).map((r) => ({
            id: cryptoRandomId(),
            text: r.text != null ? String(r.text) : "",
            score: r.score != null ? Number(r.score) : 1,
            box: Array.isArray(r.box) ? r.box.map((p) => [Number(p[0]), Number(p[1])]) : [[0, 0], [0, 0], [0, 0], [0, 0]],
          }));
          setCropTextRegions(regions);
          if (json.crop_width != null && json.crop_height != null) {
            setCropSize({ width: json.crop_width, height: json.crop_height });
          }
        } else {
          console.warn("Empty or invalid JSON from /process");
          alert("Received invalid response from server.");
        }
      } catch (err) {
        console.error("Upload/processing failed", err);
        alert("Failed to process image: " + err.message);
      } finally {
        setProcessing(false);
      }
    })();
  }

  return (
    <div className="app">
      <div className="container">
        <div className="left card">
          <h2>Upload</h2>
          <input type="file" multiple accept="image/*" onChange={handleFiles} />
          <div className="preview thumbs" aria-live="polite">
            {thumbs.map(t => (
              <img
                key={t.id}
                src={t.url}
                alt={t.name}
                className="thumb"
                onClick={() => {
                  setSelected(t.url);
                  setSelectedFile(t.file || null);
                  clearSelection();
                }}
              />
            ))}
            {thumbs.length === 0 && <div className="placeholder">No images yet</div>}
          </div>
        </div>

        <div className="right card">
          <h2>Crop Area</h2>
          <div
            className="image-container"
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
          >
            {selected ? (
              <>
                <img ref={rightImgRef} src={selected} alt="Selected" draggable={false} />
                <div
                  ref={overlayRef}
                  className="overlay"
                  onClick={(e) => {
                    // add polygon point when in polygon mode; close when click near first point
                    if (!polygonMode) return;
                    if (!rightImgRef.current) return;
                    const pos = getRelativePos(e.clientX, e.clientY, rightImgRef.current);
                    const imgRect = rightImgRef.current.getBoundingClientRect();
                    const scaleX = rightImgRef.current.naturalWidth / imgRect.width;
                    const scaleY = rightImgRef.current.naturalHeight / imgRect.height;
                    const nx = pos.x * scaleX;
                    const ny = pos.y * scaleY;
                    const overlayRect = overlayRef.current.getBoundingClientRect();
                    const imgOffsetX = imgRect.left - overlayRect.left;
                    const imgOffsetY = imgRect.top - overlayRect.top;
                    const dx = imgOffsetX + pos.x;
                    const dy = imgOffsetY + pos.y;
                    setPolygonPoints(prev => {
                      const first = prev[0];
                      const nearFirst = first && prev.length >= 3 &&
                        Math.hypot(dx - first.dx, dy - first.dy) <= CLOSE_POLYGON_THRESHOLD_PX;
                      if (nearFirst) {
                        const newPolyId = cryptoRandomId();
                        setPolygons(ps => [...ps, { id: newPolyId, points: prev }]);
                        // Fetch color for the new polygon
                        fetchPolygonColor(newPolyId, prev);
                        return [];
                      }
                      return [...prev, { dx, dy, nx, ny }];
                    });
                  }}
                >
                  {/* existing saved polygons */}
                  <svg className="poly-svg" width="100%" height="100%" viewBox={`0 0 ${overlayRef.current ? overlayRef.current.clientWidth : 0} ${overlayRef.current ? overlayRef.current.clientHeight : 0}`} preserveAspectRatio="none">
                    {polygons.map((poly, pi) => {
                      const polyColor = polygonColors[poly.id];
                      // Calculate center point for color indicator
                      const centerX = poly.points.reduce((sum, p) => sum + p.dx, 0) / poly.points.length;
                      const centerY = poly.points.reduce((sum, p) => sum + p.dy, 0) / poly.points.length;
                      return (
                        <g key={poly.id}>
                          <polyline
                            points={poly.points.map(p => `${p.dx},${p.dy}`).concat(`${poly.points[0].dx},${poly.points[0].dy}`).join(" ")}
                            fill="none"
                            stroke="rgba(16,185,129,0.6)"
                            strokeWidth="2"
                          />
                          {/* Color point indicator - show by default if color is available */}
                          {polyColor && polyColor.color && (
                            <circle
                              cx={centerX}
                              cy={centerY}
                              r="8"
                              fill={polyColor.color}
                              stroke="#fff"
                              strokeWidth="2"
                              style={{ cursor: "pointer" }}
                              title={`Text color: ${polyColor.color}`}
                            />
                          )}
                        </g>
                      );
                    })}
                    {/* currently drawing polygon: show segment to first point when 3+ points (click there to close) */}
                    {polygonPoints.length > 0 && (
                      <>
                        <polyline
                          points={polygonPoints.length >= 3
                            ? polygonPoints.map(p => `${p.dx},${p.dy}`).concat(`${polygonPoints[0].dx},${polygonPoints[0].dy}`).join(" ")
                            : polygonPoints.map(p => `${p.dx},${p.dy}`).join(" ")}
                          fill="none"
                          stroke="rgba(37,99,235,0.6)"
                          strokeWidth="2"
                          strokeDasharray="6,4"
                        />
                        {/* Show color point on first click */}
                        {polygonPoints.length >= 1 && (
                          <circle
                            cx={polygonPoints[0].dx}
                            cy={polygonPoints[0].dy}
                            r="6"
                            fill="rgba(37,99,235,0.8)"
                            stroke="#fff"
                            strokeWidth="1.5"
                            style={{ cursor: "pointer" }}
                            title="First point - color will be detected when polygon is closed"
                          />
                        )}
                      </>
                    )}
                  </svg>
                </div>
              </>
            ) : (
              <div className="placeholder">Click a thumbnail to load image</div>
            )}
          </div>
          <div className="controls">
            <button
              onClick={() => {
                setPolygonMode(p => !p);
                setPolygonPoints([]);
              }}
              style={{ background: polygonMode ? "#1d4ed8" : undefined }}
            >
              {polygonMode ? "Polygon: ON" : "Polygon: OFF"}
            </button>
            <button
              onClick={() => {
                // collect polygons: saved ones + current if 3+ points (treat as closed for mask)
                const allPolys = [...polygons.map(p => p.points)];
                if (polygonPoints.length >= 3) allPolys.push(polygonPoints);
                if (allPolys.length === 0) {
                  alert("Draw at least one polygon (min 3 points; click near the first point to close) to generate a mask.");
                  return;
                }
                try {
                  const dataUrl = generateMaskFromPolygons(allPolys);
                  setMaskDataUrl(dataUrl);
                  downloadDataUrl(dataUrl, `mask-${Date.now()}.png`);
                } catch (err) {
                  console.error("Generate mask failed", err);
                  alert("Failed to generate mask: " + err.message);
                }
              }}
              disabled={processing}
            >
              Download Mask
            </button>
            <button
              onClick={async () => {
                // Process full original image using ALL ROI polygons
                if (!selectedFile) {
                  alert("Select the original image (click a thumbnail) before processing.");
                  return;
                }
                // Collect all polygons: saved ones + current if 3+ points
                const allPolys = [...polygons.map(p => p.points)];
                if (polygonPoints.length >= 3) allPolys.push(polygonPoints);
                if (allPolys.length === 0) {
                  alert("Draw at least one polygon (min 3 points; click near the first point to close) to process.");
                  return;
                }
                try {
                  setProcessing(true);
                  // Convert all polygons to server format: array of polygons, each polygon is array of {x, y}
                  const polygonsForServer = allPolys.map(poly => 
                    poly.map(p => ({ x: Math.round(p.nx), y: Math.round(p.ny) }))
                  );
                  const form = new FormData();
                  form.append("file", selectedFile, selectedFile.name || "original.png");
                  form.append("polygons", JSON.stringify(polygonsForServer));
                  const res = await fetch("http://127.0.0.1:8000/process_roi", {
                    method: "POST",
                    body: form
                  });
                  if (!res.ok) {
                    const text = await res.text();
                    throw new Error(`Server error ${res.status}: ${text}`);
                  }
                  const json = await res.json().catch(() => null);
                  if (!json || !json.final) {
                    alert("Server returned no final image. Check logs.");
                    return;
                  }
                  setFullFinal("data:image/png;base64," + json.final);
                  setOpenAiMeta(json.openai || null);
                  if (json.image_width != null && json.image_height != null) {
                    setFullFinalImageSize({ width: json.image_width, height: json.image_height });
                  }
                  // Parse text regions from OCR if available.
                  // Preferred geometry source: backend-echoed original polygons.
                  const responsePolygons = Array.isArray(json.polygons) ? json.polygons : [];
                  const ocrRegions = Array.isArray(json.text_regions) ? json.text_regions : [];
                  const polygonToBox = (poly) => {
                    if (!Array.isArray(poly) || poly.length === 0) return [[0, 0], [0, 0], [0, 0], [0, 0]];
                    const xs = poly.map((p) => Number(p?.x ?? (Array.isArray(p) ? p[0] : 0)));
                    const ys = poly.map((p) => Number(p?.y ?? (Array.isArray(p) ? p[1] : 0)));
                    const minX = Math.min(...xs);
                    const minY = Math.min(...ys);
                    const maxX = Math.max(...xs);
                    const maxY = Math.max(...ys);
                    return [[minX, minY], [maxX, minY], [maxX, maxY], [minX, maxY]];
                  };

                  let regions = [];
                  if (responsePolygons.length > 0) {
                    regions = responsePolygons.map((poly, idx) => {
                      const textInfo = ocrRegions[idx] || {};
                      return {
                        id: cryptoRandomId(),
                        text: textInfo.text != null ? String(textInfo.text) : "",
                        score: textInfo.score != null ? Number(textInfo.score) : 1,
                        polygon: poly,
                        angle_deg: textInfo.angle_deg != null ? Number(textInfo.angle_deg) : 0,
                        box: polygonToBox(poly),
                        color: textInfo.color || null,
                        color_bgr: textInfo.color_bgr || null,
                        background_color: textInfo.background_color || null,
                        background_color_bgr: textInfo.background_color_bgr || null,
                      };
                    });
                  } else {
                    // Backward compatibility for older backend responses.
                    regions = ocrRegions.map((r) => {
                      const poly = Array.isArray(r.polygon) ? r.polygon : [];
                      return {
                        id: cryptoRandomId(),
                        text: r.text != null ? String(r.text) : "",
                        score: r.score != null ? Number(r.score) : 1,
                        polygon: poly,
                        angle_deg: r.angle_deg != null ? Number(r.angle_deg) : 0,
                        box: polygonToBox(poly),
                        color: r.color || null,
                        color_bgr: r.color_bgr || null,
                        background_color: r.background_color || null,
                        background_color_bgr: r.background_color_bgr || null,
                      };
                    });
                  }
                  setTextRegions(regions);
                  
                  // Store polygon colors from response
                  if (json.text_regions && json.text_regions.length > 0) {
                    const colorsMap = {};
                    json.text_regions.forEach((r, idx) => {
                      // Match polygon by comparing coordinates
                      polygons.forEach((poly, polyIdx) => {
                        if (polyIdx === idx && r.color) {
                          colorsMap[poly.id] = {
                            color: r.color,
                            background_color: r.background_color,
                          };
                        }
                      });
                    });
                    setPolygonColors(prev => ({ ...prev, ...colorsMap }));
                  }
                  // Update processed mask if present
                  if (json.mask) setProcessed(prev => ({ ...prev, mask: "data:image/png;base64," + json.mask }));
                } catch (err) {
                  console.error("process_roi failed", err);
                  alert("Full inpaint failed: " + err.message);
                } finally {
                  setProcessing(false);
                }
              }}
              disabled={processing}
            >
              Process ROI (Full Inpaint)
            </button>
            {/* saved polygons list with delete */}
            {polygons.length > 0 && (
              <div style={{ marginLeft: 12 }}>
                <strong>Saved polygons:</strong>
                <ul style={{ margin: "6px 0", paddingLeft: 18 }}>
                  {polygons.map((poly, idx) => (
                    <li key={poly.id} style={{ marginBottom: 6 }}>
                      Polygon {idx + 1}{" "}
                      <button
                        onClick={() => {
                          setPolygons(ps => ps.filter(p => p.id !== poly.id));
                        }}
                        style={{ marginLeft: 8 }}
                      >
                        Delete
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="cropped-area card">
        <h3>Cropped Result</h3>
        <div id="croppedContainer">
          {cropped.map((c, i) => (
            // eslint-disable-next-line react/no-array-index-key
            <img key={i} src={c} alt={`crop-${i}`} />
          ))}
        </div>
      </div>
      <div className="processed-area card" style={{ marginTop: 12 }}>
        <h3>Processed Results</h3>
        <div id="processedContainer">
          {processed && (processed.annotated || processed.mask || processed.erased || processed.original) ? (
            <>
              {processed.annotated && <img src={processed.annotated} alt="processed-annotated" />}
              {processed.mask && <img src={processed.mask} alt="processed-mask" />}
              {processed.erased && <img src={processed.erased} alt="processed-erased" />}
            </>
          ) : (
            <div className="placeholder">No processed images yet</div>
          )}
          {maskDataUrl && (
            <div style={{ marginTop: 8 }}>
              <h4>Mask Preview</h4>
              <img src={maskDataUrl} alt="mask-preview" style={{ maxWidth: 320 }} />
            </div>
          )}
          <div style={{ marginTop: 8 }}>
            <button
              onClick={async () => {
                // apply processed erased patch onto the original image in the browser
                if (!processed || !processed.erased) {
                  alert("No erased image available to apply.");
                  return;
                }
                if (!lastCrop || !rightImgRef.current) {
                  alert("No crop info available. Make a crop first.");
                  return;
                }
                // composite routine
                const loadImage = src =>
                  new Promise((res, rej) => {
                    const img = new Image();
                    img.crossOrigin = "anonymous";
                    img.onload = () => res(img);
                    img.onerror = rej;
                    img.src = src;
                  });

                try {
                  const origEl = rightImgRef.current;
                  const origW = origEl.naturalWidth;
                  const origH = origEl.naturalHeight;
                  const mainCanvas = document.createElement("canvas");
                  mainCanvas.width = origW;
                  mainCanvas.height = origH;
                  const mctx = mainCanvas.getContext("2d");
                  // draw original full-size
                  mctx.drawImage(origEl, 0, 0, origW, origH);

                  const erasedImg = await loadImage(processed.erased);
                  let maskImg = null;
                  if (processed.mask) {
                    maskImg = await loadImage(processed.mask);
                  }

                  // polygon-aware paste: if lastCrop.polygon exists, inverse-warp rect -> quad
                  if (lastCrop && lastCrop.polygon) {
                    const { nWidth, nHeight, polygon } = lastCrop;
                    // prepare source canvas (erased patch) and apply mask if present
                    const srcCanvasRect = document.createElement("canvas");
                    srcCanvasRect.width = nWidth;
                    srcCanvasRect.height = nHeight;
                    const sctx = srcCanvasRect.getContext("2d");
                    sctx.drawImage(erasedImg, 0, 0, nWidth, nHeight);
                    if (maskImg) {
                      sctx.globalCompositeOperation = "destination-in";
                      sctx.drawImage(maskImg, 0, 0, nWidth, nHeight);
                      sctx.globalCompositeOperation = "source-over";
                    }

                    // polygon points are in natural pixel coords (ordered)
                    const dst0 = polygon[0];
                    const dst1 = polygon[1];
                    const dst2 = polygon[2];
                    const dst3 = polygon[3];

                    // map rectangle->quad by two triangles
                    // rect tri1: (0,0),(nWidth,0),(nWidth,nHeight) -> dst0,dst1,dst2
                    drawTriangleToCtx(
                      mctx,
                      srcCanvasRect,
                      [{ x: 0, y: 0 }, { x: nWidth, y: 0 }, { x: nWidth, y: nHeight }],
                      [dst0, dst1, dst2]
                    );

                    // rect tri2: (0,0),(nWidth,nHeight),(0,nHeight) -> dst0,dst2,dst3
                    drawTriangleToCtx(
                      mctx,
                      srcCanvasRect,
                      [{ x: 0, y: 0 }, { x: nWidth, y: nHeight }, { x: 0, y: nHeight }],
                      [dst0, dst2, dst3]
                    );

                    const finalDataUrl = mainCanvas.toDataURL("image/png");
                    setFinalImage(finalDataUrl);
                    setCompositeImageSize({ width: origW, height: origH });
                    const mappedPoly = cropTextRegions.map((r) => ({
                      ...r,
                      box: r.box.map(([cx, cy]) => {
                        const p = mapCropPointToFull(cx, cy, lastCrop);
                        return [p.x, p.y];
                      }),
                    }));
                    setCompositeTextRegions(mappedPoly);
                  } else {
                    // axis-aligned paste (existing flow)
                    const { nx, ny, nWidth, nHeight } = lastCrop;
                    // temporary canvas for the patch
                    const tmp = document.createElement("canvas");
                    tmp.width = nWidth;
                    tmp.height = nHeight;
                    const tctx = tmp.getContext("2d");
                    // draw erased patch to tmp
                    tctx.drawImage(erasedImg, 0, 0, nWidth, nHeight);
                    if (maskImg) {
                      // apply mask: keep only where mask is white
                      tctx.globalCompositeOperation = "destination-in";
                      tctx.drawImage(maskImg, 0, 0, nWidth, nHeight);
                      tctx.globalCompositeOperation = "source-over";
                    }
                    // draw tmp onto main at the correct location
                    mctx.drawImage(tmp, nx, ny);
                    const finalDataUrl = mainCanvas.toDataURL("image/png");
                    setFinalImage(finalDataUrl);
                    setCompositeImageSize({ width: origW, height: origH });
                    const mapped = cropTextRegions.map((r) => ({
                      ...r,
                      box: r.box.map(([cx, cy]) => {
                        const p = mapCropPointToFull(cx, cy, lastCrop);
                        return [p.x, p.y];
                      }),
                    }));
                    setCompositeTextRegions(mapped);
                  }
                } catch (err) {
                  console.error("Composite failed", err);
                  alert("Failed to composite image: " + err.message);
                }
              }}
              disabled={processing}
            >
              Apply to Original
            </button>
          </div>
        </div>
      </div>
      <div className="processed-area card" style={{ marginTop: 12 }}>
        <h3>Full Processing Results</h3>
        <div style={{ display: "flex", gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
          <div>
            <h4>Annotated</h4>
            {processed && processed.annotated ? (
              <>
                <img src={processed.annotated} alt="annotated" style={{ maxWidth: 420, display: "block" }} />
                <div style={{ marginTop: 6 }}>
                  <button onClick={() => downloadDataUrl(processed.annotated, `annotated-${Date.now()}.png`)}>Download Annotated</button>
                </div>
              </>
            ) : (
              <div className="placeholder">No annotated image</div>
            )}
          </div>

          <div>
            <h4>Mask</h4>
            {processed && processed.mask ? (
              <>
                <img src={processed.mask} alt="mask" style={{ maxWidth: 420, display: "block" }} />
                <div style={{ marginTop: 6 }}>
                  <button onClick={() => downloadDataUrl(processed.mask, `mask-${Date.now()}.png`)}>Download Mask</button>
                </div>
              </>
            ) : (
              <div className="placeholder">No mask image</div>
            )}
          </div>

          <div style={{ flexBasis: "100%" }} />

          <div style={{ width: "100%" }}>
            <h4>Final (Full Image) — detected text (drag to move, click to edit)</h4>
            {fullFinal ? (
              <>
                {openAiMeta && (
                  <div className="card" style={{ margin: "8px 0", padding: 10 }}>
                    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
                      <strong>OpenAI</strong>
                      <span>enabled: {String(!!openAiMeta.enabled)}</span>
                      {openAiMeta.model ? <span>model: {String(openAiMeta.model)}</span> : null}
                      {typeof openAiMeta.cost_usd === "number" ? (
                        <span>cost: ${openAiMeta.cost_usd.toFixed(6)}</span>
                      ) : null}
                      {openAiMeta.error ? (
                        <span style={{ color: "#b91c1c" }}>error: {String(openAiMeta.error)}</span>
                      ) : null}
                    </div>
                  </div>
                )}
                <div className="final-image-wrapper" style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
                  <img
                    ref={fullFinalImgRef}
                    src={fullFinal}
                    alt="final-full"
                    style={{ maxWidth: "100%", display: "block", verticalAlign: "top" }}
                    onLoad={() => {
                      const img = fullFinalImgRef.current;
                      if (!img) return;
                      if (!fullFinalImageSize) {
                        setFullFinalImageSize({
                          width: img.naturalWidth,
                          height: img.naturalHeight,
                        });
                      }
                      const rect = img.getBoundingClientRect();
                      setFullFinalDisplaySize({ width: rect.width, height: rect.height });
                    }}
                  />
                  {textRegions.length > 0 && fullFinalImageSize && fullFinalDisplaySize && (
                    <div
                      className="final-text-overlay"
                      style={{
                        position: "absolute",
                        left: 0,
                        top: 0,
                        width: fullFinalDisplaySize.width,
                        height: fullFinalDisplaySize.height,
                        pointerEvents: "auto",
                      }}
                      onClick={() => setSelectedFullFinalTextId(null)}
                    >
                      {textRegions.map((region) => {
                        const box = region.box;
                        const minX = Math.min(...box.map((p) => p[0]));
                        const minY = Math.min(...box.map((p) => p[1]));
                        const maxX = Math.max(...box.map((p) => p[0]));
                        const maxY = Math.max(...box.map((p) => p[1]));
                        const scaleX = fullFinalDisplaySize.width / fullFinalImageSize.width;
                        const scaleY = fullFinalDisplaySize.height / fullFinalImageSize.height;
                        const left = minX * scaleX;
                        const top = minY * scaleY;
                        const width = Math.max(20, (maxX - minX) * scaleX);
                        const height = Math.max(14, (maxY - minY) * scaleY);
                        const isSelected = selectedFullFinalTextId === region.id;
                        return (
                          <div
                            key={region.id}
                            className="text-region-box"
                            style={{
                              position: "absolute",
                              left,
                              top,
                              width,
                              height,
                              transform: `rotate(${Number(region.angle_deg || 0)}deg)`,
                              transformOrigin: "top left",
                              pointerEvents: "auto",
                              cursor: "move",
                              border: isSelected ? "1px solid rgba(37,99,235,0.7)" : "1px solid transparent",
                              background: "transparent",
                              padding: "1px 4px",
                              fontSize: Math.max(10, height * 0.7),
                              lineHeight: 1.1,
                              overflow: "visible",
                              boxSizing: "border-box",
                            }}
                            onClick={(e) => e.stopPropagation()}
                            onMouseDown={(e) => {
                              setSelectedFullFinalTextId(region.id);
                              if (e.target.closest(".text-region-content")) return;
                              e.preventDefault();
                              textDragRef.current = {
                                regionId: region.id,
                                startX: e.clientX,
                                startY: e.clientY,
                                startBox: region.box.map((p) => [...p]),
                              };
                            }}
                          >
                            <div
                              className="text-region-content"
                              onClick={(e) => { e.stopPropagation(); setSelectedFullFinalTextId(region.id); }}
                              onMouseDown={(e) => e.stopPropagation()}
                              style={{
                                width: "100%",
                                height: "100%",
                                overflow: "hidden",
                                fontSize: "inherit",
                                lineHeight: 1.2,
                                cursor: "text",
                                color: region.color || undefined, // Use detected text color
                              }}
                              title={`Click to edit | Text color: ${region.color || 'default'}`}
                            >
                              {useHtmlForDisplay(region.html, region.text) ? (
                                <div 
                                  className="text-region-html" 
                                  dangerouslySetInnerHTML={{ __html: region.html }}
                                  style={{ color: region.color || undefined }}
                                />
                              ) : (
                                <span style={{ color: region.color || undefined }}>
                                  {region.text || " "}
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
                <div style={{ marginTop: 6, display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <button onClick={() => downloadDataUrl(fullFinal, `final-${Date.now()}.png`)}>Download Final</button>
                  <button
                    onClick={async () => {
                      const dataUrl = await exportImageWithText(fullFinal, fullFinalImageSize || undefined, textRegions);
                      downloadDataUrl(dataUrl, `final-edited-${Date.now()}.png`);
                    }}
                  >
                    Download edited
                  </button>
                </div>
                {selectedFullFinalTextId && (() => {
                  const id = selectedFullFinalTextId;
                  const region = textRegions.find((r) => r.id === id);
                  if (!region) return null;
                  const initialContent = region.html ?? region.text ?? "";
                  return (
                    <div className="card" style={{ marginTop: 12 }}>
                      <h3>Edit text — Full image</h3>
                      <div className="final-full-editor" style={{ marginBottom: 8 }} key={id}>
                        <ReactQuill
                          ref={fullFinalQuillRef}
                          theme="snow"
                          defaultValue={initialContent}
                          modules={quillModules}
                          style={{ minHeight: 140 }}
                        />
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          const editor = typeof fullFinalQuillRef.current?.getEditor === "function" ? fullFinalQuillRef.current.getEditor() : null;
                          const rootEl = editor?.root ?? document.querySelector(".final-full-editor .ql-editor");
                          const newText = editor ? (editor.getText().replace(/\n+$/, "") ?? "").trim() : (rootEl ? (rootEl.innerText ?? rootEl.textContent ?? "").trim() : "");
                          let newHtml = editor ? (editor.root?.innerHTML ?? "") : (rootEl ? rootEl.innerHTML ?? "" : "");
                          const stripped = (newHtml.replace(/<[^>]+>/g, "").trim() || "").replace(/\s+/g, " ").trim();
                          if (stripped.length === 0) newHtml = "";
                          const htmlToStore = stripped.length > 0 ? newHtml : undefined;
                          let newFontFamily = rootEl ? getEffectiveFontFamily(rootEl) : "";
                          setTextRegions((prev) => {
                            const r = prev.find((x) => x.id === id);
                            if (!r || (r.text ?? "").trim() === newText) return prev;
                            const payload = { ...r, text: newText, html: htmlToStore };
                            if (newFontFamily) payload.fontFamily = newFontFamily;
                            if (fullFinalDisplaySize && fullFinalImageSize) {
                              const b = r.box;
                              const maxY = Math.max(...b.map((p) => p[1]));
                              const minY = Math.min(...b.map((p) => p[1]));
                              const scaleY = fullFinalDisplaySize.height / fullFinalImageSize.height;
                              const displayH = (maxY - minY) * scaleY;
                              const fontSizeDisplay = Math.max(10, displayH * 0.7);
                              payload.box = boxFromMeasuredText(r, newText, fullFinalDisplaySize, fullFinalImageSize, fontSizeDisplay);
                            }
                            return prev.map((x) => (x.id === id ? payload : x));
                          });
                          setTimeout(() => setSelectedFullFinalTextId(null), 0);
                        }}
                      >
                        Done editing
                      </button>
                    </div>
                  );
                })()}
              </>
            ) : (
              <div className="placeholder">No final image yet</div>
            )}
          </div>
        </div>
      </div>
      {finalImage && (
        <div className="cropped-area card" style={{ marginTop: 12 }}>
          <h3>Final Composite — detected text (drag to move, click to edit)</h3>
          <div className="final-image-wrapper" style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
            <img
              ref={finalCompositeImgRef}
              src={finalImage}
              alt="final-composite"
              style={{ maxWidth: "100%", display: "block", verticalAlign: "top" }}
              onLoad={() => {
                const img = finalCompositeImgRef.current;
                if (!img) return;
                const rect = img.getBoundingClientRect();
                setCompositeDisplaySize({ width: rect.width, height: rect.height });
              }}
            />
            {compositeTextRegions.length > 0 && compositeImageSize && compositeDisplaySize && (
              <div
                className="final-text-overlay"
                style={{
                  position: "absolute",
                  left: 0,
                  top: 0,
                  width: compositeDisplaySize.width,
                  height: compositeDisplaySize.height,
                  pointerEvents: "auto",
                }}
                onClick={() => setSelectedCompositeTextId(null)}
              >
                {compositeTextRegions.map((region) => {
                  const box = region.box;
                  const minX = Math.min(...box.map((p) => p[0]));
                  const minY = Math.min(...box.map((p) => p[1]));
                  const maxX = Math.max(...box.map((p) => p[0]));
                  const maxY = Math.max(...box.map((p) => p[1]));
                  const scaleX = compositeDisplaySize.width / compositeImageSize.width;
                  const scaleY = compositeDisplaySize.height / compositeImageSize.height;
                  const left = minX * scaleX;
                  const top = minY * scaleY;
                  const width = Math.max(20, (maxX - minX) * scaleX);
                  const height = Math.max(14, (maxY - minY) * scaleY);
                  const isSelected = selectedCompositeTextId === region.id;
                  return (
                    <div
                      key={region.id}
                      className="text-region-box"
                      style={{
                        position: "absolute",
                        left,
                        top,
                        width,
                        height,
                        transform: `rotate(${Number(region.angle_deg || 0)}deg)`,
                        transformOrigin: "top left",
                        pointerEvents: "auto",
                        cursor: "move",
                        border: isSelected ? "1px solid rgba(37,99,235,0.7)" : "1px solid transparent",
                        background: "transparent",
                        padding: "1px 4px",
                        fontSize: Math.max(10, height * 0.7),
                        lineHeight: 1.1,
                        overflow: "visible",
                        boxSizing: "border-box",
                      }}
                      onClick={(e) => e.stopPropagation()}
                      onMouseDown={(e) => {
                        setSelectedCompositeTextId(region.id);
                        if (e.target.closest(".text-region-content")) return;
                        e.preventDefault();
                        compositeDragRef.current = {
                          regionId: region.id,
                          startX: e.clientX,
                          startY: e.clientY,
                          startBox: region.box.map((p) => [...p]),
                        };
                      }}
                    >
                      <div
                        className="text-region-content"
                        onClick={(e) => { e.stopPropagation(); setSelectedCompositeTextId(region.id); }}
                        onMouseDown={(e) => e.stopPropagation()}
                        style={{
                          width: "100%",
                          height: "100%",
                          overflow: "hidden",
                          fontSize: "inherit",
                          lineHeight: 1.2,
                          cursor: "text",
                          color: region.color || undefined, // Use detected text color
                        }}
                        title={`Click to edit | Text color: ${region.color || 'default'}`}
                      >
                        {useHtmlForDisplay(region.html, region.text) ? (
                          <div 
                            className="text-region-html" 
                            dangerouslySetInnerHTML={{ __html: region.html }}
                            style={{ color: region.color || undefined }}
                          />
                        ) : (
                          <span style={{ color: region.color || undefined }}>
                            {region.text || " "}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
            {selectedCompositeTextId && compositeDisplaySize && compositeImageSize && (() => {
              const id = selectedCompositeTextId;
              const region = compositeTextRegions.find((r) => r.id === id);
              if (!region) return null;
              const box = region.box;
              const minX = Math.min(...box.map((p) => p[0]));
              const minY = Math.min(...box.map((p) => p[1]));
              const scaleX = compositeDisplaySize.width / compositeImageSize.width;
              const scaleY = compositeDisplaySize.height / compositeImageSize.height;
              const regionLeft = minX * scaleX;
              const regionTop = minY * scaleY;
              const EDITOR_HEIGHT = 220;
              const GAP = 8;
              const SHIFT_UP = 50;
              const defaultTop = Math.max(8, regionTop - EDITOR_HEIGHT - GAP - SHIFT_UP);
              const posLeft = compositeEditorPosition?.left ?? regionLeft;
              const posTop = compositeEditorPosition?.top ?? defaultTop;
              return (
                <CompositeTextEditor
                  key={id}
                  region={region}
                  posLeft={posLeft}
                  posTop={posTop}
                  compositeDisplaySize={compositeDisplaySize}
                  compositeImageSize={compositeImageSize}
                  boxFromMeasuredText={boxFromMeasuredText}
                  onDone={({ text: newText, html: newHtml, fontFamily: newFontFamily }) => {
                    const htmlToStore = (newHtml && (newHtml.replace(/<[^>]+>/g, "").trim() || "").replace(/\s+/g, " ").trim().length > 0) ? newHtml : undefined;
                    setCompositeTextRegions((prev) => {
                      const r = prev.find((x) => x.id === id);
                      if (!r) return prev;
                      const payload = { ...r, text: newText, html: htmlToStore };
                      if (newFontFamily && newFontFamily.trim()) payload.fontFamily = newFontFamily.trim();
                      if (compositeDisplaySize && compositeImageSize) {
                        const b = r.box;
                        const maxY = Math.max(...b.map((p) => p[1]));
                        const minY = Math.min(...b.map((p) => p[1]));
                        const scaleY = compositeDisplaySize.height / compositeImageSize.height;
                        const displayH = (maxY - minY) * scaleY;
                        const fontSizeDisplay = Math.max(10, displayH * 0.7);
                        payload.box = boxFromMeasuredText(r, newText, compositeDisplaySize, compositeImageSize, fontSizeDisplay);
                      }
                      return prev.map((x) => (x.id === id ? payload : x));
                    });
                    setTimeout(() => {
                      setSelectedCompositeTextId(null);
                      setCompositeEditorPosition(null);
                    }, 0);
                  }}
                  onDragStart={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    compositeEditorDragRef.current = {
                      clientX: e.clientX,
                      clientY: e.clientY,
                      startLeft: posLeft,
                      startTop: posTop,
                    };
                  }}
                />
              );
            })()}
          </div>
          <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button onClick={() => downloadDataUrl(finalImage, `composite-${Date.now()}.png`)}>Download image</button>
            <button
              onClick={async () => {
                const dataUrl = await exportImageWithText(finalImage, compositeImageSize || undefined, compositeTextRegions);
                downloadDataUrl(dataUrl, `composite-edited-${Date.now()}.png`);
              }}
            >
              Download edited
            </button>
          </div>
        </div>
      )}
    </div>
  );
}