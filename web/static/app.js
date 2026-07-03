/* wifisim planner front-end.
   Hero = coverage overlay; pan/zoom view transform; transmitter markers with
   antenna boresight on a graticule; resizable sidebar; live RT-solver params. */
"use strict";

const api = {
  async get(u){ const r = await fetch(u); return r.json(); },
  async send(u, method, body){
    const r = await fetch(u, {method, headers:{"Content-Type":"application/json"},
      body: body?JSON.stringify(body):undefined});
    return r.json();
  }
};

const LS = {                         // tiny localStorage helper (standalone app)
  get(k,d){ try{ const v=localStorage.getItem(k); return v==null?d:JSON.parse(v); }catch(e){ return d; } },
  set(k,v){ try{ localStorage.setItem(k, JSON.stringify(v)); }catch(e){} }
};

const state = {
  grid:null, scene:null, txs:[], selected:null, options:null, patterns:[],
  metric:"best_rsrp", overlay:null, cacheStatus:{},
  base:{x0:0,y0:0,w:1,h:1}, view:LS.get("view",{k:1,ox:0,oy:0}),
  drag:null, dpr: window.devicePixelRatio||1,
  geoFiles:[], footprint:{geometry:[],mesh:[]},
  routes:{}, viewMode:"2d",
  geo3d:{geometry:[],mesh:[]}, routeHover:null
};
const txVisible = name => { const t=state.txs.find(x=>x.name===name); return !t||t.enabled; };

const el = id => document.getElementById(id);
const display = el("display"), canvas = el("overlay"), heatmap = el("heatmap");
const ctx = canvas.getContext("2d");

/* ---------- view transform: world <-> screen -------------------------- */
function computeBase(){
  const r = display.getBoundingClientRect();
  const W=r.width, H=r.height, m=26;
  const g=state.grid, ax=(g.x_max-g.x_min), ay=(g.y_max-g.y_min);
  const availW=W-2*m, availH=H-2*m, sceneAspect=ax/ay;
  // Fit by width, then shrink to height if needed - preserving aspect so the
  // metres-per-pixel scale is identical on both axes (no stretch).
  let w=availW, h=w/sceneAspect;
  if(h>availH){ h=availH; w=h*sceneAspect; }
  state._W=W; state._H=H;
  state.base={x0:(W-w)/2, y0:(H-h)/2, w, h};
}
function sX(wx){ const g=state.grid; return state.base.x0 + (wx-g.x_min)/(g.x_max-g.x_min)*state.base.w; }
function sY(wy){ const g=state.grid; return state.base.y0 + (g.y_max-wy)/(g.y_max-g.y_min)*state.base.h; }
function worldToScreen(wx,wy){ const v=state.view; return [v.ox+v.k*sX(wx), v.oy+v.k*sY(wy)]; }
function screenToWorld(px,py){
  const v=state.view, g=state.grid, b=state.base;
  const ssx=(px-v.ox)/v.k, ssy=(py-v.oy)/v.k;
  return [ g.x_min + (ssx-b.x0)/b.w*(g.x_max-g.x_min),
           g.y_max - (ssy-b.y0)/b.h*(g.y_max-g.y_min) ];
}
function metersToPx(m){ const g=state.grid; return m/(g.x_max-g.x_min)*state.base.w*state.view.k; }

function fitDisplay(){
  computeBase();
  const W=state._W, H=state._H, dpr=state.dpr;
  canvas.width=W*dpr; canvas.height=H*dpr;
  canvas.style.width=W+"px"; canvas.style.height=H+"px";
  ctx.setTransform(dpr,0,0,dpr,0,0);
  positionHeatmap(); draw(); resize3D();
  if(chart.profiles && el("routePlotBox").style.display!=="none") drawRouteChart();
}
function resetView(){ state.view={k:1,ox:0,oy:0}; LS.set("view",state.view); fitDisplay(); updateZoomReadout(); fit3D(); }

function positionHeatmap(){
  const v=state.view, b=state.base;
  heatmap.style.left  = (v.ox+v.k*b.x0)+"px";
  heatmap.style.top   = (v.oy+v.k*b.y0)+"px";
  heatmap.style.width = (v.k*b.w)+"px";
  heatmap.style.height= (v.k*b.h)+"px";
}
function updateZoomReadout(){ el("zoomReadout").textContent=Math.round(state.view.k*100)+"%"; }

/* ---------- drawing --------------------------------------------------- */
function draw(){
  const W=state._W, H=state._H;
  ctx.clearRect(0,0,W,H);
  drawGraticule(); drawFootprint(); drawRoutes();
  state.txs.filter(t=>txVisible(t.name)).forEach(drawTx);
}
function drawFootprint(){
  const fp=state.footprint; if(!fp) return;
  const drawSegs=(segs,color,width)=>{
    if(!segs||!segs.length) return;
    ctx.save(); ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath();
    for(const s of segs){
      const a=worldToScreen(s[0],s[1]), b=worldToScreen(s[2],s[3]);
      ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
    }
    ctx.stroke(); ctx.restore();
  };
  drawSegs(fp.geometry, "rgba(0,0,80,.45)", 1);        // building/scene edges (blue)
  drawSegs(fp.mesh, "rgba(227,0,20,.45)", 1);          // prediction mesh (red)
}
function niceStep(span){
  const raw=span/9, p=Math.pow(10,Math.floor(Math.log10(raw)));
  const n=raw/p; return (n<1.5?1:n<3.5?2:n<7.5?5:10)*p;
}
function drawGraticule(){
  const g=state.grid, W=state._W, H=state._H;
  const visSpanX=(W/state.view.k)/state.base.w*(g.x_max-g.x_min);
  const step=Math.max(niceStep(visSpanX), 0.001);
  ctx.save();
  ctx.strokeStyle="rgba(0,0,80,.10)"; ctx.lineWidth=1;
  ctx.fillStyle="rgba(0,0,80,.45)"; ctx.font="10px ui-monospace,monospace";
  for(let x=Math.ceil(g.x_min/step)*step; x<=g.x_max+1e-6; x+=step){
    const px=worldToScreen(x,0)[0]; if(px<-20||px>W+20) continue;
    ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,H); ctx.stroke();
    ctx.fillText(x.toFixed(step<1?1:0), px+3, H-4);
  }
  for(let y=Math.ceil(g.y_min/step)*step; y<=g.y_max+1e-6; y+=step){
    const py=worldToScreen(0,y)[1]; if(py<-20||py>H+20) continue;
    ctx.beginPath(); ctx.moveTo(0,py); ctx.lineTo(W,py); ctx.stroke();
    ctx.fillText(y.toFixed(step<1?1:0), 3, py-3);
  }
  ctx.restore();
}
function drawTx(tx){
  const [px,py]=worldToScreen(tx.position[0],tx.position[1]);
  if(px<-60||px>state._W+60||py<-60||py>state._H+60) return;
  const sel=state.selected===tx.name;
  const yaw=(tx.orientation?.[0]||0)*Math.PI/180;
  ctx.save();
  const directional = tx.antenna && tx.antenna.pattern!=="iso" && tx.antenna.pattern!=="dipole";
  if(directional){
    const len=Math.max(20, metersToPx(7)), spread=(tx.antenna.az_hpbw_deg||65)*Math.PI/180/2;
    ctx.translate(px,py); ctx.rotate(-yaw);
    const grd=ctx.createRadialGradient(0,0,3,0,0,len);
    grd.addColorStop(0, tx.color+"cc"); grd.addColorStop(1, tx.color+"00");
    ctx.fillStyle=grd;
    ctx.beginPath(); ctx.moveTo(0,0); ctx.arc(0,0,len,-spread,spread); ctx.closePath(); ctx.fill();
    ctx.setTransform(state.dpr,0,0,state.dpr,0,0);
  }
  ctx.translate(px,py);
  ctx.fillStyle = tx.enabled? tx.color : "#9aa3b2";
  ctx.strokeStyle = sel? "#e30014" : "#000050"; ctx.lineWidth = sel?2.5:1.5;
  ctx.beginPath(); ctx.moveTo(0,-8); ctx.lineTo(7,6); ctx.lineTo(-7,6); ctx.closePath();
  ctx.fill(); ctx.stroke();
  ctx.fillStyle="#000050"; ctx.font="600 11px system-ui,sans-serif";
  ctx.strokeStyle="rgba(255,255,255,.85)"; ctx.lineWidth=3;
  ctx.strokeText(tx.name, 10, -8); ctx.fillText(tx.name, 10, -8);
  const c=state.cacheStatus[tx.name];
  if(c!==undefined){ ctx.fillStyle=c?"#00963f":"#e30014"; ctx.fillText(c?"\u25CF":"\u25CB", 10, 6); }
  ctx.restore();
}

/* ---------- transmitter list + editor --------------------------------- */
function renderTxList(){
  const box=el("txlist"); box.innerHTML="";
  if(!state.txs.length){ box.innerHTML='<span class="empty">No transmitters yet. Use + Add AP.</span>'; }
  state.txs.forEach(tx=>{
    const d=document.createElement("div");
    d.className="txitem"+(state.selected===tx.name?" sel":"")+(tx.enabled?"":" hidden");
    const c=state.cacheStatus[tx.name];
    d.innerHTML=`<input type="checkbox" title="include in simulation" ${tx.enabled?"checked":""}>
      <span class="dot" style="background:${tx.color}"></span>
      <span class="nm">${tx.name}</span>
      <span class="meta">ch${tx.channel} \u00b7 ${tx.tx_power_dbm}dBm \u00b7 ${tx.antenna.pattern}</span>
      <span class="cache ${c?'hit':'miss'}">${c?'cached':'dirty'}</span>`;
    const cb=d.querySelector("input");
    cb.onclick=e=>e.stopPropagation();
    cb.onchange=async()=>{
      const payload={...tx, enabled:cb.checked};
      const res=await api.send("/api/transmitter","POST",payload);
      if(res.error){ cb.checked=tx.enabled; return; }
      state.cacheStatus=res.cache_status;
      await refreshState(false);
      maybeAutorun();
    };
    d.onclick=()=>selectTx(tx.name);
    box.appendChild(d);
  });
}
function selectTx(name){
  state.selected=name; const tx=state.txs.find(t=>t.name===name);
  const card=el("editorCard");
  if(!tx){ card.style.display="none"; draw(); renderTxList(); return; }
  card.style.display="block"; el("editorName").textContent=name;
  el("f_name").value=tx.name;
  el("f_x").value=tx.position[0]; el("f_y").value=tx.position[1]; el("f_z").value=tx.position[2];
  el("f_pwr").value=tx.tx_power_dbm;
  fillSelect(el("f_ch"), state.options.channels, tx.channel);
  fillSelect(el("f_bw"), state.options.bandwidths_mhz, tx.bandwidth_mhz);
  fillAntennaSelect(el("f_ant"), tx);
  el("f_az").value=tx.orientation[0];
  el("f_gain").value=tx.antenna.boresight_gain_dbi;
  el("f_hpbw").value=tx.antenna.az_hpbw_deg;
  draw(); renderTxList();
}
function fillSelect(sel, items, val){
  sel.innerHTML="";
  items.forEach(it=>{ const o=document.createElement("option");
    o.value=it; o.textContent=it; if(String(it)===String(val)) o.selected=true; sel.appendChild(o); });
}
function fillAntennaSelect(sel, tx){
  sel.innerHTML="";
  state.options.antenna_patterns.filter(p=>p!=="msi").forEach(p=>{
    const o=document.createElement("option"); o.value=p; o.textContent=p;
    if(tx.antenna.pattern===p) o.selected=true; sel.appendChild(o); });
  if(state.patterns.length){
    const grp=document.createElement("optgroup"); grp.label="MSI files";
    state.patterns.forEach(m=>{ if(m.error) return;
      const o=document.createElement("option"); o.value="msi::"+m.file;
      o.textContent=`${m.name} (${m.gain_dbi} dBi)`;
      if(tx.antenna.pattern==="msi" && tx.antenna.msi_file===m.file) o.selected=true;
      grp.appendChild(o); });
    sel.appendChild(grp);
  }
}
function editorToTx(){
  const tx=state.txs.find(t=>t.name===state.selected); if(!tx) return null;
  const antVal=el("f_ant").value; let antenna;
  if(antVal.startsWith("msi::")){
    antenna={...tx.antenna, pattern:"msi", msi_file:antVal.slice(5),
      num_rows:tx.antenna.num_rows||1, num_cols:tx.antenna.num_cols||1,
      polarization:tx.antenna.polarization||"V"};
  } else {
    antenna={...tx.antenna, pattern:antVal,
      boresight_gain_dbi:+el("f_gain").value, az_hpbw_deg:+el("f_hpbw").value};
  }
  return { name: el("f_name").value.trim()||tx.name,
    position:[+el("f_x").value, +el("f_y").value, +el("f_z").value],
    orientation:[+el("f_az").value, 0, 0], tx_power_dbm:+el("f_pwr").value,
    channel:+el("f_ch").value, bandwidth_mhz:+el("f_bw").value,
    enabled:tx.enabled, color:tx.color, antenna };
}
async function commitEditor(){
  const oldName=state.selected, payload=editorToTx(); if(!payload) return;
  if(payload.name!==oldName) await api.send("/api/transmitter/"+encodeURIComponent(oldName),"DELETE");
  const res=await api.send("/api/transmitter","POST",payload);
  if(res.error){ alert("Antenna error: "+res.error); await refreshState(false); return; }
  state.cacheStatus=res.cache_status; state.selected=payload.name;
  await refreshState(false); selectTx(payload.name); maybeAutorun();
}

/* ---------- server sync ----------------------------------------------- */
async function refreshState(redraw=true){
  const s=await api.get("/api/state");
  state.grid=s.grid; state.scene=s.scene; state.txs=s.transmitters; state.cacheStatus=s.cache_status;
  el("enginePill").innerHTML=`engine <b>${s.engine.name}</b>`;
  el("enginePill").className="enginepill"+(s.engine.name==="sionna_rt"?" rt":"");
  renderCacheStats(s.cache); renderTxList(); syncSceneInputs(); syncEngineInputs(s.engine);
  if(redraw){ fitDisplay(); updateZoomReadout(); }
  rebuild3D();
}
function renderCacheStats(c){
  el("cacheStats").innerHTML=
    row("hit rate",(c.hit_rate*100).toFixed(0)+"%")+
    row("hits / misses",`${c.hits} / ${c.misses}`)+
    row("layers on disk",c.disk_entries)+ row("memory",c.mem_entries);
}
const row=(k,v)=>`<div class="row"><span class="muted">${k}</span><span>${v}</span></div>`;

/* ---------- simulate -------------------------------------------------- */
async function run(){
  const btn=el("runBtn"); btn.disabled=true; btn.textContent="Running\u2026";
  try{
    const res=await api.send("/api/simulate","POST",{metric:state.metric, force:el("forceRun").checked});
    state.overlay=res.overlay; heatmap.src=res.overlay.image; positionHeatmap();
    decodeValues(res.overlay);
    res.per_tx.forEach(t=>state.cacheStatus[t.name]=true);
    renderSummary(res.summary); renderCacheStats(res.cache); updateLegend(res.overlay);
    renderTxList(); draw(); rebuild3D();
  } finally { btn.disabled=false; btn.textContent="Run"; }
}
function maybeAutorun(){ if(el("autorun").checked) run(); }
function renderSummary(s){
  el("stats").innerHTML=
    row("engine",s.engine)+ row("transmitters",s.n_transmitters)+
    row("coverage \u2265 \u221267 dBm",s["coverage_pct_-67dBm"]+"%")+
    row("coverage \u2265 \u221280 dBm",s["coverage_pct_-80dBm"]+"%")+
    row("median SINR",(s.median_sinr_db??"\u2014")+" dB")+
    row("compute time",(s.timing_s*1000).toFixed(0)+" ms")+
    row("cache hits/misses",`${s.cache_hits} / ${s.cache_misses}`);
}
function updateLegend(ov){
  el("legMin").textContent=ov.vmin; el("legMax").textContent=ov.vmax;
  el("legUnit").textContent=ov.unit; el("legBar").style.background=cmapCss(ov.cmap);
}
function cmapCss(name){
  if(name==="rssi") return "linear-gradient(90deg,rgb(0,0,255) 0%,rgb(0,125,255) 11.1%,"+
    "rgb(0,255,251) 22.2%,rgb(0,255,66) 33.3%,rgb(130,255,0) 44.4%,rgb(220,255,0) 55.6%,"+
    "rgb(255,219,0) 66.7%,rgb(255,137,0) 77.8%,rgb(255,0,55) 88.9%,rgb(255,0,255) 100%)";
  if(name==="turbo") return "linear-gradient(90deg,#30123b,#4669db,#26bf8c,#c4e02e,#fb8022,#7a0403)";
  return "linear-gradient(90deg,#222,#888,#eee)";
}

/* ---------- value grid (for hover read-out) --------------------------- */
function b64ToF32(b64){
  const bin=atob(b64), n=bin.length, u=new Uint8Array(n);
  for(let i=0;i<n;i++) u[i]=bin.charCodeAt(i);
  return new Float32Array(u.buffer);
}
function decodeValues(ov){
  if(!ov || !ov.values){ state.valueGrid=null; return; }
  state.valueGrid={arr:b64ToF32(ov.values), ny:ov.shape[0], nx:ov.shape[1],
    ext:ov.extent, unit:ov.unit, label:ov.label};
}
function valueAt(wx,wy){
  const g=state.valueGrid; if(!g) return undefined;
  const [xmin,xmax,ymin,ymax]=g.ext;
  if(wx<xmin||wx>xmax||wy<ymin||wy>ymax) return undefined;
  let c=Math.floor((wx-xmin)/(xmax-xmin)*g.nx);
  let r=Math.floor((wy-ymin)/(ymax-ymin)*g.ny);   // canonical: row 0 = y_min
  c=Math.max(0,Math.min(g.nx-1,c)); r=Math.max(0,Math.min(g.ny-1,r));
  return g.arr[r*g.nx+c];
}
function hideHover(){ el("hover").style.display="none"; }

/* ---------- canvas interaction: select / move / pan / zoom ------------ */
function txAt(px,py){
  for(let i=state.txs.length-1;i>=0;i--){
    if(!txVisible(state.txs[i].name)) continue;
    const [tx,ty]=worldToScreen(state.txs[i].position[0],state.txs[i].position[1]);
    if(Math.hypot(px-tx,py-ty)<13) return state.txs[i];
  } return null;
}
function localXY(e){ const r=canvas.getBoundingClientRect(); return [e.clientX-r.left, e.clientY-r.top]; }

canvas.addEventListener("mousedown", e=>{
  const [px,py]=localXY(e); const hit=txAt(px,py);
  if(hit){ state.drag={type:"tx",name:hit.name,moved:false}; selectTx(hit.name); }
  else { state.drag={type:"pan",sx:px,sy:py,ox:state.view.ox,oy:state.view.oy,moved:false};
         display.classList.add("panning"); }
});
window.addEventListener("mousemove", e=>{
  if(!state.drag){
    const [px,py]=localXY(e); const [wx,wy]=screenToWorld(px,py);
    const inside = px>=0&&py>=0&&px<=state._W&&py<=state._H;
    const hov=el("hover");
    if(inside){
      el("cursorReadout").innerHTML=`x ${wx.toFixed(1)} \u00b7 y ${wy.toFixed(1)} m`;
      const g=state.valueGrid;
      if(g){
        const v=valueAt(wx,wy);
        const txt=(v===undefined||isNaN(v))?"no coverage":`${v.toFixed(1)} ${g.unit}`;
        hov.innerHTML=`x ${wx.toFixed(1)}, y ${wy.toFixed(1)} m \u00b7 <b>${txt}</b>`;
        hov.style.left=px+"px"; hov.style.top=py+"px"; hov.style.display="block";
      } else hov.style.display="none";
    } else hov.style.display="none";
    return;
  }
  hideHover();
  const [px,py]=localXY(e);
  if(state.drag.type==="tx"){
    const [wx,wy]=screenToWorld(px,py); const tx=state.txs.find(t=>t.name===state.drag.name);
    tx.position[0]=+wx.toFixed(2); tx.position[1]=+wy.toFixed(2); state.drag.moved=true;
    if(state.selected===tx.name){ el("f_x").value=tx.position[0]; el("f_y").value=tx.position[1]; }
    draw();
  } else {
    state.view.ox=state.drag.ox+(px-state.drag.sx);
    state.view.oy=state.drag.oy+(py-state.drag.sy);
    state.drag.moved=true; positionHeatmap(); draw();
  }
});
window.addEventListener("mouseup", async ()=>{
  const d=state.drag; state.drag=null; display.classList.remove("panning");
  if(!d) return;
  if(d.type==="tx" && d.moved){
    const tx=state.txs.find(t=>t.name===d.name);
    const res=await api.send(`/api/transmitter/${encodeURIComponent(tx.name)}/move`,"POST",
      {x:tx.position[0], y:tx.position[1]});
    state.cacheStatus=res.cache_status; renderTxList(); draw(); maybeAutorun();
  } else if(d.type==="pan" && d.moved){ LS.set("view",state.view); }
  else if(d.type==="pan"){ selectTx(null); }    // click on empty = deselect
});
canvas.addEventListener("mouseleave", hideHover);
canvas.addEventListener("wheel", e=>{
  e.preventDefault();
  const [px,py]=localXY(e);
  const f=Math.exp(-e.deltaY*0.0015);
  const k0=state.view.k, k1=Math.min(40,Math.max(0.2,k0*f));
  state.view.ox = px - (k1/k0)*(px-state.view.ox);
  state.view.oy = py - (k1/k0)*(py-state.view.oy);
  state.view.k=k1; LS.set("view",state.view);
  positionHeatmap(); draw(); updateZoomReadout();
},{passive:false});
canvas.addEventListener("dblclick", e=>{ const [px,py]=localXY(e); if(!txAt(px,py)) addAPAt(...screenToWorld(px,py)); });

/* ---------- add / duplicate ------------------------------------------- */
const PALETTE=["#e30014","#a862a4","#ef7c00","#00963f","#008f95","#9d6830"]; // U1..U6
function nextColor(){ return PALETTE[state.txs.length % PALETTE.length]; }
function uniqueName(){
  let i=state.txs.length+1, n=`AP${i}`; const names=new Set(state.txs.map(t=>t.name));
  while(names.has(n)){ i++; n=`AP${i}`; } return n;
}
async function addAPAt(wx,wy){
  const name=uniqueName();
  const payload={name, position:[+wx.toFixed(2),+wy.toFixed(2),3.0], orientation:[0,0,0],
    tx_power_dbm:20, channel:36, bandwidth_mhz:20, enabled:true, color:nextColor(),
    antenna:{pattern:"iso"}};
  const res=await api.send("/api/transmitter","POST",payload);
  state.cacheStatus=res.cache_status; await refreshState(false); selectTx(name); maybeAutorun();
}
function addAPCenter(){ const [wx,wy]=screenToWorld(state._W/2, state._H/2); addAPAt(wx,wy); }

/* ---------- scene geometry + prediction mesh -------------------------- */
async function loadGeometryList(){
  try{
    const g=await api.get("/api/geometry"); state.geoFiles=g.files||[];
    const geoSel=el("g_geo"), meshSel=el("g_mesh");
    const opt=(v,t,sel)=>{ const o=document.createElement("option"); o.value=v; o.textContent=t;
      if(sel) o.selected=true; return o; };
    geoSel.innerHTML=""; meshSel.innerHTML="";
    geoSel.appendChild(opt("","(none)", !g.scene.geometry_file));
    meshSel.appendChild(opt("","(none)", !g.scene.mesh_file));
    state.geoFiles.forEach(f=>{
      const sel = (x)=> x && (f.path===x || f.file===x.split(/[\\/]/).pop());
      if(f.ext===".xml") geoSel.appendChild(opt(f.file, f.file, sel(g.scene.geometry_file)));
      else { geoSel.appendChild(opt(f.file, f.file, sel(g.scene.geometry_file)));
             meshSel.appendChild(opt(f.file, f.file, sel(g.scene.mesh_file))); }
    });
  }catch(e){ state.geoFiles=[]; }
}
async function fetchFootprint(){
  try{ state.footprint=await api.get("/api/footprint"); }catch(e){ state.footprint={geometry:[],mesh:[]}; }
  draw(); rebuild3D();
}
async function applyGeometry(){
  const body={ geometry_file: el("g_geo").value, mesh_file: el("g_mesh").value,
    grid_source: state.gridSource||"bbox",
    cell_size: +el("s_cell").value || state.grid.cell_size };
  const res=await api.send("/api/geometry","POST",body);
  if(res.error){ alert("Geometry error: "+res.error); return; }
  const g=res.geometry||{};
  const fitted=g.fitted;
  let hint = fitted ? ("fitted \u00b7 "+(g.grid_source==="mesh"?"mesh":"bbox")) : "loaded";
  if(g.footprint_segments!=null) hint += ` \u00b7 ${g.footprint_segments} edges`;
  el("geoFit").textContent = hint;
  if(g.warning){ alert("Geometry loaded, but: "+g.warning); }
  resetView();                        // recenter on the new extent
  await refreshState(true);
  await fetchFootprint();
  await fetchGeo3D();
  maybeAutorun();
}
function setGridSource(v){
  state.gridSource=v; LS.set("gridSource",v);
  document.querySelectorAll("#gridSource button").forEach(b=>
    b.classList.toggle("on", b.dataset.v===v));
}
async function clearGeometry(){
  await api.send("/api/geometry","POST",{geometry_file:"", mesh_file:""});
  state.footprint={geometry:[],mesh:[]};
  state.geo3d={geometry:[],mesh:[]};
  el("g_geo").value=""; el("g_mesh").value=""; el("geoFit").textContent="";
  await refreshState(true); maybeAutorun();
}

/* ---------- scene + engine settings ----------------------------------- */
function syncSceneInputs(){
  const g=state.grid;
  el("s_z").value=g.z; el("s_cell").value=g.cell_size;
  el("s_xmin").value=g.x_min; el("s_xmax").value=g.x_max;
  el("s_ymin").value=g.y_min; el("s_ymax").value=g.y_max;
}
async function applyScene(){
  const body={ grid:{z:+el("s_z").value, cell_size:+el("s_cell").value, x_min:+el("s_xmin").value,
      x_max:+el("s_xmax").value, y_min:+el("s_ymin").value, y_max:+el("s_ymax").value} };
  await api.send("/api/scene","POST",body); resetView(); await refreshState(true); maybeAutorun();
}
function syncEngineInputs(engine){
  const p=engine.params||{};
  if(p.max_depth!=null) el("e_depth").value=p.max_depth;
  if(p.samples_per_tx!=null) el("e_samples").value=String(p.samples_per_tx);
  el("e_refraction").checked=!!p.refraction; el("e_diffraction").checked=!!p.diffraction;
  el("solverApplies").textContent = p.applies? "active" : "Sionna engine only";
}
async function applyEngine(){
  const body={ max_depth:+el("e_depth").value, samples_per_tx:+el("e_samples").value,
    refraction:el("e_refraction").checked, diffraction:el("e_diffraction").checked };
  const s=await api.send("/api/engine","POST",body);
  renderCacheStats(s.cache); state.cacheStatus=s.cache_status; renderTxList(); maybeAutorun();
}

/* ---------- sidebar splitter ------------------------------------------ */
function initSplitter(){
  const sp=el("splitter"); let dragging=false;
  const saved=LS.get("asideW",null); if(saved) document.documentElement.style.setProperty("--aside-w", saved+"px");
  sp.addEventListener("mousedown", e=>{ dragging=true; sp.classList.add("active"); e.preventDefault(); });
  window.addEventListener("mousemove", e=>{
    if(!dragging) return;
    const w=Math.min(640, Math.max(260, window.innerWidth-e.clientX));
    document.documentElement.style.setProperty("--aside-w", w+"px");
    fitDisplay();
  });
  window.addEventListener("mouseup", ()=>{ if(dragging){ dragging=false; sp.classList.remove("active");
    const w=parseInt(getComputedStyle(document.documentElement).getPropertyValue("--aside-w")); LS.set("asideW",w); } });
}

/* ---------- routes: load CSVs, draw on map, plot profiles -------------- */
const ROUTE_COLORS={A:"#008f95",B:"#ef7c00"};

async function loadRoutes(){
  try{ state.routes=(await api.get("/api/routes")).routes||{}; }
  catch(e){ state.routes={}; }
  renderRouteSlots(); draw(); rebuild3D();
}
function renderRouteSlots(){
  for(const slot of ["A","B"]){
    const r=state.routes[slot];
    el("routeName"+slot).innerHTML = r
      ? `${r.name}<span class="meta">${r.n_points} pts \u00b7 ${r.length_m} m</span>`
      : `<span class="empty">no route ${slot}</span>`;
    el("routeClear"+slot).disabled=!r;
  }
  el("routePlot").disabled = !Object.keys(state.routes||{}).length;
}
async function uploadRoute(slot,file){
  const text=await file.text();
  const res=await api.send("/api/routes/"+slot,"POST",
    {csv:text, name:file.name.replace(/\.csv$/i,""), source:file.name});
  if(res.error){ alert("Route error: "+res.error); return; }
  state.routes=res.routes||{}; renderRouteSlots(); draw(); rebuild3D();
}
async function clearRoute(slot){
  const res=await api.send("/api/routes/"+slot,"DELETE");
  state.routes=res.routes||{}; renderRouteSlots(); draw(); rebuild3D();
  if(!Object.keys(state.routes).length) el("routePlotBox").style.display="none";
}
async function plotRoutes(){
  const btn=el("routePlot"); btn.disabled=true; btn.textContent="Sampling\u2026";
  try{
    const res=await api.send("/api/routes/profile","POST",{
      metric:state.metric,
      interval:+el("routeInterval").value||1,
      radius:Math.max(0,+el("routeRadius").value||0)});
    if(res.error){ alert("Profile error: "+res.error); return; }
    showRouteChart(res.profiles);
    el("routePng").disabled=false;
    renderCacheStats(res.cache); draw();
  } finally {
    btn.disabled=!Object.keys(state.routes||{}).length; btn.textContent="Plot profiles";
  }
}
async function exportRoutePng(){
  const res=await api.send("/api/routes/profile","POST",{
    metric:state.metric, render:true,
    interval:+el("routeInterval").value||1,
    radius:Math.max(0,+el("routeRadius").value||0)});
  if(res.error||!res.image){ alert("Export error: "+(res.error||"no image")); return; }
  const w=window.open("","_blank");
  if(w) w.document.write('<img src="'+res.image+'" style="width:100%">');
}
function drawRoutes(){
  for(const slot of Object.keys(state.routes||{})){
    const r=state.routes[slot]; if(!r||!r.points||!r.points.length) continue;
    const c=ROUTE_COLORS[slot]||"#5b6478";
    ctx.save();
    ctx.strokeStyle=c; ctx.fillStyle=c; ctx.lineWidth=2; ctx.setLineDash([6,4]);
    ctx.beginPath();
    r.points.forEach((p,i)=>{ const [px,py]=worldToScreen(p[0],p[1]);
      i?ctx.lineTo(px,py):ctx.moveTo(px,py); });
    ctx.stroke(); ctx.setLineDash([]);
    r.points.forEach(p=>{ const [px,py]=worldToScreen(p[0],p[1]);
      ctx.beginPath(); ctx.arc(px,py,3,0,2*Math.PI); ctx.fill(); });
    const [lx,ly]=worldToScreen(r.points[0][0],r.points[0][1]);
    ctx.font="700 11px system-ui,sans-serif";
    ctx.strokeStyle="rgba(255,255,255,.85)"; ctx.lineWidth=3;
    ctx.strokeText(slot,lx+6,ly-6); ctx.fillText(slot,lx+6,ly-6);
    ctx.restore();
  }
  /* cross-highlight: position hovered in the profile chart */
  if(state.routeHover){
    for(const h of state.routeHover){
      const [px,py]=worldToScreen(h.x,h.y);
      ctx.save();
      ctx.strokeStyle="rgba(255,255,255,.9)"; ctx.lineWidth=4.5;
      ctx.beginPath(); ctx.arc(px,py,7,0,2*Math.PI); ctx.stroke();
      ctx.strokeStyle=ROUTE_COLORS[h.slot]||"#5b6478"; ctx.lineWidth=2.5;
      ctx.beginPath(); ctx.arc(px,py,7,0,2*Math.PI); ctx.stroke();
      ctx.restore();
    }
  }
}

/* ---------- interactive route profile chart ---------------------------- */
const chart={ profiles:null, hidden:{}, zoom:null, hoverX:null, sel:null,
  dragged:false, legendRects:[], wired:false, area:null };

function showRouteChart(profiles){
  chart.profiles=profiles||[]; chart.zoom=null; chart.hoverX=null; chart.sel=null;
  el("routePlotBox").style.display="block";
  wireChart(); drawRouteChart();
}
function chartVisibleProfiles(){
  return (chart.profiles||[]).filter(p=>!chart.hidden[p.slot]);
}
function chartXRange(){
  if(chart.zoom) return chart.zoom;
  let m=1;
  (chart.profiles||[]).forEach(p=>{ m=Math.max(m,p.length_m||1); });
  return [0,m];
}
function nearestIdx(dist,x){                 // binary search on sorted distances
  let lo=0, hi=dist.length-1;
  while(hi-lo>1){ const mid=(lo+hi)>>1; (dist[mid]<x)?(lo=mid):(hi=mid); }
  return (x-dist[lo]<=dist[hi]-x)?lo:hi;
}
function drawRouteChart(){
  const cv=el("routeChart"); if(!cv||!chart.profiles) return;
  const cssW=cv.clientWidth||300, cssH=cv.clientHeight||250, dpr=state.dpr;
  cv.width=cssW*dpr; cv.height=cssH*dpr;
  const g=cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,cssW,cssH);
  const padL=46,padR=10,padT=26,padB=28;
  const pw=cssW-padL-padR, ph=cssH-padT-padB;
  chart.area={l:padL,t:padT,w:pw,h:ph};
  const [x0,x1]=chartXRange();
  const vis=chartVisibleProfiles();

  /* y range from visible bands inside the x range */
  let y0=Infinity, y1=-Infinity;
  vis.forEach(p=>{
    p.distance.forEach((d,i)=>{
      if(d<x0||d>x1) return;
      for(const v of [p.value[i],p.vmin[i],p.vmax[i]])
        if(v!=null && isFinite(v)){ y0=Math.min(y0,v); y1=Math.max(y1,v); }
    });
  });
  if(!isFinite(y0)){ y0=0; y1=1; }
  if(y1-y0<1e-9){ y0-=1; y1+=1; }
  const yPad=(y1-y0)*0.07; y0-=yPad; y1+=yPad;
  const X=d=>padL+(d-x0)/(x1-x0)*pw;
  const Y=v=>padT+(y1-v)/(y1-y0)*ph;

  /* grid + ticks */
  g.save();
  g.strokeStyle="rgba(0,0,80,.09)"; g.fillStyle="#5b6478";
  g.font="10px ui-monospace,monospace"; g.lineWidth=1;
  const sx=niceStep(x1-x0);
  for(let x=Math.ceil(x0/sx)*sx; x<=x1+1e-9; x+=sx){
    g.beginPath(); g.moveTo(X(x),padT); g.lineTo(X(x),padT+ph); g.stroke();
    g.textAlign="center"; g.fillText(x.toFixed(sx<1?1:0),X(x),padT+ph+13);
  }
  const sy=niceStep(y1-y0);
  for(let y=Math.ceil(y0/sy)*sy; y<=y1+1e-9; y+=sy){
    g.beginPath(); g.moveTo(padL,Y(y)); g.lineTo(padL+pw,Y(y)); g.stroke();
    g.textAlign="right"; g.fillText(y.toFixed(sy<1?1:0),padL-5,Y(y)+3);
  }
  g.strokeStyle="#d5dbe6";
  g.strokeRect(padL,padT,pw,ph);
  const unit=(chart.profiles[0]||{}).unit||"", label=(chart.profiles[0]||{}).label||"";
  g.textAlign="left"; g.fillText(`${label}${unit?" ["+unit+"]":""}`,padL,padT-14+8);
  g.textAlign="right"; g.fillText("distance [m]",padL+pw,padT+ph+25);
  g.restore();

  /* clip to plot area for series */
  g.save(); g.beginPath(); g.rect(padL,padT,pw,ph); g.clip();
  vis.forEach(p=>{
    const c=ROUTE_COLORS[p.slot]||"#5b6478";
    /* min-max band, per run of consecutive non-null samples */
    g.fillStyle=c+"29";
    let run=[];
    const flush=()=>{ if(run.length>1){
      g.beginPath();
      run.forEach(([d,,hi],i)=> i?g.lineTo(X(d),Y(hi)):g.moveTo(X(d),Y(hi)));
      for(let i=run.length-1;i>=0;i--) g.lineTo(X(run[i][0]),Y(run[i][1]));
      g.closePath(); g.fill(); } run=[]; };
    p.distance.forEach((d,i)=>{
      (p.vmin[i]!=null&&p.vmax[i]!=null)?run.push([d,p.vmin[i],p.vmax[i]]):flush(); });
    flush();
    /* mean line */
    g.strokeStyle=c; g.lineWidth=1.8; g.beginPath();
    let pen=false;
    p.distance.forEach((d,i)=>{
      const v=p.value[i];
      if(v==null){ pen=false; return; }
      pen?g.lineTo(X(d),Y(v)):g.moveTo(X(d),Y(v)); pen=true;
    });
    g.stroke();
  });

  /* hover crosshair + dots */
  const tips=[];
  if(chart.hoverX!=null && chart.hoverX>=padL && chart.hoverX<=padL+pw){
    const wx=x0+(chart.hoverX-padL)/pw*(x1-x0);
    g.strokeStyle="rgba(0,0,80,.35)"; g.setLineDash([3,3]); g.lineWidth=1;
    g.beginPath(); g.moveTo(chart.hoverX,padT); g.lineTo(chart.hoverX,padT+ph); g.stroke();
    g.setLineDash([]);
    vis.forEach(p=>{
      const i=nearestIdx(p.distance,wx), v=p.value[i];
      if(v==null) return;
      const c=ROUTE_COLORS[p.slot]||"#5b6478";
      g.fillStyle=c; g.strokeStyle="#fff"; g.lineWidth=1.5;
      g.beginPath(); g.arc(X(p.distance[i]),Y(v),3.5,0,2*Math.PI); g.fill(); g.stroke();
      tips.push({slot:p.slot,name:p.name,d:p.distance[i],v,
        lo:p.vmin[i],hi:p.vmax[i],c});
    });
  }
  g.restore();

  /* selection rectangle (drag-zoom) */
  if(chart.sel){
    g.save();
    g.fillStyle="rgba(227,0,20,.08)"; g.strokeStyle="rgba(227,0,20,.5)";
    const a=Math.min(chart.sel.x0,chart.sel.x1), b=Math.max(chart.sel.x0,chart.sel.x1);
    g.fillRect(a,padT,b-a,ph); g.strokeRect(a,padT,b-a,ph);
    g.restore();
  }

  /* legend (click to toggle) */
  chart.legendRects=[];
  g.save(); g.font="600 11px system-ui,sans-serif";
  let lx=padL;
  (chart.profiles||[]).forEach(p=>{
    const c=ROUTE_COLORS[p.slot]||"#5b6478", off=!!chart.hidden[p.slot];
    const txt=`${p.slot}: ${p.name}`;
    const w=g.measureText(txt).width+22;
    g.globalAlpha=off?0.35:1;
    g.fillStyle=c; g.fillRect(lx,8,10,10);
    g.fillStyle="#000050"; g.fillText(txt,lx+14,17);
    if(off){ g.strokeStyle="#5b6478"; g.beginPath();
      g.moveTo(lx+12,13); g.lineTo(lx+w-8,13); g.stroke(); }
    chart.legendRects.push({x:lx,y:4,w,h:16,slot:p.slot});
    lx+=w+10;
  });
  g.restore();

  /* tooltip */
  if(tips.length){
    g.save(); g.font="11px ui-monospace,monospace";
    const lines=[`d = ${tips[0].d.toFixed(1)} m`].concat(tips.map(t=>
      `${t.slot} ${t.v.toFixed(1)}${unit?" "+unit:""}`+
      ((t.lo!=null&&t.hi!=null)?`  (${t.lo.toFixed(1)}\u2026${t.hi.toFixed(1)})`:"")));
    const bw=Math.max(...lines.map(s=>g.measureText(s).width))+14, bh=lines.length*15+8;
    let tx=chart.hoverX+10, ty=padT+8;
    if(tx+bw>cssW-4) tx=chart.hoverX-10-bw;
    g.fillStyle="rgba(0,0,80,.92)"; g.strokeStyle="#e30014";
    g.beginPath();
    (g.roundRect?g.roundRect(tx,ty,bw,bh,6):g.rect(tx,ty,bw,bh));
    g.fill(); g.stroke();
    lines.forEach((s,i)=>{
      g.fillStyle=(i===0)?"#ffd0d4":"#fff";
      g.fillText(s,tx+10,ty+14+i*15);
    });
    /* colour swatches per route line */
    tips.forEach((t,i)=>{ g.fillStyle=t.c; g.fillRect(tx+4,ty+14+(i+1)*15-9,3,10); });
    g.restore();
  }
}
function chartLocalX(e){
  const r=el("routeChart").getBoundingClientRect(); return e.clientX-r.left;
}
function updateMapHoverFromChart(){
  if(chart.hoverX==null||!chart.area){ state.routeHover=null; draw(); return; }
  const [x0,x1]=chartXRange(), a=chart.area;
  const wx=x0+(chart.hoverX-a.l)/a.w*(x1-x0);
  const hov=[];
  chartVisibleProfiles().forEach(p=>{
    const i=nearestIdx(p.distance,wx);
    if(p.value[i]==null||!p.samples||!p.samples[i]) return;
    hov.push({slot:p.slot,x:p.samples[i][0],y:p.samples[i][1]});
  });
  state.routeHover=hov.length?hov:null; draw();
}
function wireChart(){
  if(chart.wired) return; chart.wired=true;
  const cv=el("routeChart");
  cv.addEventListener("mousemove",e=>{
    const x=chartLocalX(e);
    if(chart.sel){ chart.sel.x1=x; chart.dragged=true; }
    chart.hoverX=x; drawRouteChart(); updateMapHoverFromChart();
  });
  cv.addEventListener("mouseleave",()=>{
    chart.hoverX=null; chart.sel=null; drawRouteChart();
    state.routeHover=null; draw();
  });
  cv.addEventListener("mousedown",e=>{
    const x=chartLocalX(e);
    if(chart.area && x>=chart.area.l) { chart.sel={x0:x,x1:x}; chart.dragged=false; }
  });
  window.addEventListener("mouseup",()=>{
    if(!chart.sel) return;
    const a=Math.min(chart.sel.x0,chart.sel.x1), b=Math.max(chart.sel.x0,chart.sel.x1);
    const ar=chart.area, [x0,x1]=chartXRange();
    chart.sel=null;
    if(b-a>6 && ar){
      const w0=x0+(Math.max(a,ar.l)-ar.l)/ar.w*(x1-x0);
      const w1=x0+(Math.min(b,ar.l+ar.w)-ar.l)/ar.w*(x1-x0);
      if(w1-w0>1e-6) chart.zoom=[w0,w1];
    }
    drawRouteChart();
  });
  cv.addEventListener("dblclick",()=>{ chart.zoom=null; drawRouteChart(); });
  cv.addEventListener("click",e=>{
    if(chart.dragged){ chart.dragged=false; return; }
    const r=cv.getBoundingClientRect(), x=e.clientX-r.left, y=e.clientY-r.top;
    for(const lr of chart.legendRects){
      if(x>=lr.x&&x<=lr.x+lr.w&&y>=lr.y&&y<=lr.y+lr.h){
        chart.hidden[lr.slot]=!chart.hidden[lr.slot];
        drawRouteChart(); break;
      }
    }
  });
}

/* ---------- 3D view (three.js, lazy-loaded from CDN) ------------------- */
const t3={ ready:false, renderer:null, scene:null, camera:null, raf:0,
  sph:{r:80, theta:-Math.PI/3, phi:Math.PI/3.2}, target:{x:0,y:0,z:0},
  dyn:null };

function loadThree(){
  if(window.THREE) return Promise.resolve();
  return new Promise((res,rej)=>{
    const s=document.createElement("script");
    s.src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
    s.onload=res;
    s.onerror=()=>rej(new Error("could not load three.js - the 3D view needs internet access to the CDN"));
    document.head.appendChild(s);
  });
}
async function setViewMode(v){
  document.querySelectorAll("#viewMode button").forEach(b=>
    b.classList.toggle("on", b.dataset.v===v));
  const box=el("display3d");
  if(v==="3d"){
    try{ await loadThree(); }catch(e){ alert(e.message); setViewMode("2d"); return; }
    state.viewMode="3d"; LS.set("viewMode","3d");
    display.classList.add("mode3d"); box.classList.add("on");
    init3D(); resize3D(); rebuild3D(); start3D();
  } else {
    state.viewMode="2d"; LS.set("viewMode","2d");
    display.classList.remove("mode3d"); box.classList.remove("on");
    stop3D();
  }
}
function gridCenter(){ const g=state.grid;
  return {x:(g.x_min+g.x_max)/2, y:(g.y_min+g.y_max)/2, z:0}; }
function fit3D(){
  if(!t3.ready||!state.grid) return;
  const g=state.grid;
  t3.target=gridCenter();
  t3.sph.r=1.15*Math.max(g.x_max-g.x_min, g.y_max-g.y_min, 10);
}
function init3D(){
  if(t3.ready) return;
  const box=el("display3d");
  t3.renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  t3.renderer.setPixelRatio(state.dpr);
  box.appendChild(t3.renderer.domElement);
  t3.scene=new THREE.Scene();
  t3.camera=new THREE.PerspectiveCamera(50,1,0.1,10000);
  t3.camera.up.set(0,0,1);                          // world is z-up
  t3.scene.add(new THREE.HemisphereLight(0xffffff,0x99a3b8,0.95));
  const dl=new THREE.DirectionalLight(0xffffff,0.45);
  dl.position.set(1,-1,2); t3.scene.add(dl);
  t3.dyn=new THREE.Group(); t3.scene.add(t3.dyn);   // rebuilt on every change
  t3.ready=true;
  fit3D();

  /* orbit controls: drag = orbit, shift/right-drag = pan, wheel = zoom */
  let drag=null;
  box.addEventListener("mousedown",e=>{
    drag={x:e.clientX,y:e.clientY,pan:(e.button===2||e.shiftKey)};
  });
  box.addEventListener("contextmenu",e=>e.preventDefault());
  window.addEventListener("mousemove",e=>{
    if(!drag) return;
    const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
    drag.x=e.clientX; drag.y=e.clientY;
    if(drag.pan){
      const s=t3.sph.r*0.0016, th=t3.sph.theta;
      const rt={x:-Math.sin(th),y:Math.cos(th)};    // camera right (world XY)
      const fw={x:-Math.cos(th),y:-Math.sin(th)};   // camera forward (world XY)
      t3.target.x -= dx*s*rt.x - dy*s*fw.x;
      t3.target.y -= dx*s*rt.y - dy*s*fw.y;
    } else {
      t3.sph.theta -= dx*0.005;
      t3.sph.phi   = Math.min(1.52, Math.max(0.08, t3.sph.phi - dy*0.005));
    }
  });
  window.addEventListener("mouseup",()=>{ drag=null; });
  box.addEventListener("wheel",e=>{
    e.preventDefault();
    t3.sph.r=Math.min(5000,Math.max(2,t3.sph.r*Math.exp(e.deltaY*0.0012)));
  },{passive:false});
}
function resize3D(){
  if(!t3.ready) return;
  const r=display.getBoundingClientRect();
  t3.renderer.setSize(r.width,r.height,false);
  t3.camera.aspect=r.width/Math.max(1,r.height);
  t3.camera.updateProjectionMatrix();
}
function start3D(){
  cancelAnimationFrame(t3.raf);
  const tick=()=>{
    const s=t3.sph, t=t3.target;
    t3.camera.position.set(
      t.x+s.r*Math.sin(s.phi)*Math.cos(s.theta),
      t.y+s.r*Math.sin(s.phi)*Math.sin(s.theta),
      t.z+s.r*Math.cos(s.phi));
    t3.camera.lookAt(t.x,t.y,t.z);
    t3.renderer.render(t3.scene,t3.camera);
    t3.raf=requestAnimationFrame(tick);
  };
  tick();
}
function stop3D(){ cancelAnimationFrame(t3.raf); }

function makeLabel(text,color){
  const c=document.createElement("canvas"); c.width=256; c.height=64;
  const g=c.getContext("2d");
  g.font="700 34px system-ui,sans-serif";
  g.strokeStyle="rgba(255,255,255,.9)"; g.lineWidth=8; g.strokeText(text,6,44);
  g.fillStyle=color||"#000050"; g.fillText(text,6,44);
  const tex=new THREE.CanvasTexture(c);
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,depthTest:false}));
  sp.scale.set(8,2,1);
  return sp;
}
function addSegs3D(group,segs,color,z){
  if(!segs||!segs.length) return;
  const pos=new Float32Array(segs.length*6);
  segs.forEach((s,i)=>{ pos.set([s[0],s[1],z, s[2],s[3],z], i*6); });
  const geo=new THREE.BufferGeometry();
  geo.setAttribute("position",new THREE.BufferAttribute(pos,3));
  group.add(new THREE.LineSegments(geo,
    new THREE.LineBasicMaterial({color, transparent:true, opacity:0.55})));
}
async function fetchGeo3D(){
  try{ state.geo3d=await api.get("/api/geometry3d"); }
  catch(e){ state.geo3d={geometry:[],mesh:[]}; }
  if(state.geo3d.geometry_error) console.warn("geometry3d:",state.geo3d.geometry_error);
  if(state.geo3d.mesh_error) console.warn("geometry3d:",state.geo3d.mesh_error);
  rebuild3D();
}
function addMesh3D(m,color,opacity){
  if(!m||!m.vertices||!m.vertices.length) return;
  const geo=new THREE.BufferGeometry();
  geo.setAttribute("position",
    new THREE.BufferAttribute(new Float32Array(m.vertices),3));
  geo.setIndex(m.faces);
  geo.computeVertexNormals();
  t3.dyn.add(new THREE.Mesh(geo,new THREE.MeshLambertMaterial({
    color, transparent:true, opacity, side:THREE.DoubleSide, depthWrite:false})));
  const edges=new THREE.EdgesGeometry(geo,12);      // crease edges for structure
  t3.dyn.add(new THREE.LineSegments(edges,
    new THREE.LineBasicMaterial({color, transparent:true,
      opacity:Math.min(1,opacity+0.35)})));
}
function rebuild3D(){
  if(!t3.ready||state.viewMode!=="3d"||!state.grid) return;
  const g=state.grid;
  /* clear the dynamic group */
  while(t3.dyn.children.length){
    const o=t3.dyn.children.pop();
    o.traverse?.(n=>{ n.geometry?.dispose?.(); n.material?.map?.dispose?.(); n.material?.dispose?.(); });
  }
  const cx=(g.x_min+g.x_max)/2, cy=(g.y_min+g.y_max)/2;
  const w=g.x_max-g.x_min, h=g.y_max-g.y_min;

  /* coverage plane (heatmap texture) at the measurement height grid.z */
  const mat = heatmap.src
    ? new THREE.MeshBasicMaterial({map:new THREE.TextureLoader().load(heatmap.src),
        transparent:true, opacity:0.92, side:THREE.DoubleSide})
    : new THREE.MeshBasicMaterial({color:0xdde3ec, transparent:true, opacity:0.55,
        side:THREE.DoubleSide});
  const plane=new THREE.Mesh(new THREE.PlaneGeometry(w,h),mat);
  plane.position.set(cx,cy,g.z);
  t3.dyn.add(plane);

  /* ground reference grid at z=0 */
  const gh=new THREE.GridHelper(Math.max(w,h),20,0x8090b0,0xc5cddd);
  gh.rotation.x=Math.PI/2; gh.position.set(cx,cy,0);
  t3.dyn.add(gh);

  /* scene geometry & prediction mesh: real 3D triangle meshes when the
     server could extract them, otherwise the flat XY footprints at z=0 */
  const g3=state.geo3d||{};
  if((g3.geometry||[]).length)
    g3.geometry.forEach(m=>addMesh3D(m,0x000050,0.22));
  else
    addSegs3D(t3.dyn, state.footprint.geometry, 0x000050, 0.02);
  if((g3.mesh||[]).length)
    g3.mesh.forEach(m=>addMesh3D(m,0xe30014,0.15));
  else
    addSegs3D(t3.dyn, state.footprint.mesh, 0xe30014, 0.04);

  /* transmitters (respect the visibility checkboxes) */
  state.txs.filter(t=>txVisible(t.name)).forEach(tx=>{
    const [x,y,z]=tx.position, col=new THREE.Color(tx.enabled?tx.color:"#9aa3b2");
    const grp=new THREE.Group();
    const pole=new THREE.Mesh(new THREE.CylinderGeometry(0.08,0.08,z,8),
      new THREE.MeshLambertMaterial({color:0x5b6478}));
    pole.rotation.x=Math.PI/2; pole.position.set(x,y,z/2); grp.add(pole);
    const head=new THREE.Mesh(new THREE.SphereGeometry(Math.max(0.35,w/120),16,12),
      new THREE.MeshLambertMaterial({color:col}));
    head.position.set(x,y,z); grp.add(head);
    const directional = tx.antenna && tx.antenna.pattern!=="iso" && tx.antenna.pattern!=="dipole";
    if(directional){
      const yaw=(tx.orientation?.[0]||0)*Math.PI/180, len=Math.max(2,w/18);
      const cone=new THREE.Mesh(new THREE.ConeGeometry(len/3.2,len,20,1,true),
        new THREE.MeshBasicMaterial({color:col, transparent:true, opacity:0.28,
          side:THREE.DoubleSide}));
      cone.rotation.z=Math.PI/2;                     // cone +y -> +x (boresight)
      const cg=new THREE.Group(); cg.add(cone);
      cone.position.set(len/2,0,0);
      cg.position.set(x,y,z); cg.rotation.z=yaw;
      grp.add(cg);
    }
    const lbl=makeLabel(tx.name,"#000050");
    lbl.position.set(x,y,z+2); grp.add(lbl);
    t3.dyn.add(grp);
  });

  /* routes (their real 3D z) */
  for(const slot of Object.keys(state.routes||{})){
    const r=state.routes[slot]; if(!r||!r.points||r.points.length<2) continue;
    const geo=new THREE.BufferGeometry().setFromPoints(
      r.points.map(p=>new THREE.Vector3(p[0],p[1],p[2])));
    t3.dyn.add(new THREE.Line(geo,
      new THREE.LineBasicMaterial({color:ROUTE_COLORS[slot]||"#5b6478"})));
    const lbl=makeLabel("route "+slot, ROUTE_COLORS[slot]);
    lbl.position.set(r.points[0][0],r.points[0][1],r.points[0][2]+1.5);
    t3.dyn.add(lbl);
  }
}

/* ---------- init ------------------------------------------------------ */
async function init(){
  state.options=await api.get("/api/options");
  try{ state.patterns=(await api.get("/api/patterns")).patterns||[]; }catch(e){ state.patterns=[]; }
  el("metric").innerHTML="";
  state.options.metrics.forEach(m=>{ const o=document.createElement("option");
    o.value=m.key; o.textContent=m.label; el("metric").appendChild(o); });
  el("metric").value=state.metric;

  el("runBtn").onclick=run;
  el("addBtn").onclick=addAPCenter;
  el("resetView").onclick=()=>{ resetView(); };
  el("metric").onchange=()=>{ state.metric=el("metric").value; run(); };
  el("clearCache").onclick=async()=>{ const r=await api.send("/api/cache/clear","POST");
    renderCacheStats(r.cache); await refreshState(false); };
  el("applyScene").onclick=applyScene;
  el("applyEngine").onclick=applyEngine;
  el("applyGeo").onclick=applyGeometry;
  el("clearGeo").onclick=clearGeometry;
  state.gridSource=LS.get("gridSource","bbox");
  document.querySelectorAll("#gridSource button").forEach(b=>
    b.onclick=()=>{ setGridSource(b.dataset.v); });
  setGridSource(state.gridSource);
  el("delBtn").onclick=async()=>{ if(!state.selected) return;
    await api.send("/api/transmitter/"+encodeURIComponent(state.selected),"DELETE");
    state.selected=null; el("editorCard").style.display="none";
    await refreshState(true); maybeAutorun(); };
  el("dupBtn").onclick=async()=>{ const t=state.txs.find(x=>x.name===state.selected); if(!t) return;
    const name=uniqueName();
    const copy={...JSON.parse(JSON.stringify(t)), name,
      position:[t.position[0]+3,t.position[1]+3,t.position[2]], color:nextColor()};
    await api.send("/api/transmitter","POST",copy);
    await refreshState(false); selectTx(name); maybeAutorun(); };
  ["f_name","f_x","f_y","f_z","f_pwr","f_az","f_gain","f_hpbw","f_ch","f_bw","f_ant"].forEach(id=>
    el(id).addEventListener("change", commitEditor));

  /* view mode toggle */
  document.querySelectorAll("#viewMode button").forEach(b=>
    b.onclick=()=>setViewMode(b.dataset.v));

  /* route profiles */
  ["A","B"].forEach(slot=>{
    el("routeLoad"+slot).onclick=()=>el("routeFile"+slot).click();
    el("routeFile"+slot).onchange=e=>{
      const f=e.target.files[0]; if(f) uploadRoute(slot,f); e.target.value=""; };
    el("routeClear"+slot).onclick=()=>clearRoute(slot);
  });
  el("routePlot").onclick=plotRoutes;
  el("routeInterval").value=LS.get("routeInterval",1);
  el("routeRadius").value=LS.get("routeRadius",0.5);
  el("routeInterval").onchange=()=>LS.set("routeInterval",+el("routeInterval").value);
  el("routeRadius").onchange=()=>LS.set("routeRadius",+el("routeRadius").value);
  el("routePng").onclick=exportRoutePng;

  initSplitter();
  if(window.ResizeObserver) new ResizeObserver(()=>fitDisplay()).observe(el("display"));
  window.addEventListener("resize", fitDisplay);

  await refreshState(true); updateZoomReadout();
  await loadGeometryList(); await fetchFootprint(); await fetchGeo3D();
  await loadRoutes();
  if(LS.get("viewMode","2d")==="3d") setViewMode("3d");
  run();
}
init();
