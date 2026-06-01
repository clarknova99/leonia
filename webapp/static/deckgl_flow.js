/* Reusable deck.gl traffic-flow renderer for the stakeholder page.
 *
 * Replaces the old folium/Leaflet iframe with a single WebGL map that
 * is created once and re-fed a new scenario dataset on every dropdown
 * change. The dataset schema (flow.json) is produced server-side by
 * leonia_traffic.sumo.visualizations.build_flow_payload:
 *
 *   { meta:  { title, frame_minutes, vmax_vph, center:[lng,lat], zoom,
 *              n_active_edges, n_frames },
 *     frames:   ["00:00", "00:15", ...],          // clock labels
 *     skeleton: [ [[lng,lat], ...], ... ],         // grey context
 *     edges:    [ { id, name, coords:[[lng,lat],...], vph:[int,...] } ] }
 *
 * Usage:
 *   const ctl = LeoniaDeckFlow.create(containerEl);
 *   ctl.update(data);            // swap scenario
 *   ctl.setHighlight(edgeIds);   // outline the picked street
 *   ctl.destroy();
 *
 * Assumes maplibre-gl and deck.gl UMD bundles are already on the page.
 */
(function () {
  "use strict";

  const RAMP = [
    [0.0, [34, 197, 94]],
    [0.25, [234, 179, 8]],
    [0.5, [249, 115, 22]],
    [1.0, [239, 68, 68]],
  ];

  function rampColor(v, vmax) {
    const t = Math.max(0, Math.min(1, v / vmax));
    for (let i = 1; i < RAMP.length; i++) {
      if (t <= RAMP[i][0]) {
        const [t0, c0] = RAMP[i - 1];
        const [t1, c1] = RAMP[i];
        const f = (t - t0) / (t1 - t0 || 1);
        return c0.map((c, j) => Math.round(c + (c1[j] - c) * f));
      }
    }
    return RAMP[RAMP.length - 1][1];
  }

  const LOOP = 100; // animation loop length (arbitrary time units)
  const FRAME_MS = 800; // wall-clock ms per 15-min frame at 1x

  // CARTO basemap styles (open-source MapLibre style JSON, no token).
  const THEMES = [
    {
      id: "dark",
      label: "Dark Matter",
      url: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    },
    {
      id: "light",
      label: "Positron (light)",
      url: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    },
    {
      id: "voyager",
      label: "Voyager",
      url: "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    },
    {
      id: "dark-nolabels",
      label: "Dark, no labels",
      url:
        "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json",
    },
  ];

  function buildTrips(data, frame, vmax) {
    const trips = [];
    for (const e of data.edges) {
      const v = e.vph[frame] || 0;
      if (v <= 0 || e.coords.length < 2) continue;
      const ts = [0];
      for (let i = 1; i < e.coords.length; i++) {
        const [x0, y0] = e.coords[i - 1];
        const [x1, y1] = e.coords[i];
        ts.push(ts[i - 1] + Math.hypot(x1 - x0, y1 - y0));
      }
      const total = ts[ts.length - 1] || 1;
      const slice = LOOP * (0.55 - 0.3 * Math.min(1, v / vmax));
      const start = Math.random() * LOOP;
      trips.push({
        path: e.coords,
        timestamps: ts.map((t) => start + (t / total) * slice),
        color: rampColor(v, vmax),
      });
    }
    return trips;
  }

  function buildPanel(container) {
    const panel = document.createElement("div");
    panel.className = "deck-panel";
    const themeOpts = THEMES.map(
      (t) => `<option value="${t.id}">${t.label}</option>`,
    ).join("");
    panel.innerHTML = [
      '<div class="deck-clock"><span class="deck-time">--:--</span>',
      '<small class="deck-stat"></small></div>',
      '<div class="deck-row">',
      '<button type="button" class="deck-play">&#10073;&#10073; Pause</button>',
      '<input type="range" class="deck-scrub" min="0" max="0" value="0" step="1" />',
      '<button type="button" class="deck-speed">1&times;</button>',
      "</div>",
      '<div class="deck-row deck-row-theme">',
      '<span class="deck-theme-label">Basemap</span>',
      `<select class="deck-theme">${themeOpts}</select>`,
      "</div>",
      '<div class="deck-row deck-row-impact">',
      '<button type="button" class="deck-impact">Impact vs baseline</button>',
      "</div>",
      '<div class="deck-impact-key" hidden>',
      '<span><i class="deck-sw deck-sw-up"></i> more than baseline</span>',
      '<span><i class="deck-sw deck-sw-down"></i> less than baseline</span>',
      "</div>",
    ].join("");
    container.appendChild(panel);
    const tip = document.createElement("div");
    tip.className = "deck-tip";
    container.appendChild(tip);
    return {
      panel,
      tip,
      time: panel.querySelector(".deck-time"),
      stat: panel.querySelector(".deck-stat"),
      play: panel.querySelector(".deck-play"),
      impact: panel.querySelector(".deck-impact"),
      impactKey: panel.querySelector(".deck-impact-key"),
      scrub: panel.querySelector(".deck-scrub"),
      speed: panel.querySelector(".deck-speed"),
      theme: panel.querySelector(".deck-theme"),
    };
  }

  function create(container, opts) {
    opts = opts || {};
    container.classList.add("deck-flow-host");

    const map = new maplibregl.Map({
      container,
      style: opts.style || THEMES[0].url,
      center: opts.center || [-73.99, 40.86],
      zoom: opts.zoom || 13,
      pitch: 0,
      bearing: 0,
      antialias: true,
      attributionControl: true,
    });
    map.addControl(
      new maplibregl.NavigationControl({ visualizePitch: true }),
      "bottom-right",
    );

    const { MapboxOverlay, PathLayer } = deck;
    const TripsLayer = deck.TripsLayer;
    const TextLayer = deck.TextLayer;
    const CollisionFilterExtension = deck.CollisionFilterExtension;
    // Overlaid (non-interleaved): deck renders into its own canvas above
    // the basemap, so swapping the basemap style with setStyle() doesn't
    // drop the traffic layers.
    const overlay = new MapboxOverlay({
      interleaved: false,
      pickingRadius: 6,
      layers: [],
    });

    const ui = buildPanel(container);

    // --- mutable render state -------------------------------------------
    let data = null;
    let vmax = 600;
    let frame = 0;
    let playing = true;
    let lastAdvance = 0;
    let time = 0;
    let tripsCache = [];
    let highlightIds = new Set();
    let hover = null; // { edge, x, y }
    let pinned = null; // { edge, x, y }
    let centred = false;
    let labelsOn = false; // street-name labels removed for now
    let impactOn = false;
    let labelData = []; // [{ name, position:[lng,lat], priority }]

    const SPEEDS = [1, 2, 4, 8];
    let speedIdx = 0;

    // A street counts as "impacted" once its vph differs from baseline
    // by at least this much in the current frame. Glow magnitude scales
    // up to ~60% of the colour scale's vmax.
    const IMPACT_THRESH = 20;

    // One label per named street, placed at the midpoint of that
    // street's longest segment. Priority (busier streets win collisions)
    // comes from the peak vph across the day so the labels that survive
    // crowding are the ones that matter most.
    function buildLabels(d) {
      const byName = new Map();
      for (const e of d.edges) {
        if (!e.name || e.name === e.id) continue;
        let len = 0;
        for (let i = 1; i < e.coords.length; i++) {
          const [x0, y0] = e.coords[i - 1];
          const [x1, y1] = e.coords[i];
          len += Math.hypot(x1 - x0, y1 - y0);
        }
        const peak = e.vph.reduce((m, v) => (v > m ? v : m), 0);
        const cur = byName.get(e.name);
        if (!cur || len > cur.len) {
          byName.set(e.name, { edge: e, len, peak });
        } else if (peak > cur.peak) {
          cur.peak = peak;
        }
      }
      const out = [];
      for (const [name, info] of byName) {
        const c = info.edge.coords;
        if (c.length < 2) continue;
        // walk to the halfway point along the chosen segment
        let half = info.len / 2;
        let pos = c[0];
        for (let i = 1; i < c.length; i++) {
          const [x0, y0] = c[i - 1];
          const [x1, y1] = c[i];
          const seg = Math.hypot(x1 - x0, y1 - y0);
          if (seg >= half) {
            const f = seg ? half / seg : 0;
            pos = [x0 + (x1 - x0) * f, y0 + (y1 - y0) * f];
            break;
          }
          half -= seg;
        }
        out.push({ name, position: pos, priority: info.peak });
      }
      return out;
    }

    function renderTip() {
      const sel = pinned || hover;
      if (!sel || !data) {
        ui.tip.style.display = "none";
        return;
      }
      const vph = sel.edge.vph[frame] || 0;
      ui.tip.style.display = "block";
      ui.tip.style.left = sel.x + "px";
      ui.tip.style.top = sel.y + "px";
      let deltaHtml = "";
      if (sel.edge.base) {
        const base = sel.edge.base[frame] || 0;
        const d = vph - base;
        const sign = d > 0 ? "+" : "";
        const cls = d > 0 ? "deck-tip-up" : d < 0 ? "deck-tip-down" : "";
        deltaHtml =
          '<div class="deck-tip-delta ' +
          cls +
          '">' +
          sign +
          d +
          " vph vs baseline (" +
          base +
          " &rarr; " +
          vph +
          ")</div>";
      }
      ui.tip.innerHTML =
        '<div class="deck-tip-name">' +
        sel.edge.name +
        "</div>" +
        '<div class="deck-tip-vph"><b>' +
        vph +
        "</b> vph &middot; " +
        (data.frames[frame] || "") +
        "</div>" +
        deltaHtml +
        '<div class="deck-tip-pin">' +
        (pinned ? "pinned — click empty space to clear" : "click to pin") +
        "</div>";
    }

    function skeletonLayer() {
      return new PathLayer({
        id: "skeleton",
        data: data.skeleton,
        getPath: (d) => d,
        getColor: [70, 80, 100, 120],
        getWidth: 0.7,
        widthUnits: "pixels",
        widthMinPixels: 0.6,
        parameters: { depthTest: false },
      });
    }

    function highlightLayer() {
      if (!highlightIds.size) return null;
      const sel = data.edges.filter((e) => highlightIds.has(e.id));
      if (!sel.length) return null;
      return new PathLayer({
        id: "highlight",
        data: sel,
        getPath: (e) => e.coords,
        getColor: [255, 255, 255, 235],
        getWidth: 11,
        widthUnits: "pixels",
        widthMinPixels: 7,
        capRounded: true,
        jointRounded: true,
        opacity: 0.85,
        parameters: { depthTest: false },
      });
    }

    function pathLayer() {
      return new PathLayer({
        id: "links",
        data: data.edges,
        getPath: (e) => e.coords,
        getColor: (e) => rampColor(e.vph[frame] || 0, vmax),
        getWidth: (e) => 1.5 + 5 * Math.min(1, (e.vph[frame] || 0) / vmax),
        widthUnits: "pixels",
        widthMinPixels: 1.2,
        capRounded: true,
        jointRounded: true,
        opacity: 0.6,
        parameters: { depthTest: false },
        pickable: true,
        onHover: (info) => {
          hover = info.object
            ? { edge: info.object, x: info.x, y: info.y }
            : null;
          renderTip();
        },
        onClick: (info) => {
          pinned = info.object
            ? { edge: info.object, x: info.x, y: info.y }
            : null;
          renderTip();
        },
        updateTriggers: { getColor: frame, getWidth: frame },
      });
    }

    function tripsLayer() {
      return new TripsLayer({
        id: "trips",
        data: tripsCache,
        getPath: (d) => d.path,
        getTimestamps: (d) => d.timestamps,
        getColor: (d) => d.color,
        getWidth: 3,
        widthUnits: "pixels",
        widthMinPixels: 2,
        capRounded: true,
        jointRounded: true,
        opacity: 0.9,
        trailLength: LOOP * 0.18,
        currentTime: time,
        fadeTrail: true,
        parameters: { depthTest: false },
      });
    }

    function labelLayer() {
      if (!labelsOn || !TextLayer || !labelData.length) return null;
      const ext = CollisionFilterExtension
        ? [new CollisionFilterExtension()]
        : [];
      return new TextLayer({
        id: "labels",
        data: labelData,
        getText: (d) => d.name,
        getPosition: (d) => d.position,
        getSize: 13,
        sizeUnits: "pixels",
        getColor: [255, 255, 255, 255],
        fontFamily: "system-ui, -apple-system, sans-serif",
        fontWeight: 700,
        // SDF + outline keeps text crisp and legible over any basemap.
        fontSettings: { sdf: true, radius: 12 },
        outlineWidth: 3,
        outlineColor: [0, 0, 0, 255],
        background: true,
        getBackgroundColor: [10, 15, 26, 170],
        backgroundPadding: [4, 2],
        billboard: true,
        getAngle: 0,
        characterSet: "auto",
        parameters: { depthTest: false },
        // Thin out overlapping labels, keeping the busiest streets.
        collisionEnabled: ext.length > 0,
        collisionGroup: "street-labels",
        getCollisionPriority: (d) => d.priority,
        extensions: ext,
      });
    }

    function hasBaseline() {
      return !!(data && data.meta && data.meta.has_baseline);
    }

    // Glowing halo on streets whose flow differs from the baseline
    // (unchanged network) in the current frame: orange where the
    // scenario added traffic (e.g. spillover from a closed street),
    // blue where it removed traffic. Pulses for attention.
    function impactLayer() {
      if (!impactOn || !hasBaseline()) return null;
      const sel = [];
      for (const e of data.edges) {
        if (!e.base) continue;
        const d = (e.vph[frame] || 0) - (e.base[frame] || 0);
        if (Math.abs(d) >= IMPACT_THRESH) sel.push({ e: e, d: d });
      }
      if (!sel.length) return null;
      const denom = vmax * 0.6 || 1;
      const pulse = 0.5 + 0.4 * (0.5 + 0.5 * Math.sin((time / LOOP) * Math.PI * 2));
      return new PathLayer({
        id: "impact",
        data: sel,
        getPath: (o) => o.e.coords,
        getColor: (o) => {
          const mag = Math.min(1, Math.abs(o.d) / denom);
          const a = Math.round((0.35 + 0.55 * mag) * pulse * 255);
          return o.d > 0 ? [249, 115, 22, a] : [56, 189, 248, a];
        },
        getWidth: (o) => 9 + 17 * Math.min(1, Math.abs(o.d) / denom),
        widthUnits: "pixels",
        widthMinPixels: 7,
        capRounded: true,
        jointRounded: true,
        parameters: { depthTest: false },
        updateTriggers: {
          getColor: [frame, Math.round(time)],
          getWidth: frame,
        },
      });
    }

    function draw() {
      if (!data) {
        overlay.setProps({ layers: [] });
        return;
      }
      const layers = [skeletonLayer()];
      const imp = impactLayer();
      if (imp) layers.push(imp);
      const hl = highlightLayer();
      if (hl) layers.push(hl);
      layers.push(pathLayer(), tripsLayer());
      const lbl = labelLayer();
      if (lbl) layers.push(lbl);
      overlay.setProps({ layers });
    }

    function refreshHud() {
      if (!data) return;
      let active = 0;
      let peak = 0;
      for (const e of data.edges) {
        const v = e.vph[frame] || 0;
        if (v > 0) active++;
        if (v > peak) peak = v;
      }
      ui.time.textContent = data.frames[frame] || "--:--";
      ui.stat.textContent = active + " active edges · peak " + peak + " vph";
    }

    function setFrame(f) {
      const n = data.frames.length;
      frame = ((f % n) + n) % n;
      ui.scrub.value = frame;
      tripsCache = buildTrips(data, frame, vmax);
      refreshHud();
    }

    // single rAF loop for the lifetime of the controller
    let rafId = null;
    function tick(ts) {
      time = (time + 0.6) % LOOP;
      if (data && playing) {
        const stepMs = FRAME_MS / SPEEDS[speedIdx];
        if (ts - lastAdvance > stepMs) {
          lastAdvance = ts;
          setFrame(frame + 1);
        }
      }
      draw();
      renderTip();
      rafId = requestAnimationFrame(tick);
    }

    // Fit the camera to all of Leonia: bounding box over every line
    // (active edges + grey skeleton) so the whole town is in frame.
    function dataBounds() {
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      const scan = (coords) => {
        for (const [x, y] of coords) {
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      };
      for (const e of data.edges) scan(e.coords);
      for (const line of data.skeleton) scan(line);
      if (!isFinite(minX)) return null;
      return [
        [minX, minY],
        [maxX, maxY],
      ];
    }

    function fitToData() {
      if (!data) return;
      const b = dataBounds();
      if (!b) return;
      map.fitBounds(b, { padding: 36, duration: 0 });
    }

    map.on("load", () => {
      map.addControl(overlay);
      if (data && !centred) {
        fitToData();
        centred = true;
      }
      rafId = requestAnimationFrame(tick);
    });

    // --- controls --------------------------------------------------------
    ui.play.addEventListener("click", () => {
      playing = !playing;
      ui.play.innerHTML = playing ? "&#10073;&#10073; Pause" : "&#9658; Play";
    });
    ui.scrub.addEventListener("input", () => {
      playing = false;
      ui.play.innerHTML = "&#9658; Play";
      setFrame(+ui.scrub.value);
    });
    ui.speed.addEventListener("click", () => {
      speedIdx = (speedIdx + 1) % SPEEDS.length;
      ui.speed.innerHTML = SPEEDS[speedIdx] + "&times;";
    });
    if (ui.theme) {
      ui.theme.addEventListener("change", () => {
        const t = THEMES.find((x) => x.id === ui.theme.value);
        if (t) map.setStyle(t.url);
      });
    }
    if (ui.impact) {
      ui.impact.addEventListener("click", () => {
        if (ui.impact.disabled) return;
        impactOn = !impactOn;
        ui.impact.classList.toggle("deck-toggle-on", impactOn);
        if (ui.impactKey) ui.impactKey.hidden = !impactOn;
      });
    }

    // --- public API ------------------------------------------------------
    function update(next) {
      data = next;
      vmax = (next.meta && next.meta.vmax_vph) || 600;
      hover = null;
      pinned = null;
      labelData = []; // street-name labels removed for now
      // The impact view needs baseline data embedded in flow.json.
      if (ui.impact) {
        const ok = !!(next.meta && next.meta.has_baseline);
        ui.impact.disabled = !ok;
        ui.impact.title = ok
          ? "Highlight streets whose traffic changed vs the baseline"
          : "No baseline data for this scenario";
        if (!ok && impactOn) {
          impactOn = false;
          ui.impact.classList.remove("deck-toggle-on");
          if (ui.impactKey) ui.impactKey.hidden = true;
        }
      }
      ui.scrub.max = Math.max(0, next.frames.length - 1);
      if (!centred && map.isStyleLoaded()) {
        fitToData();
        centred = true;
      }
      setFrame(0);
      playing = true;
      ui.play.innerHTML = "&#10073;&#10073; Pause";
    }

    function setHighlight(ids) {
      highlightIds = new Set(Array.isArray(ids) ? ids.map(String) : []);
    }

    function clear() {
      data = null;
      overlay.setProps({ layers: [] });
      ui.time.textContent = "--:--";
      ui.stat.textContent = "";
      ui.tip.style.display = "none";
    }

    function destroy() {
      if (rafId) cancelAnimationFrame(rafId);
      try {
        map.remove();
      } catch (_e) {
        /* ignore */
      }
    }

    return { update, setHighlight, clear, destroy, map };
  }

  window.LeoniaDeckFlow = { create };
})();
