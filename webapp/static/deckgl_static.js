/* Static (non-animated) deck.gl renderer for the Static Maps tab.
 *
 * Renders two kinds of map on a single MapLibre + deck.gl surface:
 *
 *   Traffic — a PathLayer of streets coloured/sized by average
 *     vehicles/hour for the selected window (All Day / Peak AM /
 *     Peak PM), from _static/traffic_<daytype>.json:
 *       { meta:{vmax_vph, center, ...}, skeleton:[...],
 *         edges:[{id, name, coords, vals:{all_day,peak_am,peak_pm}}] }
 *
 *   Crash — a ScatterplotLayer of NJDOT crash points coloured by KABCO
 *     severity and filtered by year, from _static/crashes.json:
 *       { meta:{...}, years:[...], skeleton:[...],
 *         points:[{lat,lon,year,severity,severity_label,epdo,ped,label,date}] }
 *
 * Usage:
 *   const ctl = LeoniaStaticMap.create(containerEl);
 *   ctl.showTraffic(data, "all_day");
 *   ctl.showCrash(data, "all");   // or a specific year (number/string)
 *   ctl.setYear(2024);            // re-filter crash points without refetch
 *   ctl.resize();                 // after the tab becomes visible
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

  // Perceptual (sqrt) normalisation. A linear v/vmax ramp anchored to
  // the high-volume arterials (~1000 vph) compresses the ~half of
  // streets under 100 vph into a single indistinguishable green, so
  // day-part / day-type changes look identical on residential blocks.
  // sqrt lifts the low end into distinguishable colours while keeping
  // the scale comparable across windows (same vmax).
  function norm(v, vmax) {
    return Math.sqrt(Math.max(0, Math.min(1, v / (vmax || 1))));
  }

  function rampColor(v, vmax) {
    const t = norm(v, vmax);
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

  // KABCO severity → [r,g,b] + pixel radius. Mirrors the colours in
  // leonia_traffic.sumo.visualizations._crash_points_to_payload.
  const SEVERITY = {
    K: { color: [183, 28, 28], radius: 7 },
    F: { color: [183, 28, 28], radius: 7 },
    A: { color: [211, 47, 47], radius: 6 },
    B: { color: [239, 108, 0], radius: 5 },
    C: { color: [245, 124, 0], radius: 4.5 },
    I: { color: [239, 108, 0], radius: 5 },
    O: { color: [97, 97, 97], radius: 3.5 },
    P: { color: [97, 97, 97], radius: 3.5 },
  };

  function severityStyle(sev) {
    return SEVERITY[(sev || "O").toUpperCase()] || SEVERITY.O;
  }

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

  function buildPanel(container) {
    const panel = document.createElement("div");
    panel.className = "deck-panel deck-panel-static";
    const themeOpts = THEMES.map(
      (t) => `<option value="${t.id}">${t.label}</option>`,
    ).join("");
    panel.innerHTML = [
      '<div class="deck-row deck-row-theme">',
      '<span class="deck-theme-label">Basemap</span>',
      `<select class="deck-theme">${themeOpts}</select>`,
      "</div>",
      '<div class="deck-static-readout"><small class="deck-stat"></small></div>',
    ].join("");
    container.appendChild(panel);
    const tip = document.createElement("div");
    tip.className = "deck-tip";
    container.appendChild(tip);
    return {
      panel,
      tip,
      stat: panel.querySelector(".deck-stat"),
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

    const { MapboxOverlay, PathLayer, ScatterplotLayer } = deck;
    const overlay = new MapboxOverlay({
      interleaved: false,
      pickingRadius: 6,
      layers: [],
    });

    const ui = buildPanel(container);

    let mode = "traffic"; // 'traffic' | 'crash'
    let data = null;
    let vmax = 600;
    let valueKey = "all_day";
    let crashYear = "all";
    let hover = null; // { html, x, y }
    let pinned = null;
    let centred = false;

    // --- helpers ---------------------------------------------------------
    function trafficValue(e) {
      return (e.vals && e.vals[valueKey]) || 0;
    }

    function visibleCrashPoints() {
      if (!data || !data.points) return [];
      if (crashYear === "all" || crashYear == null) return data.points;
      const y = +crashYear;
      return data.points.filter((p) => p.year === y);
    }

    function renderTip() {
      const sel = pinned || hover;
      if (!sel) {
        ui.tip.style.display = "none";
        return;
      }
      ui.tip.style.display = "block";
      ui.tip.style.left = sel.x + "px";
      ui.tip.style.top = sel.y + "px";
      ui.tip.innerHTML =
        sel.html +
        '<div class="deck-tip-pin">' +
        (pinned ? "pinned — click empty space to clear" : "click to pin") +
        "</div>";
    }

    function trafficTipHtml(e) {
      const w = { all_day: "All day", peak_am: "Peak AM", peak_pm: "Peak PM" };
      return (
        '<div class="deck-tip-name">' +
        (e.name || e.id) +
        "</div>" +
        '<div class="deck-tip-vph"><b>' +
        trafficValue(e) +
        "</b> avg vph &middot; " +
        (w[valueKey] || valueKey) +
        "</div>"
      );
    }

    function crashTipHtml(p) {
      const bits = [];
      bits.push('<div class="deck-tip-name">' + (p.label || "(crash)") + "</div>");
      bits.push(
        '<div class="deck-tip-vph">' +
          (p.severity_label || p.severity || "") +
          (p.year ? " &middot; " + p.year : "") +
          (p.date ? " &middot; " + p.date : "") +
          "</div>",
      );
      if (p.ped) bits.push('<div class="deck-tip-delta">pedestrian involved</div>');
      return bits.join("");
    }

    // --- layers ----------------------------------------------------------
    function skeletonLayer() {
      return new PathLayer({
        id: "skeleton",
        data: (data && data.skeleton) || [],
        getPath: (d) => d,
        getColor: [70, 80, 100, 120],
        getWidth: 0.7,
        widthUnits: "pixels",
        widthMinPixels: 0.6,
        parameters: { depthTest: false },
      });
    }

    function trafficLayer() {
      return new PathLayer({
        id: "traffic",
        data: data.edges,
        getPath: (e) => e.coords,
        getColor: (e) => rampColor(trafficValue(e), vmax),
        getWidth: (e) => 1.5 + 5 * norm(trafficValue(e), vmax),
        widthUnits: "pixels",
        widthMinPixels: 1.2,
        capRounded: true,
        jointRounded: true,
        opacity: 0.85,
        parameters: { depthTest: false },
        pickable: true,
        onHover: (info) => {
          hover = info.object
            ? { html: trafficTipHtml(info.object), x: info.x, y: info.y }
            : null;
          renderTip();
        },
        onClick: (info) => {
          pinned = info.object
            ? { html: trafficTipHtml(info.object), x: info.x, y: info.y }
            : null;
          renderTip();
        },
        updateTriggers: { getColor: valueKey, getWidth: valueKey },
      });
    }

    function crashLayer() {
      const pts = visibleCrashPoints();
      return new ScatterplotLayer({
        id: "crashes",
        data: pts,
        getPosition: (p) => [p.lon, p.lat],
        getFillColor: (p) => {
          const c = severityStyle(p.severity).color;
          return [c[0], c[1], c[2], 220];
        },
        getLineColor: [10, 15, 26, 200],
        lineWidthMinPixels: 0.5,
        stroked: true,
        getRadius: (p) => severityStyle(p.severity).radius,
        radiusUnits: "pixels",
        radiusMinPixels: 2,
        radiusMaxPixels: 14,
        parameters: { depthTest: false },
        pickable: true,
        onHover: (info) => {
          hover = info.object
            ? { html: crashTipHtml(info.object), x: info.x, y: info.y }
            : null;
          renderTip();
        },
        onClick: (info) => {
          pinned = info.object
            ? { html: crashTipHtml(info.object), x: info.x, y: info.y }
            : null;
          renderTip();
        },
        updateTriggers: { data: crashYear },
      });
    }

    function draw() {
      if (!data) {
        overlay.setProps({ layers: [] });
        return;
      }
      const layers = [skeletonLayer()];
      if (mode === "traffic") layers.push(trafficLayer());
      else layers.push(crashLayer());
      overlay.setProps({ layers });
    }

    function refreshHud() {
      if (!ui.stat) return;
      if (!data) {
        ui.stat.textContent = "";
        return;
      }
      if (mode === "traffic") {
        ui.stat.textContent = data.edges.length + " streets with measured data";
      } else {
        const n = visibleCrashPoints().length;
        ui.stat.textContent =
          n + " crashes" + (crashYear === "all" ? " (all years)" : " in " + crashYear);
      }
    }

    // --- camera ----------------------------------------------------------
    function dataBounds() {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      const scanPt = (x, y) => {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      };
      if (data.skeleton) for (const line of data.skeleton) for (const [x, y] of line) scanPt(x, y);
      if (mode === "traffic" && data.edges) {
        for (const e of data.edges) for (const [x, y] of e.coords) scanPt(x, y);
      } else if (data.points) {
        for (const p of data.points) scanPt(p.lon, p.lat);
      }
      if (!isFinite(minX)) return null;
      return [[minX, minY], [maxX, maxY]];
    }

    function fitToData() {
      const b = dataBounds();
      if (b) map.fitBounds(b, { padding: 36, duration: 0 });
    }

    map.on("load", () => {
      map.addControl(overlay);
      if (data && !centred) {
        fitToData();
        centred = true;
      }
      draw();
    });

    if (ui.theme) {
      ui.theme.addEventListener("change", () => {
        const t = THEMES.find((x) => x.id === ui.theme.value);
        if (t) map.setStyle(t.url);
      });
    }

    // --- public API ------------------------------------------------------
    function _afterDataSet() {
      hover = null;
      pinned = null;
      ui.tip.style.display = "none";
      if (map.isStyleLoaded()) {
        fitToData();
        centred = true;
      } else {
        centred = false;
      }
      refreshHud();
      draw();
    }

    function showTraffic(next, key) {
      mode = "traffic";
      data = next;
      valueKey = key || "all_day";
      vmax = (next && next.meta && next.meta.vmax_vph) || 600;
      _afterDataSet();
    }

    function setValueKey(key) {
      if (mode !== "traffic") return;
      valueKey = key || "all_day";
      hover = null;
      pinned = null;
      ui.tip.style.display = "none";
      refreshHud();
      draw();
    }

    function showCrash(next, year) {
      mode = "crash";
      data = next;
      crashYear = year == null ? "all" : year;
      _afterDataSet();
    }

    function setYear(year) {
      if (mode !== "crash") return;
      crashYear = year == null ? "all" : year;
      hover = null;
      pinned = null;
      ui.tip.style.display = "none";
      refreshHud();
      draw();
    }

    function clear() {
      data = null;
      overlay.setProps({ layers: [] });
      if (ui.stat) ui.stat.textContent = "";
      ui.tip.style.display = "none";
    }

    function resize() {
      try {
        map.resize();
      } catch (_e) {
        /* ignore */
      }
    }

    function destroy() {
      try {
        map.remove();
      } catch (_e) {
        /* ignore */
      }
    }

    return {
      showTraffic,
      setValueKey,
      showCrash,
      setYear,
      clear,
      resize,
      destroy,
      map,
    };
  }

  window.LeoniaStaticMap = { create };
})();
