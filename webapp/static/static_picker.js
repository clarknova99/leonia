/* Static Maps tab glue + tab controller.
 *
 * - Owns the tab bar: shows/hides panels and calls each panel's lazy
 *   show() hook (registered via window.LeoniaTabs) so each MapLibre map
 *   is created only once its tab is first visible.
 * - Drives the Static Maps controls (Map type / Day type / Day part /
 *   Year), fetches the matching _static/*.json artefact, and feeds it to
 *   the deck.gl static renderer (deckgl_static.js).
 */

window.LeoniaTabs = window.LeoniaTabs || {
  _hooks: {},
  register(name, hooks) {
    this._hooks[name] = hooks;
  },
};

(function () {
  "use strict";

  const els = {
    maptype: document.getElementById("static-maptype"),
    daytype: document.getElementById("static-daytype"),
    daypart: document.getElementById("static-daypart"),
    year: document.getElementById("static-year"),
    status: document.getElementById("static-status"),
    map: document.getElementById("static-map"),
    panel: document.querySelector('.tab-panel[data-panel="static"]'),
    topPanel: document.getElementById("static-top-panel"),
    topTitle: document.getElementById("static-top-title"),
    topMetric: document.getElementById("static-top-metric"),
    topBody: document.getElementById("static-top-body"),
  };

  // Last-loaded payloads, kept so the top-roads table can be recomputed
  // when only the day-part / year filter changes (the map renderer is
  // updated in place without a refetch in those cases).
  let lastTraffic = null;
  let lastCrash = null;
  // Hour currently shown by the renderer during "Hourly (24 hrs)" playback;
  // the renderer pushes updates here via its onHourChange callback so the
  // top-roads table can re-rank for the active hour.
  let currentHour = 0;

  const ESCAPES = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ESCAPES[c]);
  }

  // Street-suffix abbreviations → full words, so crash-location spellings
  // like "BROAD AVE" and "Broad Avenue" collapse onto one road row.
  const SUFFIX = {
    AVE: "Avenue", AV: "Avenue", ST: "Street", RD: "Road", DR: "Drive",
    PL: "Place", LN: "Lane", BLVD: "Boulevard", CT: "Court",
    TER: "Terrace", TERR: "Terrace", HWY: "Highway", PKWY: "Parkway",
    CIR: "Circle", SQ: "Square", PLZ: "Plaza",
  };

  function titleCase(s) {
    return s.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
  }

  // Roads that carry different OSM names along one physical corridor and
  // should collapse to a single row. Keyed by UPPERCASE display name →
  // canonical display name. Fort Lee Road becomes "Main Street" once it
  // crosses into Fort Lee, but it's the same road through Leonia.
  const ROAD_ALIASES = {
    "MAIN STREET": "Fort Lee Road",
  };

  function aliasRoad(name) {
    return ROAD_ALIASES[String(name).toUpperCase()] || name;
  }

  // Normalise a crash label / road string to a display road name. Crash
  // labels are "<on-road> × <cross-street>" or "<on-road> / <cross>"; we
  // keep the on-road only so a road groups to one row.
  function roadDisplay(raw) {
    if (!raw) return "Unknown";
    let s = String(raw).split(/[\u00d7/]/)[0].trim().replace(/\s+/g, " ");
    if (!s || s === "(unknown)") return "Unknown";
    const parts = s.split(" ");
    const lastKey = parts[parts.length - 1].toUpperCase().replace(/\.$/, "");
    if (SUFFIX[lastKey]) parts[parts.length - 1] = SUFFIX[lastKey];
    return titleCase(parts.join(" "));
  }

  function setStatus(text, variant) {
    if (!els.status) return;
    els.status.textContent = text || "";
    els.status.classList.remove("warn", "error");
    if (variant === "warn") els.status.classList.add("warn");
    if (variant === "error") els.status.classList.add("error");
  }

  // --- tab controller --------------------------------------------------
  function setupTabs() {
    const tabs = Array.from(document.querySelectorAll(".tab"));
    const panels = Array.from(document.querySelectorAll(".tab-panel"));
    if (!tabs.length) return;
    function activate(name) {
      panels.forEach((p) => {
        const on = p.dataset.panel === name;
        p.hidden = !on;
        p.classList.toggle("is-active", on);
      });
      tabs.forEach((b) => {
        const on = b.dataset.tab === name;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
      });
      const hooks = window.LeoniaTabs._hooks[name];
      if (hooks && hooks.show) hooks.show();
    }
    tabs.forEach((b) =>
      b.addEventListener("click", () => activate(b.dataset.tab)),
    );
    const initial =
      tabs.find((b) => b.classList.contains("is-active")) || tabs[0];
    activate(initial.dataset.tab);
  }

  // --- static map state ------------------------------------------------
  let ctl = null;
  let staticCatalog = null;
  let started = false;
  let seq = 0;
  const cache = new Map();

  function ensureDeck() {
    if (ctl) return ctl;
    if (
      !window.LeoniaStaticMap ||
      typeof window.maplibregl === "undefined" ||
      typeof window.deck === "undefined" ||
      !els.map
    ) {
      return null;
    }
    ctl = window.LeoniaStaticMap.create(els.map);
    // Re-rank the top-roads table for each hour as the player advances.
    if (ctl.onHourChange) {
      ctl.onHourChange((h) => {
        currentHour = h;
        if (
          els.maptype && els.maptype.value === "traffic" &&
          els.daypart && els.daypart.value === "hourly"
        ) {
          renderTopRoads();
        }
      });
    }
    return ctl;
  }

  async function loadJson(path) {
    if (cache.has(path)) return cache.get(path);
    const res = await fetch(`/precache/${path}`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const j = await res.json();
    cache.set(path, j);
    return j;
  }

  function fmtHour(h) {
    const ampm = h < 12 || h === 24 ? "a" : "p";
    let hr = h % 12;
    if (hr === 0) hr = 12;
    return hr + ampm;
  }

  // Long clock label ("12 AM" … "11 PM") for the hourly-playback table title.
  function fmtClock(h) {
    const hr = ((h % 24) + 24) % 24;
    const ampm = hr < 12 ? "AM" : "PM";
    const display = hr % 12 === 0 ? 12 : hr % 12;
    return `${display} ${ampm}`;
  }

  function fmtWindow(win) {
    if (!Array.isArray(win) || win.length !== 2) return "";
    const [lo, hi] = win;
    // Collapse the am/pm suffix when both ends share it (e.g. "7–10a").
    const loPM = lo >= 12;
    const hiPM = hi >= 12;
    if (loPM === hiPM) {
      let a = lo % 12 || 12;
      let b = hi % 12 || 12;
      return `${a}\u2013${b}${hiPM ? "p" : "a"}`;
    }
    return `${fmtHour(lo)}\u2013${fmtHour(hi)}`;
  }

  // Relabel Peak AM / Peak PM options with the day-type-specific window.
  function updateDaypartLabels(dayType) {
    if (!els.daypart) return;
    const pw =
      (staticCatalog &&
        staticCatalog.peak_windows &&
        staticCatalog.peak_windows[dayType]) ||
      null;
    // Base labels for the off-peak windows differ by day type: weekday's
    // early off-peak is the midday lull, while Sunday's is the quiet morning
    // before the midday plateau. The hour range is appended from the catalog.
    const earlyBase =
      dayType === "sunday" ? "Off-peak morning" : "Off-peak midday";
    const lateBase = "Off-peak evening";
    const withWin = (base, win) =>
      pw && win ? `${base} (${fmtWindow(win)})` : base;
    for (const opt of els.daypart.options) {
      if (opt.value === "all_day") opt.textContent = "All day";
      else if (opt.value === "peak_am") {
        opt.textContent = withWin("Peak AM", pw && pw.peak_am);
      } else if (opt.value === "peak_pm") {
        opt.textContent = withWin("Peak PM", pw && pw.peak_pm);
      } else if (opt.value === "off_peak_early") {
        opt.textContent = withWin(earlyBase, pw && pw.off_peak_early);
      } else if (opt.value === "off_peak_late") {
        opt.textContent = withWin(lateBase, pw && pw.off_peak_late);
      } else if (opt.value === "overnight") {
        opt.textContent = withWin("Off-peak overnight", pw && pw.overnight);
      }
    }
  }

  function populateYears() {
    if (!els.year) return;
    const years = (staticCatalog && staticCatalog.crash_years) || [];
    // Keep the "All years" option, then one per available year (newest first).
    els.year.innerHTML = '<option value="all">All years</option>';
    years
      .slice()
      .sort((a, b) => b - a)
      .forEach((y) => {
        const opt = document.createElement("option");
        opt.value = String(y);
        opt.textContent = String(y);
        els.year.appendChild(opt);
      });
  }

  // Show only the controls/legends relevant to the chosen map type.
  function toggleControls(mapType) {
    if (!els.panel) return;
    els.panel.querySelectorAll("[data-when]").forEach((node) => {
      node.hidden = node.dataset.when !== mapType;
    });
  }

  function dayTypeLabel(dt) {
    return dt === "sunday" ? "Sunday" : "Weekday";
  }

  function partLabel(part) {
    const labels = {
      peak_am: "Peak AM",
      peak_pm: "Peak PM",
      off_peak_early: "off-peak early",
      off_peak_late: "off-peak late",
      overnight: "off-peak overnight",
      hourly: "hourly playback",
    };
    return labels[part] || "all day";
  }

  // --- top-10 roads table ----------------------------------------------
  function hideTopRoads() {
    if (els.topBody) els.topBody.innerHTML = "";
    if (els.topPanel) els.topPanel.hidden = true;
  }

  // Busiest roads for the selected window. Edges are per-segment, so we
  // roll segments up by street name and take the busiest segment's vph as
  // the road's value (its peak point). Restricted to Leonia-local roads:
  // the builder flags each edge `in_leonia` (strictly in-borough + a
  // local-road name), so the GWB approach corridor (turnpike, US-1-9-46,
  // Bergen Boulevard, ramps…) the map shows for context is excluded here.
  function topTrafficRowsBy(getVal) {
    if (!lastTraffic || !Array.isArray(lastTraffic.edges)) return [];
    const edges = lastTraffic.edges;
    const hasFlag = edges.some((e) => "in_leonia" in e);
    const byName = new Map();
    for (const e of edges) {
      if (hasFlag && !e.in_leonia) continue;
      const name = aliasRoad(e.name || e.id);
      const v = getVal(e) || 0;
      if (v > (byName.get(name) || 0)) byName.set(name, v);
    }
    return Array.from(byName, ([name, value]) => ({ name, value }))
      .filter((r) => r.value > 0)
      .sort((a, b) => b.value - a.value)
      .slice(0, 10);
  }

  function topTrafficRows(part) {
    return topTrafficRowsBy((e) => (e.vals && e.vals[part]) || 0);
  }

  // Same ranking but for one hour of the measured 24h profile (the
  // "Hourly (24 hrs)" day-part), so the table tracks the player.
  function topTrafficRowsHourly(h) {
    return topTrafficRowsBy((e) => (e.hourly && e.hourly[h]) || 0);
  }

  // Roads with the most crashes for the selected year filter. Crash points
  // carry an optional `road`; otherwise the on-road is parsed from `label`.
  function topCrashRows(yr) {
    if (!lastCrash || !Array.isArray(lastCrash.points)) return [];
    const y = yr === "all" || yr == null ? null : +yr;
    const byRoad = new Map();
    for (const p of lastCrash.points) {
      if (y !== null && p.year !== y) continue;
      const disp = aliasRoad(roadDisplay(p.road || p.label));
      const key = disp.toUpperCase();
      const cur = byRoad.get(key) || { name: disp, value: 0 };
      cur.value += 1;
      byRoad.set(key, cur);
    }
    return Array.from(byRoad.values())
      .sort((a, b) => b.value - a.value)
      .slice(0, 10);
  }

  function renderTopRoads() {
    if (!els.topPanel || !els.topBody) return;
    const mapType = els.maptype ? els.maptype.value : "traffic";
    let rows;
    let title;
    let metric;
    let fmt;
    if (mapType === "traffic") {
      const dt = els.daytype ? els.daytype.value : "weekday";
      const part = els.daypart ? els.daypart.value : "all_day";
      if (part === "hourly") {
        rows = topTrafficRowsHourly(currentHour);
        title =
          `Top 10 busiest roads \u2014 ${dayTypeLabel(dt)}, ${fmtClock(currentHour)}`;
        metric = "vph";
      } else {
        rows = topTrafficRows(part);
        title =
          `Top 10 busiest roads \u2014 ${dayTypeLabel(dt)}, ${partLabel(part)}`;
        metric = "Avg vph";
      }
      fmt = (v) => Math.round(v).toLocaleString();
    } else {
      const yr = els.year ? els.year.value : "all";
      rows = topCrashRows(yr);
      title =
        "Top 10 roads by crashes \u2014 " +
        (yr === "all" ? "all years" : yr);
      metric = "Crashes";
      fmt = String;
    }
    if (!rows.length) {
      hideTopRoads();
      return;
    }
    if (els.topTitle) els.topTitle.textContent = title;
    if (els.topMetric) els.topMetric.textContent = metric;
    els.topBody.innerHTML = rows
      .map(
        (r, i) =>
          `<tr><td class="rank-col">${i + 1}</td>` +
          `<td class="road-col">${escapeHtml(r.name)}</td>` +
          `<td class="num-col">${fmt(r.value)}</td></tr>`,
      )
      .join("");
    els.topPanel.hidden = false;
  }

  async function apply() {
    const renderer = ensureDeck();
    if (!renderer) {
      setStatus("Map libraries failed to load (check your connection).", "error");
      return;
    }
    const mapType = els.maptype ? els.maptype.value : "traffic";
    toggleControls(mapType);
    hideTopRoads();
    const mySeq = ++seq;

    if (mapType === "traffic") {
      const dt = els.daytype ? els.daytype.value : "weekday";
      const part = els.daypart ? els.daypart.value : "all_day";
      const path =
        staticCatalog && staticCatalog.traffic && staticCatalog.traffic[dt];
      if (!path) {
        setStatus(`No traffic data for ${dayTypeLabel(dt)}.`, "warn");
        renderer.clear();
        return;
      }
      setStatus(`Loading ${dayTypeLabel(dt)} traffic…`);
      let data;
      try {
        data = await loadJson(path);
      } catch (err) {
        if (mySeq !== seq) return;
        setStatus(`Could not load traffic: ${err.message}`, "error");
        renderer.clear();
        return;
      }
      if (mySeq !== seq) return;
      renderer.showTraffic(data, part);
      lastTraffic = data;
      if (part === "hourly" && renderer.currentHour) {
        currentHour = renderer.currentHour();
      }
      renderTopRoads();
      setStatus(
        part === "hourly"
          ? `Measured traffic — ${dayTypeLabel(dt)}, hourly playback (vph by hour).`
          : `Measured traffic — ${dayTypeLabel(dt)}, ${partLabel(part)} (avg vph).`,
      );
    } else {
      const path = staticCatalog && staticCatalog.crash;
      if (!path) {
        setStatus("No crash data available.", "warn");
        renderer.clear();
        return;
      }
      setStatus("Loading crashes…");
      let data;
      try {
        data = await loadJson(path);
      } catch (err) {
        if (mySeq !== seq) return;
        setStatus(`Could not load crashes: ${err.message}`, "error");
        renderer.clear();
        return;
      }
      if (mySeq !== seq) return;
      const yr = els.year ? els.year.value : "all";
      renderer.showCrash(data, yr);
      lastCrash = data;
      renderTopRoads();
      setStatus(
        `NJDOT crashes — ${yr === "all" ? "all years" : yr}.`,
      );
    }
  }

  function wireControls() {
    if (els.maptype) {
      els.maptype.addEventListener("change", apply);
    }
    if (els.daytype) {
      els.daytype.addEventListener("change", () => {
        updateDaypartLabels(els.daytype.value);
        apply(); // different artefact for a different day type
      });
    }
    if (els.daypart) {
      // Same artefact — just swap which window is drawn (keeps camera).
      els.daypart.addEventListener("change", () => {
        if (!ctl) return apply();
        ctl.setValueKey(els.daypart.value);
        if (els.daypart.value === "hourly" && ctl.currentHour) {
          currentHour = ctl.currentHour();
        }
        renderTopRoads();
        setStatus(
          els.daypart.value === "hourly"
            ? `Measured traffic — ${dayTypeLabel(els.daytype.value)}, ` +
                "hourly playback (vph by hour)."
            : `Measured traffic — ${dayTypeLabel(els.daytype.value)}, ` +
                `${partLabel(els.daypart.value)} (avg vph).`,
        );
      });
    }
    if (els.year) {
      els.year.addEventListener("change", () => {
        if (!ctl) return apply();
        ctl.setYear(els.year.value);
        renderTopRoads();
        setStatus(
          `NJDOT crashes — ${els.year.value === "all" ? "all years" : els.year.value}.`,
        );
      });
    }
  }

  async function initStatic() {
    setStatus("Loading catalog…");
    let catalog;
    try {
      const res = await fetch("/api/catalog.json", { cache: "no-store" });
      if (!res.ok) throw new Error(`catalog fetch failed: ${res.status}`);
      catalog = await res.json();
    } catch (err) {
      setStatus(`Could not load catalog: ${err.message}`, "error");
      return;
    }
    staticCatalog = catalog.static || {};
    populateYears();
    updateDaypartLabels(els.daytype ? els.daytype.value : "weekday");
    wireControls();
    apply();
  }

  // Register the lazy show hook for the Static Maps tab, then set up the
  // tab bar (this script loads last, so all panels are registered).
  if (els.map) {
    window.LeoniaTabs.register("static", {
      show() {
        if (!started) {
          started = true;
          initStatic();
          return;
        }
        const renderer = ensureDeck();
        if (renderer) setTimeout(() => renderer.resize(), 0);
      },
    });
  }

  setupTabs();
})();
