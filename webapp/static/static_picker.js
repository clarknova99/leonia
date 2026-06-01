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
  };

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
    };
    return labels[part] || "all day";
  }

  async function apply() {
    const renderer = ensureDeck();
    if (!renderer) {
      setStatus("Map libraries failed to load (check your connection).", "error");
      return;
    }
    const mapType = els.maptype ? els.maptype.value : "traffic";
    toggleControls(mapType);
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
      setStatus(
        `Measured traffic — ${dayTypeLabel(dt)}, ${partLabel(part)} (avg vph).`,
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
        setStatus(
          `Measured traffic — ${dayTypeLabel(els.daytype.value)}, ` +
            `${partLabel(els.daypart.value)} (avg vph).`,
        );
      });
    }
    if (els.year) {
      els.year.addEventListener("change", () => {
        if (!ctl) return apply();
        ctl.setYear(els.year.value);
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
