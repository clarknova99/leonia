/* Scenario picker glue.
 *
 * Fetches /api/catalog.json once, populates the street dropdown, and on
 * any dropdown change builds a key (street_slug__change__demand_label),
 * fetches that scenario's flow.json from the precache, and feeds it to
 * the deck.gl renderer (deckgl_flow.js). The selected street is
 * outlined in white.
 *
 * If a combination isn't in the precache (or predates the flow.json
 * artefact), we surface a clear status instead of swapping the map.
 */

(function () {
  "use strict";

  // Shared, minimal tab registry. The full controller (wiring + default
  // activation) lives in static_picker.js, which loads last; each picker
  // just registers lazy init/show hooks here so its MapLibre map is only
  // created once its tab is first shown (avoids initialising a map in a
  // display:none container).
  window.LeoniaTabs = window.LeoniaTabs || {
    _hooks: {},
    register(name, hooks) {
      this._hooks[name] = hooks;
    },
  };

  const DEMAND_VALUE_TO_LABEL = {
    "bridge_od_weekday_24h": "weekday",
    "bridge_od_sunday_24h": "sunday",
  };

  const els = {
    street: document.getElementById("street-select"),
    change: document.getElementById("change-select"),
    demand: document.getElementById("demand-select"),
    deckMap: document.getElementById("deck-map"),
    status: document.getElementById("scenario-status"),
    catalogMeta: document.getElementById("catalog-meta"),
    nScenarios: document.getElementById("n-scenarios"),
  };

  if (!els.street || !els.deckMap) {
    return;
  }

  // street_slug -> SUMO edge ids (for the white highlight outline).
  const streetEdgeIndex = new Map();

  // deck.gl controller, created lazily once the libraries are present.
  let deck = null;

  // Monotonic token so an in-flight fetch for a stale selection can't
  // clobber a newer one when the user clicks through quickly.
  let requestSeq = 0;

  // Small client-side cache: flow.json payloads can be ~1 MB, so we
  // avoid re-fetching when the user toggles back to a recent scenario.
  const flowCache = new Map();

  function setStatus(text, variant) {
    if (!els.status) return;
    els.status.textContent = text || "";
    els.status.classList.remove("warn", "error");
    if (variant === "warn") els.status.classList.add("warn");
    if (variant === "error") els.status.classList.add("error");
  }

  function ensureDeck() {
    if (deck) return deck;
    if (
      !window.LeoniaDeckFlow ||
      typeof window.maplibregl === "undefined" ||
      typeof window.deck === "undefined"
    ) {
      return null;
    }
    deck = window.LeoniaDeckFlow.create(els.deckMap);
    return deck;
  }

  function buildKey() {
    const slug = els.street.value;
    const change = els.change.value;
    const demand = els.demand.value;
    const demandLabel = DEMAND_VALUE_TO_LABEL[demand] || demand;
    return `${slug}__${change}__${demandLabel}`;
  }

  function optionText(sel) {
    return sel.options[sel.selectedIndex]
      ? sel.options[sel.selectedIndex].text
      : sel.value;
  }

  // Fetch a flow.json payload (with a small client-side cache) and bail
  // out if a newer selection has superseded this request. Returns null
  // when superseded; throws on network/parse errors.
  async function loadFlow(flowPath, cacheKey, seq) {
    let data = flowCache.get(cacheKey);
    if (!data) {
      // Revalidate against the server (cheap 304 when unchanged) rather
      // than force-cache: precache rebuilds change flow.json in place.
      const res = await fetch(`/precache/${flowPath}`, { cache: "no-cache" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      data = await res.json();
      flowCache.set(cacheKey, data);
    }
    if (seq !== requestSeq) return null; // a newer selection won
    return data;
  }

  async function applyScenario(catalog) {
    const ctl = ensureDeck();
    if (!ctl) {
      setStatus("Map libraries failed to load (check your connection).", "error");
      return;
    }

    // With no street selected we show the unchanged baseline for the
    // chosen demand (defaults to the weekday simulation on page load).
    if (!els.street.value) {
      const seq = ++requestSeq;
      ctl.setHighlight([]);
      const demandVal = els.demand.value;
      const base = (catalog.baselines || {})[demandVal];
      if (!base || !base.flow_json) {
        setStatus("Select a street to see its 24-hour traffic flow.");
        ctl.clear();
        return;
      }
      const label = base.demand_label || demandVal;
      setStatus(`Baseline — ${label} (no street changes) — loading flow…`);
      try {
        const data = await loadFlow(base.flow_json, `__baseline__${demandVal}`, seq);
        if (data === null) return; // superseded
        ctl.update(data);
        setStatus(
          `Baseline — ${label} traffic, no street changes. ` +
            `Select a street to model a scenario.`,
        );
      } catch (err) {
        if (seq !== requestSeq) return;
        setStatus(`Could not load baseline flow: ${err.message}`, "error");
        ctl.clear();
      }
      return;
    }

    const key = buildKey();
    const entry = catalog.scenarios[key];

    if (!entry) {
      setStatus(
        `No precomputed scenario for ${optionText(els.street)} · ` +
          `${optionText(els.change)} · ${optionText(els.demand)}.`,
        "warn",
      );
      ctl.clear();
      return;
    }

    if (!entry.ok || !entry.flow_json) {
      const warns = (entry.warnings || []).join("; ");
      setStatus(
        entry.flow_json
          ? `Scenario built with warnings: ${warns || "no map available"}.`
          : "This scenario predates the deck.gl flow data — rebuild the " +
              "precache (build_precache.py --force) to generate flow.json.",
        "warn",
      );
      ctl.clear();
      return;
    }

    const hasWarnings = entry.warnings && entry.warnings.length;
    const baseStatus =
      `Loaded: ${entry.street_name} · ${entry.change_type} · ` +
      `${entry.demand_label}`;

    const seq = ++requestSeq;
    setStatus(`${baseStatus} — loading flow…`);

    let data;
    try {
      data = await loadFlow(entry.flow_json, key, seq);
    } catch (err) {
      if (seq !== requestSeq) return; // superseded
      setStatus(`Could not load scenario flow: ${err.message}`, "error");
      ctl.clear();
      return;
    }
    if (data === null) return; // a newer selection won

    ctl.setHighlight(streetEdgeIndex.get(els.street.value) || []);
    ctl.update(data);
    setStatus(
      hasWarnings
        ? `${baseStatus} (warnings: ${entry.warnings.join("; ")})`
        : baseStatus,
      hasWarnings ? "warn" : null,
    );
  }

  function populateStreets(catalog) {
    const streets = (catalog.streets || []).slice().sort((a, b) => {
      return (a.name || "").localeCompare(b.name || "", undefined, {
        sensitivity: "base",
      });
    });
    els.street.innerHTML = "";
    streetEdgeIndex.clear();
    // Placeholder so the page opens with no street selected.
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select a street…";
    placeholder.selected = true;
    els.street.appendChild(placeholder);
    streets.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.slug;
      opt.textContent = s.name;
      els.street.appendChild(opt);
      streetEdgeIndex.set(
        s.slug,
        Array.isArray(s.sumo_edge_ids) ? s.sumo_edge_ids : [],
      );
    });
  }

  function populateChangeTypes(catalog) {
    if (!Array.isArray(catalog.change_types) || catalog.change_types.length === 0) {
      return;
    }
    const have = new Set(Array.from(els.change.options).map((o) => o.value));
    const want = new Set(catalog.change_types.map((c) => c.value));
    const sameMembers =
      have.size === want.size && [...want].every((v) => have.has(v));
    if (sameMembers) return;

    els.change.innerHTML = "";
    catalog.change_types.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c.value;
      opt.textContent = c.label || c.value;
      els.change.appendChild(opt);
    });
  }

  function populateDemands(catalog) {
    if (!Array.isArray(catalog.demands) || catalog.demands.length === 0) {
      return;
    }
    const have = new Set(Array.from(els.demand.options).map((o) => o.value));
    const want = new Set(catalog.demands.map((d) => d.value));
    const sameMembers =
      have.size === want.size && [...want].every((v) => have.has(v));
    if (sameMembers) return;

    els.demand.innerHTML = "";
    catalog.demands.forEach((d) => {
      const opt = document.createElement("option");
      opt.value = d.value;
      opt.textContent = d.label || d.value;
      els.demand.appendChild(opt);
    });
  }

  async function init() {
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

    populateStreets(catalog);
    populateChangeTypes(catalog);
    populateDemands(catalog);

    if (els.catalogMeta) {
      const built = catalog.built_at || "(unknown)";
      const n = Object.keys(catalog.scenarios || {}).length;
      els.catalogMeta.textContent = `Catalog built ${built} · ${n} scenarios`;
    }
    if (els.nScenarios) {
      els.nScenarios.textContent = Object.keys(catalog.scenarios || {}).length;
    }

    [els.street, els.change, els.demand].forEach((sel) => {
      sel.addEventListener("change", () => applyScenario(catalog));
    });

    applyScenario(catalog);
  }

  // Defer map creation until the Simulation tab is first shown.
  let started = false;
  window.LeoniaTabs.register("simulation", {
    show() {
      if (!started) {
        started = true;
        init();
        return;
      }
      const ctl = ensureDeck();
      if (ctl && ctl.map) {
        // Container was hidden; MapLibre needs a resize once visible.
        setTimeout(() => ctl.map.resize(), 0);
      }
    },
  });
})();
