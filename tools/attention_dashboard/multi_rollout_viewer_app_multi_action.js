(function () {
  const ACTION_PLOT_HEIGHT = 150;
  const CELL_SIZE_MIN = 1;
  const CELL_SIZE_MAX = 10;
  const CELL_SIZE_STEP = 1;
  const CELL_SIZE_DEFAULT = 2;
  const FRAME_PREVIEW_MIN_PERCENT = 50;
  const FRAME_PREVIEW_MAX_PERCENT = 150;
  const FRAME_PREVIEW_STEP_PERCENT = 10;
  const FRAME_PREVIEW_DEFAULT_PERCENT = 90;
  const CAMERA_GRID_CSS_SIZE = 220;
  const ACTION_LINE_COLORS = [
    "#1f5aa6",
    "#c84c24",
    "#2f8f46",
    "#7a3db8",
    "#c59a00",
    "#118a8a",
    "#a14076",
    "#5d5d5d",
  ];

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function decodeBase64(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  function decodeUint8Base64(base64) {
    return decodeBase64(base64);
  }

  function float16ToFloat32(bits) {
    const sign = (bits & 0x8000) ? -1 : 1;
    const exponent = (bits >> 10) & 0x1f;
    const fraction = bits & 0x03ff;
    if (exponent === 0) {
      if (fraction === 0) return sign * 0;
      return sign * Math.pow(2, -14) * (fraction / 1024);
    }
    if (exponent === 31) {
      return fraction === 0 ? sign * Infinity : NaN;
    }
    return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
  }

  function decodeFloat16Base64(base64) {
    const bytes = decodeBase64(base64);
    const words = new Uint16Array(bytes.buffer);
    const values = new Float32Array(words.length);
    for (let i = 0; i < words.length; i++) {
      values[i] = float16ToFloat32(words[i]);
    }
    return values;
  }

  function decodeFloat32Base64(base64) {
    const bytes = decodeBase64(base64);
    return new Float32Array(bytes.buffer);
  }

  function percentile(sortedValues, q) {
    if (sortedValues.length === 0) return 0;
    const idx = clamp(Math.floor((sortedValues.length - 1) * q), 0, sortedValues.length - 1);
    return sortedValues[idx];
  }

  function lerp(a, b, t) {
    return Math.round(a + (b - a) * t);
  }

  function devicePixelRatioValue() {
    return window.devicePixelRatio || 1;
  }

  function divergingRgb(normalized) {
    const t = clamp(normalized, 0, 1);
    const blue = [45, 92, 255];
    const white = [255, 255, 255];
    const red = [214, 41, 41];
    if (t <= 0.5) {
      const u = t / 0.5;
      return [
        lerp(blue[0], white[0], u),
        lerp(blue[1], white[1], u),
        lerp(blue[2], white[2], u),
      ];
    }
    const u = (t - 0.5) / 0.5;
    return [
      lerp(white[0], red[0], u),
      lerp(white[1], red[1], u),
      lerp(white[2], red[2], u),
    ];
  }

  function cameraHeatRgb(probability, maxProbability) {
    if (!(maxProbability > 0) || !(probability > 0)) {
      return [255, 251, 244];
    }
    const t = Math.sqrt(clamp(probability / maxProbability, 0, 1));
    const low = [255, 251, 244];
    const mid = [244, 188, 87];
    const high = [120, 42, 13];
    if (t <= 0.65) {
      const u = t / 0.65;
      return [
        lerp(low[0], mid[0], u),
        lerp(low[1], mid[1], u),
        lerp(low[2], mid[2], u),
      ];
    }
    const u = (t - 0.65) / 0.35;
    return [
      lerp(mid[0], high[0], u),
      lerp(mid[1], high[1], u),
      lerp(mid[2], high[2], u),
    ];
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatFloat(value) {
    if (!Number.isFinite(Number(value))) {
      return "n/a";
    }
    return Number(value).toFixed(6);
  }

  function formatRolloutInfoValueHtml(value) {
    if (Array.isArray(value)) {
      return value.length
        ? value.map((item) => escapeHtml(item)).join("<br>")
        : "<span style=\"color:#675f56;\">none</span>";
    }
    if (value == null || value === "") {
      return "<span style=\"color:#675f56;\">n/a</span>";
    }
    return escapeHtml(String(value));
  }

  function swatchHtml(color) {
    return "<span style=\"display:inline-block;width:10px;height:10px;border:1px solid rgba(0,0,0,0.2);background:" + color + ";vertical-align:middle;margin-right:6px;\"></span>";
  }

  window.addEventListener("DOMContentLoaded", function () {
    const config = window.__ATTN_VIEWER_CONFIG || {};
    const DIFF_STRATEGY = config.diffStrategy || "linear";
    const DIFFERENCE_LABEL = config.differenceLabel || ("Difference (" + DIFF_STRATEGY + ")");
    const MANIFEST = window.__ATTN_ROLLOUT_MANIFEST || [];
    const DEFAULT_ROLLOUT_ID = window.__ATTN_DEFAULT_ROLLOUT || (MANIFEST[0] ? MANIFEST[0].id : null);
    const ROLLOUT_MAP = Object.fromEntries(MANIFEST.map((entry) => [entry.id, entry]));
    const bundles = window.__ATTN_ROLLOUT_BUNDLES = window.__ATTN_ROLLOUT_BUNDLES || {};
    const bundlePromises = {};
    const bundleScripts = {};

    let currentRolloutEntry = null;
    let currentBundle = null;
    let activationToken = 0;
    let cellSize = CELL_SIZE_DEFAULT;
    let framePreviewPercent = FRAME_PREVIEW_DEFAULT_PERCENT;
    let lastHoverFrame = 0;
    let lastHoverPosition = null;
    let selectedActionValues = [];
    let actionPlotStyle = "lines";

    const heatmapCanvas = document.getElementById("heatmapCanvas");
    const heatmapCtx = heatmapCanvas.getContext("2d", { alpha: false });
    const actionPlotCanvas = document.getElementById("actionPlotCanvas");
    const actionPlotCtx = actionPlotCanvas.getContext("2d", { alpha: false });
    const cameraGridCanvas = document.getElementById("cameraGridCanvas");
    const cameraGridCtx = cameraGridCanvas.getContext("2d", { alpha: false });
    const tooltip = document.getElementById("tooltip");
    const scrollShell = document.getElementById("scrollShell");
    const positionAxisNote = document.getElementById("positionAxisNote");
    const actionAxisNote = document.getElementById("actionAxisNote");
    const frameAxisNote = document.getElementById("frameAxisNote");
    const actionLegend = document.getElementById("actionLegend");
    const cameraGridCaption = document.getElementById("cameraGridCaption");

    const rolloutSelect = document.getElementById("rolloutSelect");
    const layerSelect = document.getElementById("layerSelect");
    const headSelect = document.getElementById("headSelect");
    const modeSelect = document.getElementById("modeSelect");
    const actionMetricRows = document.getElementById("actionMetricRows");
    const addActionMetricButton = document.getElementById("addActionMetricButton");
    const scaleDecrease = document.getElementById("scaleDecrease");
    const scaleIncrease = document.getElementById("scaleIncrease");
    const scaleReadout = document.getElementById("scaleReadout");
    const plotStyleLines = document.getElementById("plotStyleLines");
    const plotStyleBars = document.getElementById("plotStyleBars");
    const hoverTabButton = document.getElementById("hoverTabButton");
    const rolloutInfoTabButton = document.getElementById("rolloutInfoTabButton");
    const hoverTabPanel = document.getElementById("hoverTabPanel");
    const rolloutInfoTabPanel = document.getElementById("rolloutInfoTabPanel");
    const hoverStats = document.getElementById("hoverStats");
    const rolloutInfo = document.getElementById("rolloutInfo");
    const currentFrameImg = document.getElementById("currentFrameImg");
    const currentFrameNumber = document.getElementById("currentFrameNumber");
    const attendedFrameImg = document.getElementById("attendedFrameImg");
    const attendedFrameNumber = document.getElementById("attendedFrameNumber");
    const frameSizeDecrease = document.getElementById("frameSizeDecrease");
    const frameSizeIncrease = document.getElementById("frameSizeIncrease");
    const frameSizeReadout = document.getElementById("frameSizeReadout");
    let activeStatsTab = "hover";

    function activeMeta() {
      return currentBundle ? currentBundle.metadata : null;
    }

    function activeAttentionValues() {
      return currentBundle ? currentBundle.attnValues : null;
    }

    function activeActionSeriesValues() {
      return currentBundle ? currentBundle.actionSeriesValues : null;
    }

    function activePerformedButtonsValues() {
      return currentBundle ? currentBundle.performedButtonsValues : null;
    }

    function activePerformedCameraValues() {
      return currentBundle ? currentBundle.performedCameraValues : null;
    }

    function setStatsTab(tabId) {
      activeStatsTab = tabId;
      const showingHover = tabId === "hover";
      hoverTabButton.classList.toggle("active", showingHover);
      rolloutInfoTabButton.classList.toggle("active", !showingHover);
      hoverTabPanel.hidden = !showingHover;
      rolloutInfoTabPanel.hidden = showingHover;
    }

    function renderRolloutInfo() {
      const meta = activeMeta();
      const info = meta && meta.rolloutInfo;
      if (!info) {
        rolloutInfo.textContent = currentRolloutEntry ? "Loading rollout metadata..." : "Select a rollout to inspect its metadata.";
        return;
      }

      const rows = Array.isArray(info.rows) ? info.rows : [];
      let html = "<div class=\"rollout-info-grid\">";
      for (const row of rows) {
        html += "<div class=\"rollout-info-label\">" + escapeHtml(row.label) + "</div>";
        html += "<div class=\"rollout-info-value\">" + formatRolloutInfoValueHtml(row.value) + "</div>";
      }
      html += "</div>";
      if (info.rawCommand) {
        html += "<div class=\"rollout-info-command-title\">Command</div>";
        html += "<pre class=\"rollout-info-command\">" + escapeHtml(info.rawCommand) + "</pre>";
      }
      rolloutInfo.innerHTML = html;
    }

    function selectedScale() {
      return cellSize;
    }

    function selectedActionPlotStyle() {
      return actionPlotStyle;
    }

    function selectedMode() {
      return modeSelect.value;
    }

    function selectedModeLabel() {
      const option = modeSelect.options[modeSelect.selectedIndex];
      return option ? option.text : "";
    }

    function selectedLayerValue() {
      return layerSelect.value;
    }

    function selectedHeadValue() {
      return headSelect.value;
    }

    function selectedLayerIndices() {
      const meta = activeMeta();
      if (!meta) return [];
      if (selectedLayerValue() === "all") {
        return Array.from({ length: meta.layerCount }, (_, i) => i);
      }
      return [Number(selectedLayerValue())];
    }

    function selectedHeadIndices() {
      const meta = activeMeta();
      if (!meta) return [];
      if (selectedHeadValue() === "all") {
        return Array.from({ length: meta.configuredHeads }, (_, i) => i);
      }
      return [Number(selectedHeadValue())];
    }

    function selectedBranchIndex() {
      const mode = selectedMode();
      if (mode === "conditional") return 0;
      if (mode === "unconditional") return 1;
      if (mode.startsWith("branch_")) return Number(mode.split("_")[1]);
      return 0;
    }

    function selectionLabel() {
      const layerText = selectedLayerValue() === "all" ? "All Layers (max)" : "Layer " + selectedLayerValue();
      const headText = selectedHeadValue() === "all" ? "All Heads (max)" : "Head " + selectedHeadValue();
      return layerText + " / " + headText;
    }

    function actionOptionByValue(value) {
      const meta = activeMeta();
      if (!meta || !meta.actionPlotOptions) return null;
      return meta.actionPlotOptions.find((option) => option.value === value) || null;
    }

    function normalizeSelectedActionValues(preferredValues) {
      const meta = activeMeta();
      if (!meta || !meta.actionPlotOptions || !meta.actionPlotOptions.length) {
        return [];
      }
      const validValues = new Set(meta.actionPlotOptions.map((option) => option.value));
      const fallbackValues = Array.isArray(meta.defaultActionValues) && meta.defaultActionValues.length
        ? meta.defaultActionValues.slice()
        : [meta.defaultActionValue || meta.actionPlotOptions[0].value];
      const source = Array.isArray(preferredValues) && preferredValues.length
        ? preferredValues
        : (selectedActionValues.length ? selectedActionValues : fallbackValues);
      const normalized = [];
      for (const value of source) {
        if (validValues.has(value)) {
          normalized.push(value);
        }
      }
      if (!normalized.length) {
        normalized.push(validValues.has(fallbackValues[0]) ? fallbackValues[0] : meta.actionPlotOptions[0].value);
      }
      return normalized;
    }

    function nextUnusedActionValue() {
      const meta = activeMeta();
      if (!meta || !meta.actionPlotOptions || !meta.actionPlotOptions.length) {
        return null;
      }
      for (const option of meta.actionPlotOptions) {
        if (!selectedActionValues.includes(option.value)) {
          return option.value;
        }
      }
      return meta.actionPlotOptions[0].value;
    }

    function selectedActionDescriptors() {
      return selectedActionValues.map((value, index) => {
        const option = actionOptionByValue(value);
        if (!option) return null;
        return {
          color: ACTION_LINE_COLORS[index % ACTION_LINE_COLORS.length],
          index: option.index,
          label: option.label,
          value: option.value,
        };
      }).filter(Boolean);
    }

    function selectedActionValuesForFrame(frame) {
      const meta = activeMeta();
      const actionValues = activeActionSeriesValues();
      if (!meta || !actionValues) return [];
      return selectedActionDescriptors().map((descriptor) => ({
        color: descriptor.color,
        label: descriptor.label,
        metricValue: actionValues[descriptor.index * meta.frameCount + frame],
        value: descriptor.value,
      }));
    }

    function performedButtonsForFrame(frame) {
      const meta = activeMeta();
      const performedButtons = activePerformedButtonsValues();
      if (!meta || !performedButtons || !meta.performedButtonNames) {
        return [];
      }
      const width = meta.performedButtonNames.length;
      const offset = frame * width;
      return meta.performedButtonNames.map((name, index) => ({
        name,
        value: performedButtons[offset + index],
      }));
    }

    function performedCameraForFrame(frame) {
      const performedCamera = activePerformedCameraValues();
      if (!performedCamera) {
        return { pitch: 0, yaw: 0, active: false };
      }
      const offset = frame * 2;
      const pitch = performedCamera[offset];
      const yaw = performedCamera[offset + 1];
      return {
        pitch,
        yaw,
        active: Math.abs(pitch) > 1e-9 || Math.abs(yaw) > 1e-9,
      };
    }

    function formatActiveButtons(buttonStates) {
      const activeButtons = buttonStates.filter((button) => button.value === 1).map((button) => button.name);
      return activeButtons.length ? activeButtons.join(", ") : "none";
    }

    function formatPerformedCamera(cameraState) {
      return (cameraState.active ? "non-zero" : "zero") +
        " (pitch=" + formatFloat(cameraState.pitch) + ", yaw=" + formatFloat(cameraState.yaw) + ")";
    }

    function setControlsDisabled(disabled) {
      rolloutSelect.disabled = false;
      layerSelect.disabled = disabled;
      headSelect.disabled = disabled;
      modeSelect.disabled = disabled;
      plotStyleLines.disabled = disabled;
      plotStyleBars.disabled = disabled;
      for (const select of actionMetricRows.querySelectorAll("select")) {
        select.disabled = disabled;
      }
      for (const button of actionMetricRows.querySelectorAll("button")) {
        const isOnlyRowRemove = button.dataset.role === "remove" && selectedActionValues.length <= 1;
        button.disabled = disabled || isOnlyRowRemove;
      }
      addActionMetricButton.disabled = disabled;
      scaleDecrease.disabled = disabled || cellSize <= CELL_SIZE_MIN;
      scaleIncrease.disabled = disabled || cellSize >= CELL_SIZE_MAX;
    }

    function selectedMetricRange() {
      const meta = activeMeta();
      const values = activeActionSeriesValues();
      const descriptors = selectedActionDescriptors();
      if (!meta || !values || !descriptors.length) {
        return { rawMin: 0, rawMax: 1, min: 0, max: 1, span: 1 };
      }

      let rawMin = Infinity;
      let rawMax = -Infinity;
      for (const descriptor of descriptors) {
        const offset = descriptor.index * meta.frameCount;
        for (let frame = 0; frame < meta.frameCount; frame++) {
          const metricValue = values[offset + frame];
          if (!Number.isFinite(metricValue)) continue;
          if (metricValue < rawMin) rawMin = metricValue;
          if (metricValue > rawMax) rawMax = metricValue;
        }
      }

      if (!Number.isFinite(rawMin) || !Number.isFinite(rawMax)) {
        return { rawMin: 0, rawMax: 1, min: 0, max: 1, span: 1 };
      }

      let min = rawMin;
      let max = rawMax;
      if (min > 0) min = 0;
      if (max < 0) max = 0;
      if (min === max) {
        const pad = min === 0 ? 1 : Math.abs(min) * 0.1;
        min -= pad;
        max += pad;
      }
      return { rawMin, rawMax, min, max, span: max - min };
    }

    function updateActionAxisNote() {
      const meta = activeMeta();
      const values = activeActionSeriesValues();
      const descriptors = selectedActionDescriptors();
      if (!meta || !values || !descriptors.length) {
        actionAxisNote.textContent = "Selected line value: -";
        return;
      }
      const range = selectedMetricRange();
      actionAxisNote.textContent = "Selected line value: " + formatFloat(range.rawMin) + " to " + formatFloat(range.rawMax);
    }

    function updateAxisNotes() {
      const meta = activeMeta();
      if (!meta) {
        positionAxisNote.textContent = "Attention position: -";
        actionAxisNote.textContent = "Selected line value: -";
        frameAxisNote.textContent = "Episode frame: -";
        return;
      }
      positionAxisNote.textContent = "Attention position: 0 to " + (meta.contextPositions - 1);
      frameAxisNote.textContent = "Episode frame: 0 to " + (meta.frameCount - 1);
      updateActionAxisNote();
    }

    function updateScaleReadout() {
      scaleReadout.textContent = String(cellSize) + " px";
    }

    function updatePlotStyleButtons() {
      const showingLines = selectedActionPlotStyle() === "lines";
      plotStyleLines.classList.toggle("active", showingLines);
      plotStyleBars.classList.toggle("active", !showingLines);
      plotStyleLines.setAttribute("aria-pressed", showingLines ? "true" : "false");
      plotStyleBars.setAttribute("aria-pressed", showingLines ? "false" : "true");
    }

    function resizeCameraGridCanvas() {
      const dpr = devicePixelRatioValue();
      const pixelSize = Math.max(1, Math.round(CAMERA_GRID_CSS_SIZE * dpr));
      if (cameraGridCanvas.width !== pixelSize) {
        cameraGridCanvas.width = pixelSize;
      }
      if (cameraGridCanvas.height !== pixelSize) {
        cameraGridCanvas.height = pixelSize;
      }
      cameraGridCanvas.style.width = String(CAMERA_GRID_CSS_SIZE) + "px";
      cameraGridCanvas.style.height = String(CAMERA_GRID_CSS_SIZE) + "px";
      cameraGridCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cameraGridCtx.imageSmoothingEnabled = false;
    }

    function clearHeatmap() {
      heatmapCtx.fillStyle = "#ffffff";
      heatmapCtx.fillRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
    }

    function actionPlotCssWidth() {
      const meta = activeMeta();
      return meta ? meta.frameCount * selectedScale() : 1000;
    }

    function resizeActionPlotCanvas() {
      const cssWidth = actionPlotCssWidth();
      const cssHeight = ACTION_PLOT_HEIGHT;
      const dpr = devicePixelRatioValue();
      const pixelWidth = Math.max(1, Math.round(cssWidth * dpr));
      const pixelHeight = Math.max(1, Math.round(cssHeight * dpr));

      if (actionPlotCanvas.width !== pixelWidth) {
        actionPlotCanvas.width = pixelWidth;
      }
      if (actionPlotCanvas.height !== pixelHeight) {
        actionPlotCanvas.height = pixelHeight;
      }

      actionPlotCanvas.style.width = String(cssWidth) + "px";
      actionPlotCanvas.style.height = String(cssHeight) + "px";
      actionPlotCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      actionPlotCtx.imageSmoothingEnabled = false;
    }

    function clearActionPlot() {
      resizeActionPlotCanvas();
      actionPlotCtx.save();
      actionPlotCtx.setTransform(1, 0, 0, 1, 0, 0);
      actionPlotCtx.clearRect(0, 0, actionPlotCanvas.width, actionPlotCanvas.height);
      actionPlotCtx.restore();
      actionPlotCtx.fillStyle = "#fffdf8";
      actionPlotCtx.fillRect(0, 0, actionPlotCssWidth(), ACTION_PLOT_HEIGHT);
    }

    function clearCameraGrid() {
      resizeCameraGridCanvas();
      cameraGridCtx.fillStyle = "#fffaf3";
      cameraGridCtx.fillRect(0, 0, CAMERA_GRID_CSS_SIZE, CAMERA_GRID_CSS_SIZE);
      cameraGridCaption.textContent = "Hover over the attention map to inspect the camera distribution for the current frame.";
    }

    function populateRolloutOptions() {
      rolloutSelect.innerHTML = "";
      for (const entry of MANIFEST) {
        const option = document.createElement("option");
        option.value = entry.id;
        option.textContent = entry.label;
        rolloutSelect.appendChild(option);
      }
    }

    function setFlatSelectOptions(select, options, preferredValue) {
      select.innerHTML = "";
      for (const opt of options) {
        const option = document.createElement("option");
        option.value = opt.value;
        option.textContent = opt.label;
        select.appendChild(option);
      }
      if (!options.length) return;
      const preferredExists = options.some((option) => option.value === preferredValue);
      select.value = preferredExists ? preferredValue : options[0].value;
    }

    function populateLayerOptions(preferredValue) {
      const meta = activeMeta();
      const options = Array.from({ length: meta.layerCount }, (_, i) => ({ value: String(i), label: String(i) }));
      options.push({ value: "all", label: "All Layers (max)" });
      setFlatSelectOptions(layerSelect, options, preferredValue);
    }

    function populateHeadOptions(preferredValue) {
      const meta = activeMeta();
      const options = Array.from({ length: meta.configuredHeads }, (_, i) => ({ value: String(i), label: String(i) }));
      options.push({ value: "all", label: "All Heads (max)" });
      setFlatSelectOptions(headSelect, options, preferredValue);
    }

    function populateModeOptions(preferredValue) {
      const meta = activeMeta();
      let options;
      if (meta.branchCount === 2) {
        options = [
          { value: "conditional", label: "Conditional" },
          { value: "unconditional", label: "Unconditional" },
          { value: "difference", label: DIFFERENCE_LABEL },
        ];
      } else {
        options = meta.branchLabels.map((label, index) => ({
          value: "branch_" + index,
          label: label.charAt(0).toUpperCase() + label.slice(1),
        }));
      }
      setFlatSelectOptions(modeSelect, options, preferredValue);
    }

    function appendActionSelectOptions(select, selectedValue) {
      const meta = activeMeta();
      if (!meta || !meta.actionPlotOptions) return;
      let currentGroup = null;
      let currentOptgroup = null;
      for (const optionData of meta.actionPlotOptions) {
        if (optionData.group !== currentGroup) {
          currentGroup = optionData.group;
          currentOptgroup = document.createElement("optgroup");
          currentOptgroup.label = currentGroup;
          select.appendChild(currentOptgroup);
        }
        const option = document.createElement("option");
        option.value = optionData.value;
        option.textContent = optionData.label;
        currentOptgroup.appendChild(option);
      }
      const fallbackValue = meta.defaultActionValue || (meta.actionPlotOptions[0] && meta.actionPlotOptions[0].value);
      select.value = actionOptionByValue(selectedValue) ? selectedValue : fallbackValue;
    }

    function renderActionLegend() {
      const descriptors = selectedActionDescriptors();
      updateActionAxisNote();
      if (descriptors.length <= 1) {
        actionLegend.innerHTML = "";
        actionLegend.style.display = "none";
        return;
      }
      actionLegend.style.display = "flex";
      actionLegend.innerHTML = descriptors.map((descriptor) =>
        "<span class=\"legend-item\">" +
        "<span class=\"legend-swatch\" style=\"background:" + descriptor.color + ";\"></span>" +
        "<span>" + escapeHtml(descriptor.label) + "</span>" +
        "</span>"
      ).join("");
    }

    function renderActionMetricControls(preferredValues) {
      const meta = activeMeta();
      actionMetricRows.innerHTML = "";
      if (!meta || !meta.actionPlotOptions || !meta.actionPlotOptions.length) {
        selectedActionValues = [];
        renderActionLegend();
        return;
      }

      selectedActionValues = normalizeSelectedActionValues(preferredValues);

      selectedActionValues.forEach((value, index) => {
        const row = document.createElement("div");
        row.className = "action-row";

        const select = document.createElement("select");
        appendActionSelectOptions(select, value);
        select.value = value;
        select.addEventListener("change", function () {
          selectedActionValues[index] = select.value;
          renderActionLegend();
          syncPreviewToCurrentHover();
        });

        const removeButton = document.createElement("button");
        removeButton.type = "button";
        removeButton.className = "tiny-button";
        removeButton.textContent = "-";
        removeButton.dataset.role = "remove";
        removeButton.disabled = selectedActionValues.length <= 1;
        removeButton.addEventListener("click", function () {
          if (selectedActionValues.length <= 1) return;
          selectedActionValues.splice(index, 1);
          renderActionMetricControls(selectedActionValues);
          syncPreviewToCurrentHover();
        });

        row.appendChild(select);
        row.appendChild(removeButton);
        actionMetricRows.appendChild(row);
      });

      renderActionLegend();
    }

    function ensureAttentionValues(bundle) {
      if (!bundle.attnValues) {
        bundle.attnValues = decodeFloat16Base64(bundle.attnBase64);
        bundle.attnBase64 = null;
      }
      return bundle.attnValues;
    }

    function ensureActionSeriesValues(bundle) {
      if (!bundle.actionSeriesValues) {
        bundle.actionSeriesValues = decodeFloat16Base64(bundle.actionSeriesBase64);
        bundle.actionSeriesBase64 = null;
      }
      return bundle.actionSeriesValues;
    }

    function ensurePerformedButtonsValues(bundle) {
      if (!bundle.performedButtonsValues) {
        bundle.performedButtonsValues = decodeUint8Base64(bundle.performedButtonsBase64);
        bundle.performedButtonsBase64 = null;
      }
      return bundle.performedButtonsValues;
    }

    function ensurePerformedCameraValues(bundle) {
      if (!bundle.performedCameraValues) {
        bundle.performedCameraValues = decodeFloat32Base64(bundle.performedCameraBase64);
        bundle.performedCameraBase64 = null;
      }
      return bundle.performedCameraValues;
    }

    function purgeInactiveBundles(activeId) {
      for (const entry of MANIFEST) {
        if (entry.id === activeId) continue;
        delete bundles[entry.id];
        delete bundlePromises[entry.id];
        if (bundleScripts[entry.id]) {
          bundleScripts[entry.id].remove();
          delete bundleScripts[entry.id];
        }
      }
    }

    function ensureRolloutBundle(entry) {
      if (bundles[entry.id]) {
        return Promise.resolve(bundles[entry.id]);
      }
      if (bundlePromises[entry.id]) {
        return bundlePromises[entry.id];
      }
      bundlePromises[entry.id] = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = entry.bundlePath;
        script.async = true;
        script.onload = () => {
          if (bundles[entry.id]) {
            bundleScripts[entry.id] = script;
            resolve(bundles[entry.id]);
          } else {
            reject(new Error("Rollout bundle loaded but did not register: " + entry.id));
          }
        };
        script.onerror = () => reject(new Error("Failed to load rollout bundle: " + entry.label));
        document.head.appendChild(script);
      });
      return bundlePromises[entry.id];
    }

    function arrayIndex(frame, layer, loggedMap, position) {
      const meta = activeMeta();
      return (((frame * meta.layerCount + layer) * meta.loggedMapsPerLayer + loggedMap) * meta.contextPositions + position);
    }

    function rawAttentionValue(frame, layer, loggedMap, position) {
      return activeAttentionValues()[arrayIndex(frame, layer, loggedMap, position)];
    }

    function branchValue(frame, position, branchIndex) {
      const meta = activeMeta();
      const layers = selectedLayerIndices();
      const heads = selectedHeadIndices();
      let maxValue = -Infinity;
      for (let li = 0; li < layers.length; li++) {
        const layer = layers[li];
        for (let hi = 0; hi < heads.length; hi++) {
          const head = heads[hi];
          const value = rawAttentionValue(frame, layer, branchIndex * meta.configuredHeads + head, position);
          if (value > maxValue) {
            maxValue = value;
          }
        }
      }
      return maxValue;
    }

    function conditionalValue(frame, position) {
      return branchValue(frame, position, 0);
    }

    function unconditionalValue(frame, position) {
      return branchValue(frame, position, 1);
    }

    function framePath(frameIndex) {
      const meta = activeMeta();
      const clamped = clamp(frameIndex, 0, meta.frameCount - 1);
      return currentRolloutEntry.framesDir + "/frame_" + String(clamped).padStart(6, "0") + ".png";
    }

    function attendedFrameIndex(currentFrame, position) {
      const meta = activeMeta();
      return clamp(currentFrame - (meta.contextPositions - position), 0, meta.frameCount - 1);
    }

    function applyScale() {
      const meta = activeMeta();
      if (!meta) return;
      const scale = selectedScale();
      heatmapCanvas.style.width = String(meta.frameCount * scale) + "px";
      heatmapCanvas.style.height = String(meta.contextPositions * scale) + "px";
      resizeActionPlotCanvas();
    }

    function applyFramePreviewScale() {
      currentFrameImg.style.width = String(framePreviewPercent) + "%";
      attendedFrameImg.style.width = String(framePreviewPercent) + "%";
      frameSizeReadout.textContent = String(framePreviewPercent) + "%";
      frameSizeDecrease.disabled = framePreviewPercent <= FRAME_PREVIEW_MIN_PERCENT;
      frameSizeIncrease.disabled = framePreviewPercent >= FRAME_PREVIEW_MAX_PERCENT;
    }

    function buildDifferenceGrid() {
      const meta = activeMeta();
      const diffs = new Float32Array(meta.frameCount * meta.contextPositions);
      const absVals = new Float32Array(meta.frameCount * meta.contextPositions);
      let maxAbs = 0;
      let p = 0;
      for (let y = 0; y < meta.contextPositions; y++) {
        for (let x = 0; x < meta.frameCount; x++) {
          const diff = conditionalValue(x, y) - unconditionalValue(x, y);
          const absDiff = Math.abs(diff);
          diffs[p] = diff;
          absVals[p] = absDiff;
          if (absDiff > maxAbs) maxAbs = absDiff;
          p += 1;
        }
      }
      const sortedAbs = Array.from(absVals).sort((a, b) => a - b);
      return { diffs, maxAbs, p995: percentile(sortedAbs, 0.995) };
    }

    function mapDifferenceToRgb(diff, stats) {
      if (stats.maxAbs === 0) {
        return [255, 255, 255];
      }
      let normalized;
      if (DIFF_STRATEGY === "linear") {
        normalized = 0.5 + diff / (2 * stats.maxAbs);
      } else if (DIFF_STRATEGY === "percentile") {
        const scale = stats.p995 > 0 ? stats.p995 : stats.maxAbs;
        const clipped = clamp(diff, -scale, scale);
        normalized = 0.5 + clipped / (2 * scale);
      } else {
        const scale = stats.p995 > 0 ? stats.p995 / 6 : stats.maxAbs / 6;
        const safe = scale > 0 ? scale : 1e-6;
        const denom = Math.asinh(stats.maxAbs / safe);
        normalized = 0.5 + Math.asinh(diff / safe) / (2 * denom);
      }
      return divergingRgb(normalized);
    }

    function renderHeatmap() {
      const meta = activeMeta();
      if (!meta || !currentBundle || !activeAttentionValues()) {
        clearHeatmap();
        return;
      }

      const mode = selectedMode();
      const imageData = heatmapCtx.createImageData(meta.frameCount, meta.contextPositions);
      let ptr = 0;
      let diffStats = null;
      if (mode === "difference") {
        diffStats = buildDifferenceGrid();
      }

      let diffPtr = 0;
      for (let y = 0; y < meta.contextPositions; y++) {
        for (let x = 0; x < meta.frameCount; x++) {
          let r, g, b;
          if (mode === "conditional") {
            const shade = 255 - clamp(Math.round(conditionalValue(x, y) * 255), 0, 255);
            r = g = b = shade;
          } else if (mode === "unconditional") {
            const shade = 255 - clamp(Math.round(unconditionalValue(x, y) * 255), 0, 255);
            r = g = b = shade;
          } else if (mode.startsWith("branch_")) {
            const shade = 255 - clamp(Math.round(branchValue(x, y, selectedBranchIndex()) * 255), 0, 255);
            r = g = b = shade;
          } else {
            const rgb = mapDifferenceToRgb(diffStats.diffs[diffPtr], diffStats);
            r = rgb[0];
            g = rgb[1];
            b = rgb[2];
            diffPtr += 1;
          }
          imageData.data[ptr++] = r;
          imageData.data[ptr++] = g;
          imageData.data[ptr++] = b;
          imageData.data[ptr++] = 255;
        }
      }

      heatmapCtx.putImageData(imageData, 0, 0);
      applyScale();
    }

    function renderActionPlot() {
      const meta = activeMeta();
      const values = activeActionSeriesValues();
      const descriptors = selectedActionDescriptors();
      clearActionPlot();
      if (!meta || !currentBundle || !values || !descriptors.length) {
        return;
      }

      const scale = selectedScale();
      const plotWidth = actionPlotCssWidth();
      const height = ACTION_PLOT_HEIGHT;
      const range = selectedMetricRange();
      const metricToY = function (metricValue) {
        return clamp(((range.max - metricValue) / range.span) * (height - 1), 0, height - 1);
      };
      const zeroY = metricToY(0);
      actionPlotCtx.strokeStyle = "#d9d1c4";
      actionPlotCtx.lineWidth = 1;
      for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
        const y = Math.round(frac * (height - 1)) + 0.5;
        actionPlotCtx.beginPath();
        actionPlotCtx.moveTo(0, y);
        actionPlotCtx.lineTo(plotWidth, y);
        actionPlotCtx.stroke();
      }
      if (range.min < 0 && range.max > 0) {
        actionPlotCtx.strokeStyle = "#a89c8a";
        actionPlotCtx.lineWidth = 1.2;
        actionPlotCtx.beginPath();
        actionPlotCtx.moveTo(0, zeroY + 0.5);
        actionPlotCtx.lineTo(plotWidth, zeroY + 0.5);
        actionPlotCtx.stroke();
      }

      if (selectedActionPlotStyle() === "bars") {
        const barSlotWidth = Math.max(scale / Math.max(descriptors.length, 1), 1);
        for (let descriptorIndex = 0; descriptorIndex < descriptors.length; descriptorIndex++) {
          const descriptor = descriptors[descriptorIndex];
          actionPlotCtx.fillStyle = descriptor.color;
          actionPlotCtx.globalAlpha = 0.72;
          for (let frame = 0; frame < meta.frameCount; frame++) {
            const metricValue = values[descriptor.index * meta.frameCount + frame];
            if (!Number.isFinite(metricValue)) continue;
            const y = metricToY(metricValue);
            const left = frame * scale + descriptorIndex * barSlotWidth;
            const barWidth = Math.max(1, barSlotWidth - 1);
            const top = Math.min(y, zeroY);
            const barHeight = Math.max(1, Math.abs(zeroY - y));
            actionPlotCtx.fillRect(left, top, barWidth, barHeight);
          }
        }
        actionPlotCtx.globalAlpha = 1;
      } else {
        for (const descriptor of descriptors) {
          actionPlotCtx.strokeStyle = descriptor.color;
          actionPlotCtx.lineWidth = 1.5;
          actionPlotCtx.beginPath();
          let hasSegment = false;
          for (let frame = 0; frame < meta.frameCount; frame++) {
            const metricValue = values[descriptor.index * meta.frameCount + frame];
            if (!Number.isFinite(metricValue)) {
              hasSegment = false;
              continue;
            }
            const y = metricToY(metricValue);
            const x = frame * scale + (scale / 2);
            if (!hasSegment) {
              actionPlotCtx.moveTo(x, y + 0.5);
              hasSegment = true;
            } else {
              actionPlotCtx.lineTo(x, y + 0.5);
            }
          }
          actionPlotCtx.stroke();
        }
      }

      const highlightFrame = clamp(lastHoverFrame, 0, meta.frameCount - 1);
      const highlightX = highlightFrame * scale + (scale / 2);
      actionPlotCtx.strokeStyle = "#bb4b31";
      actionPlotCtx.lineWidth = 1;
      actionPlotCtx.beginPath();
      actionPlotCtx.moveTo(highlightX, 0);
      actionPlotCtx.lineTo(highlightX, height);
      actionPlotCtx.stroke();

      if (selectedActionPlotStyle() === "lines") {
        for (const descriptor of descriptors) {
          const metricValue = values[descriptor.index * meta.frameCount + highlightFrame];
          if (!Number.isFinite(metricValue)) continue;
          const y = metricToY(metricValue);
          actionPlotCtx.fillStyle = descriptor.color;
          actionPlotCtx.beginPath();
          actionPlotCtx.arc(highlightX, y + 0.5, 3, 0, Math.PI * 2);
          actionPlotCtx.fill();
        }
      }
    }

    function nearestCameraBinIndex(value, binValues) {
      let bestIndex = 0;
      let bestDistance = Infinity;
      for (let i = 0; i < binValues.length; i++) {
        const distance = Math.abs(binValues[i] - value);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = i;
        }
      }
      return bestIndex;
    }

    function renderCameraGrid(frame) {
      const meta = activeMeta();
      const actionValues = activeActionSeriesValues();
      resizeCameraGridCanvas();
      cameraGridCtx.fillStyle = "#fffaf3";
      cameraGridCtx.fillRect(0, 0, CAMERA_GRID_CSS_SIZE, CAMERA_GRID_CSS_SIZE);

      if (!meta || !actionValues || !meta.cameraJointToFactored || !meta.cameraJointCount) {
        cameraGridCaption.textContent = "Hover over the attention map to inspect the camera distribution for the current frame.";
        return null;
      }

      const side = meta.cameraGridSide || 11;
      const cellSize = CAMERA_GRID_CSS_SIZE / side;
      const gridValues = new Float32Array(side * side);
      let maxProbability = 0;
      let maxPitchIndex = 0;
      let maxYawIndex = 0;

      for (let jointIndex = 0; jointIndex < meta.cameraJointCount; jointIndex++) {
        const probability = actionValues[(meta.cameraJointStartIndex + jointIndex) * meta.frameCount + frame];
        const factored = meta.cameraJointToFactored[jointIndex];
        if (!factored) continue;
        const pitchIndex = factored[0];
        const yawIndex = factored[1];
        gridValues[pitchIndex * side + yawIndex] = probability;
        if (probability > maxProbability) {
          maxProbability = probability;
          maxPitchIndex = pitchIndex;
          maxYawIndex = yawIndex;
        }
      }

      for (let pitchIndex = 0; pitchIndex < side; pitchIndex++) {
        for (let yawIndex = 0; yawIndex < side; yawIndex++) {
          const probability = gridValues[pitchIndex * side + yawIndex];
          const rgb = cameraHeatRgb(probability, maxProbability);
          cameraGridCtx.fillStyle = "rgb(" + rgb[0] + ", " + rgb[1] + ", " + rgb[2] + ")";
          cameraGridCtx.fillRect(yawIndex * cellSize, pitchIndex * cellSize, cellSize, cellSize);
        }
      }

      cameraGridCtx.strokeStyle = "rgba(67, 58, 48, 0.18)";
      cameraGridCtx.lineWidth = 1;
      for (let i = 0; i <= side; i++) {
        const coord = i * cellSize + 0.5;
        cameraGridCtx.beginPath();
        cameraGridCtx.moveTo(0, coord);
        cameraGridCtx.lineTo(CAMERA_GRID_CSS_SIZE, coord);
        cameraGridCtx.stroke();
        cameraGridCtx.beginPath();
        cameraGridCtx.moveTo(coord, 0);
        cameraGridCtx.lineTo(coord, CAMERA_GRID_CSS_SIZE);
        cameraGridCtx.stroke();
      }

      const performedCamera = performedCameraForFrame(frame);
      const performedPitchIndex = nearestCameraBinIndex(performedCamera.pitch, meta.cameraBinValues);
      const performedYawIndex = nearestCameraBinIndex(performedCamera.yaw, meta.cameraBinValues);
      cameraGridCtx.strokeStyle = performedCamera.active ? "#0f172a" : "#6b675f";
      cameraGridCtx.lineWidth = 2;
      cameraGridCtx.strokeRect(
        performedYawIndex * cellSize + 1,
        performedPitchIndex * cellSize + 1,
        cellSize - 2,
        cellSize - 2
      );

      const peakPitch = meta.cameraBinValues[maxPitchIndex];
      const peakYaw = meta.cameraBinValues[maxYawIndex];
      cameraGridCaption.innerHTML =
        "Yaw: left to right. Pitch: up to down.<br>" +
        "Color scaled to hovered frame max. Outline = performed camera bin.";

      return {
        maxProbability,
        peakPitch,
        peakYaw,
        performedCamera,
      };
    }

    function formatActionValuesHtml(actionValues) {
      return actionValues.map((entry) =>
        "<strong>" + swatchHtml(entry.color) + escapeHtml(entry.label) + "</strong>: " + formatFloat(entry.metricValue)
      ).join("<br>");
    }

    function updatePreview(frame, attended, position) {
      const cond = conditionalValue(frame, position);
      const uncond = unconditionalValue(frame, position);
      const diff = cond - uncond;
      let displayed;
      const mode = selectedMode();
      if (mode === "conditional") {
        displayed = cond;
      } else if (mode === "unconditional") {
        displayed = uncond;
      } else if (mode.startsWith("branch_")) {
        displayed = branchValue(frame, position, selectedBranchIndex());
      } else {
        displayed = diff;
      }

      const actionValues = selectedActionValuesForFrame(frame);
      const buttonStates = performedButtonsForFrame(frame);
      const performedButtons = formatActiveButtons(buttonStates);
      const cameraState = performedCameraForFrame(frame);
      renderCameraGrid(frame);

      currentFrameImg.src = framePath(frame);
      attendedFrameImg.src = framePath(attended);
      currentFrameNumber.textContent = String(frame);
      attendedFrameNumber.textContent = String(attended);

      const statsLines = [
        "<strong>Selection</strong>: " + selectionLabel(),
        "<strong>View</strong>: " + selectedModeLabel(),
        "<strong>Current frame</strong>: " + frame,
        "<strong>Attended frame</strong>: " + attended,
        "<strong>Attention position</strong>: " + position,
      ];
      if (actionValues.length) {
        statsLines.push("<strong>Selected line values</strong>:");
        statsLines.push(formatActionValuesHtml(actionValues));
      }
      statsLines.push(
        "<strong>Conditional value</strong>: " + formatFloat(cond),
        "<strong>Unconditional value</strong>: " + formatFloat(uncond),
        "<strong>Difference</strong>: " + formatFloat(diff),
        "<strong>Performed buttons</strong>: " + escapeHtml(performedButtons),
        "<strong>Performed camera</strong>: " + escapeHtml(formatPerformedCamera(cameraState))
      );
      hoverStats.innerHTML = statsLines.join("<br>");

      return {
        actionValues,
        currentFrame: frame,
        attended,
        attentionValue: displayed,
      };
    }

    function updateTooltip(event, frame, attended, position, values) {
      tooltip.style.display = "block";
      tooltip.style.left = String(event.clientX + 16) + "px";
      tooltip.style.top = String(event.clientY + 16) + "px";

      tooltip.innerHTML =
        "<strong>Current frame</strong>: " + values.currentFrame + "<br>" +
        "<strong>Attended frame</strong>: " + values.attended + "<br>" +
        "<strong>Attention value</strong>: " + formatFloat(values.attentionValue) +
        (values.actionValues.length
          ? "<br>" + values.actionValues.map((entry) =>
              "<strong>" + escapeHtml(entry.label) + "</strong>: " + formatFloat(entry.metricValue)
            ).join("<br>")
          : "");
    }

    function syncPreviewToCurrentHover() {
      const meta = activeMeta();
      if (!meta) return;
      const frame = clamp(lastHoverFrame, 0, meta.frameCount - 1);
      const position = lastHoverPosition == null ? meta.contextPositions - 1 : clamp(lastHoverPosition, 0, meta.contextPositions - 1);
      const attended = attendedFrameIndex(frame, position);
      updatePreview(frame, attended, position);
      renderActionPlot();
    }

    function onHeatmapHover(event) {
      const meta = activeMeta();
      if (!meta || !currentBundle) return;
      const rect = heatmapCanvas.getBoundingClientRect();
      const rawX = Math.floor((event.clientX - rect.left) * meta.frameCount / rect.width);
      const rawY = Math.floor((event.clientY - rect.top) * meta.contextPositions / rect.height);
      const frame = clamp(rawX, 0, meta.frameCount - 1);
      const position = clamp(rawY, 0, meta.contextPositions - 1);
      const attended = attendedFrameIndex(frame, position);
      lastHoverFrame = frame;
      lastHoverPosition = position;
      const values = updatePreview(frame, attended, position);
      renderActionPlot();
      updateTooltip(event, frame, attended, position, values);
    }

    function onActionPlotHover(event) {
      const meta = activeMeta();
      if (!meta || !currentBundle) return;
      const rect = actionPlotCanvas.getBoundingClientRect();
      const rawX = Math.floor((event.clientX - rect.left) * meta.frameCount / rect.width);
      const frame = clamp(rawX, 0, meta.frameCount - 1);
      const position = lastHoverPosition == null ? meta.contextPositions - 1 : clamp(lastHoverPosition, 0, meta.contextPositions - 1);
      const attended = attendedFrameIndex(frame, position);
      lastHoverFrame = frame;
      const values = updatePreview(frame, attended, position);
      renderActionPlot();
      updateTooltip(event, frame, attended, position, values);
    }

    function hideTooltip() {
      tooltip.style.display = "none";
    }

    async function activateRollout(rolloutId) {
      if (!ROLLOUT_MAP[rolloutId]) {
        hoverStats.textContent = "Unknown rollout: " + rolloutId;
        rolloutInfo.textContent = "Unknown rollout: " + rolloutId;
        return;
      }
      const token = ++activationToken;
      const entry = ROLLOUT_MAP[rolloutId];
      const preferredLayer = layerSelect.value || "1";
      const preferredHead = headSelect.value || "14";
      const preferredMode = modeSelect.value || "conditional";
      const preferredActions = selectedActionValues.slice();

      setControlsDisabled(true);
      hoverStats.textContent = "Loading rollout...";
      currentRolloutEntry = entry;
      rolloutInfo.textContent = "Loading rollout metadata...";
      hideTooltip();

      try {
        const bundle = await ensureRolloutBundle(entry);
        if (token !== activationToken) return;

        purgeInactiveBundles(entry.id);
        ensureAttentionValues(bundle);
        ensureActionSeriesValues(bundle);
        ensurePerformedButtonsValues(bundle);
        ensurePerformedCameraValues(bundle);

        currentRolloutEntry = entry;
        currentBundle = bundle;
        rolloutSelect.value = entry.id;
        renderRolloutInfo();

        const meta = bundle.metadata;
        heatmapCanvas.width = meta.frameCount;
        heatmapCanvas.height = meta.contextPositions;
        lastHoverFrame = 0;
        lastHoverPosition = meta.contextPositions - 1;

        updateAxisNotes();
        populateLayerOptions(preferredLayer);
        populateHeadOptions(preferredHead);
        populateModeOptions(preferredMode);
        renderActionMetricControls(preferredActions);
        renderHeatmap();
        syncPreviewToCurrentHover();
        applyFramePreviewScale();
        updateScaleReadout();
        scrollShell.scrollLeft = 0;
        setControlsDisabled(false);
      } catch (error) {
        currentRolloutEntry = null;
        currentBundle = null;
        clearHeatmap();
        clearActionPlot();
        clearCameraGrid();
        updateAxisNotes();
        hoverStats.textContent = "Failed to load rollout.";
        rolloutInfo.textContent = "Failed to load rollout metadata.";
      }
    }

    function handleAttentionSelectionChange() {
      renderHeatmap();
      syncPreviewToCurrentHover();
    }

    rolloutSelect.addEventListener("change", function () {
      activateRollout(rolloutSelect.value);
    });
    layerSelect.addEventListener("change", handleAttentionSelectionChange);
    headSelect.addEventListener("change", handleAttentionSelectionChange);
    modeSelect.addEventListener("change", handleAttentionSelectionChange);
    addActionMetricButton.addEventListener("click", function () {
      const nextValue = nextUnusedActionValue();
      if (!nextValue) return;
      selectedActionValues = selectedActionValues.concat([nextValue]);
      renderActionMetricControls(selectedActionValues);
      syncPreviewToCurrentHover();
    });
    scaleDecrease.addEventListener("click", function () {
      cellSize = clamp(cellSize - CELL_SIZE_STEP, CELL_SIZE_MIN, CELL_SIZE_MAX);
      updateScaleReadout();
      setControlsDisabled(false);
      applyScale();
      renderActionPlot();
    });
    scaleIncrease.addEventListener("click", function () {
      cellSize = clamp(cellSize + CELL_SIZE_STEP, CELL_SIZE_MIN, CELL_SIZE_MAX);
      updateScaleReadout();
      setControlsDisabled(false);
      applyScale();
      renderActionPlot();
    });
    plotStyleLines.addEventListener("click", function () {
      actionPlotStyle = "lines";
      updatePlotStyleButtons();
      renderActionPlot();
    });
    plotStyleBars.addEventListener("click", function () {
      actionPlotStyle = "bars";
      updatePlotStyleButtons();
      renderActionPlot();
    });
    frameSizeDecrease.addEventListener("click", function () {
      framePreviewPercent = clamp(framePreviewPercent - FRAME_PREVIEW_STEP_PERCENT, FRAME_PREVIEW_MIN_PERCENT, FRAME_PREVIEW_MAX_PERCENT);
      applyFramePreviewScale();
    });
    frameSizeIncrease.addEventListener("click", function () {
      framePreviewPercent = clamp(framePreviewPercent + FRAME_PREVIEW_STEP_PERCENT, FRAME_PREVIEW_MIN_PERCENT, FRAME_PREVIEW_MAX_PERCENT);
      applyFramePreviewScale();
    });
    hoverTabButton.addEventListener("click", function () {
      setStatsTab("hover");
    });
    rolloutInfoTabButton.addEventListener("click", function () {
      setStatsTab("rollout-info");
    });
    heatmapCanvas.addEventListener("mousemove", onHeatmapHover);
    heatmapCanvas.addEventListener("mouseleave", hideTooltip);
    actionPlotCanvas.addEventListener("mousemove", onActionPlotHover);
    actionPlotCanvas.addEventListener("mouseleave", hideTooltip);

    populateRolloutOptions();
    clearHeatmap();
    clearActionPlot();
    clearCameraGrid();
    applyFramePreviewScale();
    updateScaleReadout();
    updatePlotStyleButtons();
    setStatsTab(activeStatsTab);
    setControlsDisabled(true);

    if (!MANIFEST.length) {
      hoverStats.textContent = "No rollout episodes were bundled into this viewer.";
      rolloutInfo.textContent = "No rollout episodes were bundled into this viewer.";
      updateAxisNotes();
      return;
    }

    activateRollout(DEFAULT_ROLLOUT_ID || MANIFEST[0].id);
  });
})();
