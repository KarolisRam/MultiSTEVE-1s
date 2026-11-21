(function () {
  const ACTION_PLOT_HEIGHT = 150;
  const FRAME_PREVIEW_MIN_PERCENT = 50;
  const FRAME_PREVIEW_MAX_PERCENT = 150;
  const FRAME_PREVIEW_STEP_PERCENT = 10;
  const FRAME_PREVIEW_DEFAULT_PERCENT = 90;

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
    let framePreviewPercent = FRAME_PREVIEW_DEFAULT_PERCENT;
    let lastHoverFrame = 0;
    let lastHoverPosition = null;

    const heatmapCanvas = document.getElementById("heatmapCanvas");
    const heatmapCtx = heatmapCanvas.getContext("2d", { alpha: false });
    const actionPlotCanvas = document.getElementById("actionPlotCanvas");
    const actionPlotCtx = actionPlotCanvas.getContext("2d", { alpha: false });
    const tooltip = document.getElementById("tooltip");
    const scrollShell = document.getElementById("scrollShell");
    const positionAxisNote = document.getElementById("positionAxisNote");
    const actionAxisNote = document.getElementById("actionAxisNote");
    const frameAxisNote = document.getElementById("frameAxisNote");

    const rolloutSelect = document.getElementById("rolloutSelect");
    const layerSelect = document.getElementById("layerSelect");
    const headSelect = document.getElementById("headSelect");
    const modeSelect = document.getElementById("modeSelect");
    const actionMetricSelect = document.getElementById("actionMetricSelect");
    const scaleSelect = document.getElementById("scaleSelect");
    const modeReadout = document.getElementById("modeReadout");
    const hoverStats = document.getElementById("hoverStats");
    const currentFrameImg = document.getElementById("currentFrameImg");
    const currentFrameCaption = document.getElementById("currentFrameCaption");
    const attendedFrameImg = document.getElementById("attendedFrameImg");
    const attendedFrameCaption = document.getElementById("attendedFrameCaption");
    const frameSizeDecrease = document.getElementById("frameSizeDecrease");
    const frameSizeIncrease = document.getElementById("frameSizeIncrease");
    const frameSizeReadout = document.getElementById("frameSizeReadout");

    function activeMeta() {
      return currentBundle ? currentBundle.metadata : null;
    }

    function activeAttentionValues() {
      return currentBundle ? currentBundle.attnValues : null;
    }

    function activeActionSeriesValues() {
      return currentBundle ? currentBundle.actionSeriesValues : null;
    }

    function selectedScale() {
      return Number(scaleSelect.value);
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

    function selectedActionOption() {
      const meta = activeMeta();
      if (!meta || !meta.actionPlotOptions || meta.actionPlotOptions.length === 0) {
        return null;
      }
      const value = actionMetricSelect.value;
      return meta.actionPlotOptions.find((option) => option.value === value) || meta.actionPlotOptions[0];
    }

    function selectedActionLabel() {
      const option = selectedActionOption();
      return option ? option.label : "Action probability";
    }

    function selectedActionSeriesIndex() {
      const option = selectedActionOption();
      return option ? option.index : -1;
    }

    function selectedActionValue(frame) {
      const meta = activeMeta();
      const values = activeActionSeriesValues();
      const seriesIndex = selectedActionSeriesIndex();
      if (!meta || !values || seriesIndex < 0) return 0;
      return values[seriesIndex * meta.frameCount + frame];
    }

    function setControlsDisabled(disabled) {
      rolloutSelect.disabled = false;
      layerSelect.disabled = disabled;
      headSelect.disabled = disabled;
      modeSelect.disabled = disabled;
      actionMetricSelect.disabled = disabled;
      scaleSelect.disabled = false;
    }

    function updateAxisNotes() {
      const meta = activeMeta();
      if (!meta) {
        positionAxisNote.textContent = "Attention position: -";
        actionAxisNote.textContent = "Selected action probability: -";
        frameAxisNote.textContent = "Episode frame: -";
        return;
      }
      positionAxisNote.textContent = "Attention position: 0 to " + (meta.contextPositions - 1);
      actionAxisNote.textContent = "Selected action probability: 0 to 1";
      frameAxisNote.textContent = "Episode frame: 0 to " + (meta.frameCount - 1);
    }

    function clearHeatmap() {
      heatmapCtx.fillStyle = "#ffffff";
      heatmapCtx.fillRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);
    }

    function clearActionPlot() {
      actionPlotCtx.save();
      actionPlotCtx.setTransform(1, 0, 0, 1, 0, 0);
      actionPlotCtx.clearRect(0, 0, actionPlotCanvas.width, actionPlotCanvas.height);
      actionPlotCtx.restore();
      actionPlotCtx.fillStyle = "#fffdf8";
      actionPlotCtx.fillRect(0, 0, actionPlotCssWidth(), ACTION_PLOT_HEIGHT);
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

    function populateActionMetricOptions(preferredValue) {
      const meta = activeMeta();
      actionMetricSelect.innerHTML = "";
      if (!meta || !meta.actionPlotOptions || !meta.actionPlotOptions.length) {
        return;
      }
      let currentGroup = null;
      let currentOptgroup = null;
      for (const optionData of meta.actionPlotOptions) {
        if (optionData.group !== currentGroup) {
          currentGroup = optionData.group;
          currentOptgroup = document.createElement("optgroup");
          currentOptgroup.label = currentGroup;
          actionMetricSelect.appendChild(currentOptgroup);
        }
        const option = document.createElement("option");
        option.value = optionData.value;
        option.textContent = optionData.label;
        currentOptgroup.appendChild(option);
      }
      const fallbackValue = meta.defaultActionValue || (meta.actionPlotOptions[0] && meta.actionPlotOptions[0].value);
      const preferredExists = meta.actionPlotOptions.some((option) => option.value === preferredValue);
      actionMetricSelect.value = preferredExists ? preferredValue : fallbackValue;
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
        modeReadout.innerHTML = "<strong>Status</strong>: Select a rollout to view attention.";
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
      if (mode === "difference") {
        modeReadout.innerHTML =
          "<strong>Selection</strong>: " + selectionLabel() + "<br>" +
          "<strong>Mode</strong>: conditional - unconditional<br>" +
          "<strong>Max abs diff</strong>: " + diffStats.maxAbs.toFixed(6) + "<br>" +
          "<strong>99.5th pct abs diff</strong>: " + diffStats.p995.toFixed(6);
      } else {
        modeReadout.innerHTML =
          "<strong>Selection</strong>: " + selectionLabel() + "<br>" +
          "<strong>Mode</strong>: " + selectedModeLabel() + "<br>" +
          "<strong>Value range</strong>: 0 to 1";
      }
      applyScale();
    }

    function renderActionPlot() {
      const meta = activeMeta();
      const values = activeActionSeriesValues();
      const seriesIndex = selectedActionSeriesIndex();
      resizeActionPlotCanvas();
      if (!meta || !currentBundle || !values || seriesIndex < 0) {
        clearActionPlot();
        return;
      }
      clearActionPlot();

      const scale = selectedScale();
      const plotWidth = actionPlotCssWidth();
      const h = ACTION_PLOT_HEIGHT;
      actionPlotCtx.strokeStyle = "#d9d1c4";
      actionPlotCtx.lineWidth = 1;
      for (const frac of [0, 0.25, 0.5, 0.75, 1]) {
        const y = Math.round((1 - frac) * (h - 1)) + 0.5;
        actionPlotCtx.beginPath();
        actionPlotCtx.moveTo(0, y);
        actionPlotCtx.lineTo(plotWidth, y);
        actionPlotCtx.stroke();
      }

      actionPlotCtx.strokeStyle = "#1f5aa6";
      actionPlotCtx.lineWidth = 1.25;
      actionPlotCtx.beginPath();
      for (let x = 0; x < meta.frameCount; x++) {
        const prob = values[seriesIndex * meta.frameCount + x];
        const y = clamp((1 - prob) * (h - 1), 0, h - 1);
        const plotX = x * scale + (scale / 2);
        if (x === 0) {
          actionPlotCtx.moveTo(plotX, y + 0.5);
        } else {
          actionPlotCtx.lineTo(plotX, y + 0.5);
        }
      }
      actionPlotCtx.stroke();

      const highlightFrame = clamp(lastHoverFrame, 0, meta.frameCount - 1);
      const highlightProb = values[seriesIndex * meta.frameCount + highlightFrame];
      const highlightY = clamp((1 - highlightProb) * (h - 1), 0, h - 1);
      const highlightX = highlightFrame * scale + (scale / 2);

      actionPlotCtx.strokeStyle = "#bb4b31";
      actionPlotCtx.lineWidth = 1;
      actionPlotCtx.beginPath();
      actionPlotCtx.moveTo(highlightX, 0);
      actionPlotCtx.lineTo(highlightX, h);
      actionPlotCtx.stroke();

      actionPlotCtx.fillStyle = "#bb4b31";
      actionPlotCtx.beginPath();
      actionPlotCtx.arc(highlightX, highlightY + 0.5, 3, 0, Math.PI * 2);
      actionPlotCtx.fill();
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
      const actionProb = selectedActionValue(frame);
      const actionLabel = selectedActionLabel();

      currentFrameImg.src = framePath(frame);
      attendedFrameImg.src = framePath(attended);
      currentFrameCaption.textContent = "Frame " + frame;
      attendedFrameCaption.textContent = "Frame " + attended;

      hoverStats.innerHTML =
        "<strong>Selection</strong>: " + selectionLabel() + "<br>" +
        "<strong>View</strong>: " + selectedModeLabel() + "<br>" +
        "<strong>Action plot</strong>: " + actionLabel + "<br>" +
        "<strong>Action probability</strong>: " + actionProb.toFixed(6) + "<br>" +
        "<strong>Current frame</strong>: " + frame + "<br>" +
        "<strong>Attended frame</strong>: " + attended + "<br>" +
        "<strong>Attention position</strong>: " + position + "<br>" +
        "<strong>Conditional value</strong>: " + cond.toFixed(6) + "<br>" +
        "<strong>Unconditional value</strong>: " + uncond.toFixed(6) + "<br>" +
        "<strong>Difference</strong>: " + diff.toFixed(6) + "<br>" +
        "<strong>Displayed value</strong>: " + displayed.toFixed(6);

      return { cond, uncond, diff, displayed, actionProb, actionLabel };
    }

    function updateTooltip(event, frame, attended, position, values) {
      tooltip.style.display = "block";
      tooltip.style.left = String(event.clientX + 16) + "px";
      tooltip.style.top = String(event.clientY + 16) + "px";
      tooltip.innerHTML =
        "frame: " + frame + "<br>" +
        "attended: " + attended + "<br>" +
        "position: " + position + "<br>" +
        "action: " + values.actionLabel + "<br>" +
        "action_prob: " + values.actionProb.toFixed(6) + "<br>" +
        "conditional: " + values.cond.toFixed(6) + "<br>" +
        "unconditional: " + values.uncond.toFixed(6) + "<br>" +
        "difference: " + values.diff.toFixed(6) + "<br>" +
        "displayed: " + values.displayed.toFixed(6);
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
        modeReadout.innerHTML = "<strong>Status</strong>: Unknown rollout: " + rolloutId;
        return;
      }
      const token = ++activationToken;
      const entry = ROLLOUT_MAP[rolloutId];
      const preferredLayer = layerSelect.value || "1";
      const preferredHead = headSelect.value || "14";
      const preferredMode = modeSelect.value || "conditional";
      const preferredActionMetric = actionMetricSelect.value || null;

      setControlsDisabled(true);
      modeReadout.innerHTML = "<strong>Status</strong>: Loading rollout...";
      hideTooltip();

      try {
        const bundle = await ensureRolloutBundle(entry);
        if (token !== activationToken) return;

        purgeInactiveBundles(entry.id);
        ensureAttentionValues(bundle);
        ensureActionSeriesValues(bundle);

        currentRolloutEntry = entry;
        currentBundle = bundle;
        rolloutSelect.value = entry.id;

        const meta = bundle.metadata;
        heatmapCanvas.width = meta.frameCount;
        heatmapCanvas.height = meta.contextPositions;
        lastHoverFrame = 0;
        lastHoverPosition = meta.contextPositions - 1;

        updateAxisNotes();
        populateLayerOptions(preferredLayer);
        populateHeadOptions(preferredHead);
        populateModeOptions(preferredMode);
        populateActionMetricOptions(preferredActionMetric);
        renderHeatmap();
        syncPreviewToCurrentHover();
        applyFramePreviewScale();
        scrollShell.scrollLeft = 0;
        setControlsDisabled(false);
      } catch (error) {
        currentRolloutEntry = null;
        currentBundle = null;
        clearHeatmap();
        clearActionPlot();
        updateAxisNotes();
        hoverStats.textContent = "Failed to load rollout.";
        modeReadout.innerHTML = "<strong>Status</strong>: Failed to load rollout.";
      }
    }

    function handleAttentionSelectionChange() {
      renderHeatmap();
      syncPreviewToCurrentHover();
    }

    function handleActionSelectionChange() {
      syncPreviewToCurrentHover();
    }

    rolloutSelect.addEventListener("change", function () {
      activateRollout(rolloutSelect.value);
    });
    layerSelect.addEventListener("change", handleAttentionSelectionChange);
    headSelect.addEventListener("change", handleAttentionSelectionChange);
    modeSelect.addEventListener("change", handleAttentionSelectionChange);
    actionMetricSelect.addEventListener("change", handleActionSelectionChange);
    scaleSelect.addEventListener("change", function () {
      applyScale();
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
    heatmapCanvas.addEventListener("mousemove", onHeatmapHover);
    heatmapCanvas.addEventListener("mouseleave", hideTooltip);
    actionPlotCanvas.addEventListener("mousemove", onActionPlotHover);
    actionPlotCanvas.addEventListener("mouseleave", hideTooltip);

    populateRolloutOptions();
    clearHeatmap();
    clearActionPlot();
    applyFramePreviewScale();
    setControlsDisabled(true);

    if (!MANIFEST.length) {
      hoverStats.textContent = "No rollout episodes were bundled into this viewer.";
      modeReadout.innerHTML = "<strong>Status</strong>: No rollout data available.";
      updateAxisNotes();
      return;
    }

    activateRollout(DEFAULT_ROLLOUT_ID || MANIFEST[0].id);
  });
})();
