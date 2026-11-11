"use strict";

(function () {
  // 1. Mobile navigation toggle
  const navToggle = document.getElementById("nav-toggle");
  const primaryNav = document.getElementById("primary-nav");
  if (navToggle && primaryNav) {
    navToggle.addEventListener("click", () => {
      const open = primaryNav.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", String(open));
    });
  }

  // 2. Restoration Studio logic
  const form = document.getElementById("restore-form");
  if (form) {
    const dropzone = document.getElementById("dropzone");
    const dropzoneEmpty = document.getElementById("dropzone-empty");
    const previewWrap = document.getElementById("upload-preview-wrap");
    const previewImg = document.getElementById("upload-preview-img");
    const changeBtn = document.getElementById("change-image-btn");
    const removeBtn = document.getElementById("remove-image-btn");
    const fileInput = document.getElementById("digit-file-input");
    const selectedName = document.getElementById("selected-file-name");
    const damageSelect = document.getElementById("damage-type-select");
    const filterDesc = document.getElementById("filter-description");
    const syntheticCheckbox = document.getElementById("synthetic-damage-checkbox");
    const runBtn = document.getElementById("run-restore-btn");
    const loadingBanner = document.getElementById("loading-banner");
    const errorBanner = document.getElementById("error-banner");
    const resultsSection = document.getElementById("results-section");
    const pipelineSummary = document.getElementById("pipeline-summary");
    const downloadBtn = document.getElementById("download-restored-btn");

    const MAX_BYTES = 5 * 1024 * 1024;
    const ALLOWED_EXT = /\.(png|jpe?g)$/i;
    const PREVIEW_TIMEOUT_MS = 6000;
    const DEFAULT_RUN_LABEL = "Run Restoration & Recognition";
    let activeFile = null;
    let activePreviewTimer = null;
    let uploadGeneration = 0;

    function showError(msg) {
      errorBanner.textContent = msg;
      errorBanner.hidden = false;
    }

    function clearError() {
      errorBanner.textContent = "";
      errorBanner.hidden = true;
    }

    function formatFileSize(bytes) {
      if (bytes < 1024) return bytes + " B";
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
      return (bytes / (1024 * 1024)).toFixed(2) + " MB";
    }

    function clearResults() {
      resultsSection.hidden = true;
      pipelineSummary.textContent = "";
      downloadBtn.href = "#";
      ["original", "preprocessed", "restored"].forEach((prefix) => {
        document.getElementById("img-" + prefix).src = "";
        document.getElementById("digit-" + prefix).textContent = "-";
        document.getElementById("conf-" + prefix).textContent = "-";
        document.getElementById("top3-" + prefix).innerHTML = "";
      });
    }

    function clearPreviewTimer() {
      if (activePreviewTimer !== null) {
        clearTimeout(activePreviewTimer);
        activePreviewTimer = null;
      }
    }

    function clearSelectedFile() {
      uploadGeneration += 1;
      clearPreviewTimer();
      activeFile = null;
      fileInput.value = "";
      previewImg.onload = null;
      previewImg.onerror = null;
      previewImg.removeAttribute("src");
      previewImg.hidden = true;
      previewWrap.hidden = true;
      dropzoneEmpty.hidden = false;
      selectedName.textContent = "No file selected.";
      runBtn.disabled = true;
      clearResults();
    }

    function validateClientFile(file) {
      if (!file) {
        return "Please choose a PNG, JPG, or JPEG image.";
      }
      if (!ALLOWED_EXT.test(file.name)) {
        return "Unsupported file type. Upload a PNG, JPG, or JPEG image.";
      }
      if (file.type && !["image/png", "image/jpeg", "image/pjpeg"].includes(file.type.toLowerCase())) {
        return "Unsupported MIME type (" + file.type + "). Expected PNG or JPEG.";
      }
      if (file.size === 0) {
        return "Selected file is empty.";
      }
      if (file.size > MAX_BYTES) {
        return "File exceeds the 5 MB upload limit.";
      }
      return null;
    }

    function setFile(file) {
      clearError();
      clearResults();
      clearPreviewTimer();
      const err = validateClientFile(file);
      if (err) {
        clearSelectedFile();
        showError(err);
        return;
      }

      const gen = ++uploadGeneration;
      activeFile = null;
      runBtn.disabled = true;

      const sizeText = formatFileSize(file.size);
      dropzoneEmpty.hidden = true;
      previewWrap.hidden = false;
      previewImg.hidden = true;
      previewImg.removeAttribute("src");
      selectedName.textContent = file.name + " · Reading dimensions... · " + sizeText;

      let settled = false;
      function failPreview(message) {
        if (settled || gen !== uploadGeneration) return;
        settled = true;
        clearPreviewTimer();
        clearSelectedFile();
        showError(message);
      }

      function finishPreview(dataUrl, width, height) {
        if (settled || gen !== uploadGeneration) return;
        if (!width || !height) {
          failPreview("Selected image reported invalid dimensions (0 px).");
          return;
        }
        settled = true;
        clearPreviewTimer();
        activeFile = file;
        previewImg.classList.toggle("pixel-art", width <= 64 && height <= 64);
        previewImg.src = dataUrl;
        previewImg.hidden = false;
        selectedName.textContent = file.name + " · " + width + " x " + height + " px · " + sizeText;
        runBtn.disabled = false;
      }

      activePreviewTimer = setTimeout(() => {
        failPreview("Timed out while reading image dimensions. Please try another PNG or JPEG file.");
      }, PREVIEW_TIMEOUT_MS);

      const reader = new FileReader();
      reader.onerror = () => {
        failPreview("Could not read the selected file from disk.");
      };
      reader.onload = () => {
        if (settled || gen !== uploadGeneration) return;
        const dataUrl = typeof reader.result === "string" ? reader.result : "";
        if (!dataUrl) {
          failPreview("Selected file produced an empty image preview.");
          return;
        }

        const decoder = new Image();
        decoder.onload = () => {
          finishPreview(dataUrl, decoder.naturalWidth, decoder.naturalHeight);
        };
        decoder.onerror = () => {
          failPreview("Selected file could not be decoded as a valid PNG or JPEG image.");
        };
        decoder.src = dataUrl;
        if (decoder.complete && decoder.naturalWidth > 0) {
          finishPreview(dataUrl, decoder.naturalWidth, decoder.naturalHeight);
        }
      };
      reader.readAsDataURL(file);
    }

    if (damageSelect && filterDesc) {
      damageSelect.addEventListener("change", () => {
        const opt = damageSelect.options[damageSelect.selectedIndex];
        const filterText = opt ? opt.getAttribute("data-filter") : "";
        filterDesc.textContent =
          "Classical branch: " + filterText + ". The denoising autoencoder never sees this selection.";
      });
    }

    if (changeBtn) {
      changeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
      });
    }

    if (removeBtn) {
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearError();
        clearSelectedFile();
      });
    }

    dropzone.addEventListener("click", (e) => {
      if (
        e.target.tagName !== "LABEL" &&
        e.target.tagName !== "BUTTON" &&
        e.target !== fileInput
      ) {
        fileInput.click();
      }
    });

    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("is-dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("is-dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-dragover");
      const files = e.dataTransfer ? e.dataTransfer.files : null;
      if (files && files.length > 0) {
        setFile(files[0]);
      }
    });

    fileInput.addEventListener("click", () => {
      fileInput.value = "";
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length > 0) {
        setFile(fileInput.files[0]);
      }
    });

    function renderStageCard(prefix, data, imageSrcOverride) {
      document.getElementById("img-" + prefix).src = imageSrcOverride || data.image;
      document.getElementById("digit-" + prefix).textContent = String(data.digit);
      document.getElementById("conf-" + prefix).textContent = data.confidence.toFixed(2);

      const listEl = document.getElementById("top3-" + prefix);
      listEl.innerHTML = "";
      data.top3.forEach((entry) => {
        const li = document.createElement("li");
        li.className = "top3-item";

        const label = document.createElement("span");
        label.textContent = "Digit " + entry.digit;

        const track = document.createElement("div");
        track.className = "top3-bar-track";
        const fill = document.createElement("div");
        fill.className = "top3-bar-fill";
        fill.style.width = Math.max(0, Math.min(100, entry.probability)) + "%";
        track.appendChild(fill);

        const prob = document.createElement("span");
        prob.className = "top3-prob";
        prob.textContent = entry.probability.toFixed(1) + "%";

        li.appendChild(label);
        li.appendChild(track);
        li.appendChild(prob);
        listEl.appendChild(li);
      });
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      clearError();

      const err = validateClientFile(activeFile);
      if (err) {
        showError(err);
        return;
      }

      const formData = new FormData();
      formData.append("image", activeFile);
      formData.append("damage_type", damageSelect.value);
      formData.append("add_synthetic_damage", syntheticCheckbox.checked ? "true" : "false");

      runBtn.disabled = true;
      runBtn.textContent = "Running...";
      loadingBanner.hidden = false;

      try {
        const resp = await fetch("/api/restore", {
          method: "POST",
          body: formData,
        });
        const payload = await resp.json().catch(() => ({ error: "Unexpected server response." }));
        if (!resp.ok) {
          showError(payload.error || "Failed to process image.");
          return;
        }

        renderStageCard("original", payload.original);
        renderStageCard("preprocessed", payload.preprocessed);
        // Display the exact image served by the restored-image download endpoint
        renderStageCard("restored", payload.restored, payload.download_url);

        const parts = ["Selected damage: " + payload.damage_label, "Classical filter: " + payload.classical_filter];
        if (payload.inverted) {
          parts.push("Auto-inverted dark-on-light input");
        }
        if (payload.synthetic_damage_applied) {
          parts.push("Synthetic corruption applied: " + payload.synthetic_damage_applied);
        }
        pipelineSummary.textContent = parts.join(" · ");

        downloadBtn.href = payload.download_url;
        resultsSection.hidden = false;
      } catch (_err) {
        showError("Network error while communicating with the DigitRescue server.");
      } finally {
        loadingBanner.hidden = true;
        runBtn.textContent = DEFAULT_RUN_LABEL;
        runBtn.disabled = activeFile === null;
      }
    });
  }

  // 3. Performance page charts (local Chart.js, accent + neutral greys only)
  const perfJsonEl = document.getElementById("perf-json");
  if (perfJsonEl && typeof Chart !== "undefined") {
    const perf = JSON.parse(perfJsonEl.textContent || "{}");
    Chart.defaults.font.family = '"Inter", sans-serif';
    Chart.defaults.color = "#5B606A";

    const accCanvas = document.getElementById("accuracy-chart");
    if (accCanvas && Array.isArray(perf.accuracy_rows) && perf.accuracy_rows.length > 0) {
      new Chart(accCanvas, {
        type: "bar",
        data: {
          labels: perf.accuracy_rows.map((r) => r.label),
          datasets: [
            {
              label: "Noisy (%)",
              data: perf.accuracy_rows.map((r) => (typeof r.noisy === "number" ? r.noisy : null)),
              backgroundColor: "#D1D5DB",
              borderRadius: 4,
            },
            {
              label: "Preprocessed (%)",
              data: perf.accuracy_rows.map((r) => (typeof r.preprocessed === "number" ? r.preprocessed : null)),
              backgroundColor: "#5B606A",
              borderRadius: 4,
            },
            {
              label: "AI Restored (%)",
              data: perf.accuracy_rows.map((r) => (typeof r.restored === "number" ? r.restored : null)),
              backgroundColor: "#0F766E",
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 150 },
          scales: {
            y: {
              min: 75,
              max: 100,
              grid: { color: "#E4E0D9" },
              title: { display: true, text: "Accuracy (%)" },
            },
            x: { grid: { display: false } },
          },
          plugins: {
            legend: { position: "top" },
          },
        },
      });
    }

    const psnrCanvas = document.getElementById("psnr-chart");
    if (psnrCanvas && Array.isArray(perf.psnr_rows) && perf.psnr_rows.length > 0) {
      new Chart(psnrCanvas, {
        type: "bar",
        data: {
          labels: perf.psnr_rows.map((r) => r.label),
          datasets: [
            {
              label: "Noisy PSNR (dB)",
              data: perf.psnr_rows.map((r) => (typeof r.noisy_psnr === "number" ? r.noisy_psnr : null)),
              backgroundColor: "#9CA3AF",
              borderRadius: 4,
            },
            {
              label: "Restored PSNR (dB)",
              data: perf.psnr_rows.map((r) => (typeof r.restored_psnr === "number" ? r.restored_psnr : null)),
              backgroundColor: "#0F766E",
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 150 },
          scales: {
            y: {
              beginAtZero: true,
              grid: { color: "#E4E0D9" },
              title: { display: true, text: "PSNR (dB)" },
            },
            x: { grid: { display: false } },
          },
          plugins: {
            legend: { position: "top" },
          },
        },
      });
    }

    const lossCanvas = document.getElementById("loss-chart");
    if (lossCanvas && Array.isArray(perf.ae_history) && perf.ae_history.length > 0) {
      new Chart(lossCanvas, {
        type: "line",
        data: {
          labels: perf.ae_history.map((r) => "Ep " + r.epoch),
          datasets: [
            {
              label: "Train Loss",
              data: perf.ae_history.map((r) => (typeof r.loss === "number" ? r.loss : null)),
              borderColor: "#5B606A",
              backgroundColor: "#5B606A",
              borderWidth: 2,
              pointRadius: 2,
              tension: 0.15,
            },
            {
              label: "Validation Loss",
              data: perf.ae_history.map((r) => (typeof r.val_loss === "number" ? r.val_loss : null)),
              borderColor: "#0F766E",
              backgroundColor: "#0F766E",
              borderWidth: 2,
              pointRadius: 2,
              tension: 0.15,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 150 },
          scales: {
            y: {
              grid: { color: "#E4E0D9" },
              title: { display: true, text: "Binary Cross-Entropy Loss" },
            },
            x: { grid: { display: false } },
          },
          plugins: {
            legend: { position: "top" },
          },
        },
      });
    }
  }
})();
