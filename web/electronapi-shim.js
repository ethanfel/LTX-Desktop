/**
 * electronAPI shim for running LTX Desktop as a web app.
 * Injected into index.html at runtime by nginx sub_filter.
 * Replaces Electron IPC with browser-native equivalents.
 */
(function () {
  "use strict";

  // FILE_ROOT matches the Docker volume mount. All server-side paths are under this.
  var FILE_ROOT = "/data";

  // =========================================================================
  // file:// URL interceptor
  //
  // The frontend converts server paths to file:// URLs in ~25 locations for
  // <video>, <img>, and <audio> src attributes. Browsers block file:// from
  // web pages. We patch the DOM property setters to rewrite them to HTTP
  // serve URLs transparently.
  // =========================================================================

  function rewriteFileUrl(url) {
    if (typeof url !== "string" || !url.startsWith("file://")) return url;
    // If we have a cached blob URL for this file:// URL, prefer it (instant display
    // while the background upload completes).
    if (_blobUrlMap && _blobUrlMap[url]) return _blobUrlMap[url];
    // file:///data/foo/bar.mp4 → /data/foo/bar.mp4
    var fsPath = decodeURIComponent(url.slice(7));
    // Strip leading slash duplication (file:/// → /)
    if (/^\/[A-Za-z]:/.test(fsPath)) fsPath = fsPath.slice(1); // Windows drive
    // Only rewrite paths under FILE_ROOT
    if (fsPath.startsWith(FILE_ROOT + "/")) {
      var relative = fsPath.substring(FILE_ROOT.length + 1);
      // Encode each path segment individually (preserve /)
      var encoded = relative.split("/").map(encodeURIComponent).join("/");
      return "/api/files/serve/" + encoded;
    }
    return url;
  }

  // Patch src property on media elements to intercept file:// URLs
  function patchSrcProperty(prototype) {
    var descriptor = Object.getOwnPropertyDescriptor(prototype, "src");
    if (!descriptor || !descriptor.set) return;
    var originalSet = descriptor.set;
    var originalGet = descriptor.get;
    Object.defineProperty(prototype, "src", {
      get: originalGet,
      set: function (value) {
        originalSet.call(this, rewriteFileUrl(value));
      },
      enumerable: descriptor.enumerable,
      configurable: descriptor.configurable,
    });
  }

  // Patch all relevant element prototypes
  if (typeof HTMLMediaElement !== "undefined") patchSrcProperty(HTMLMediaElement.prototype);
  if (typeof HTMLImageElement !== "undefined") patchSrcProperty(HTMLImageElement.prototype);
  if (typeof HTMLSourceElement !== "undefined") patchSrcProperty(HTMLSourceElement.prototype);

  // Also patch setAttribute for src (React sometimes uses this)
  var origSetAttribute = Element.prototype.setAttribute;
  Element.prototype.setAttribute = function (name, value) {
    if (name === "src" || name === "poster") {
      value = rewriteFileUrl(value);
    }
    return origSetAttribute.call(this, name, value);
  };

  // =========================================================================
  // URL.createObjectURL interceptor
  //
  // ImageUploader.tsx uses (file as any).path (Electron-only). In browsers,
  // File.path is undefined so it falls back to URL.createObjectURL(file),
  // producing a blob: URL. fileUrlToPath() can't convert blob: URLs, so i2v
  // silently degrades to t2v. We intercept createObjectURL for File objects:
  // upload the file in the background and return a file:// URL immediately.
  // The DOM interceptor rewrites file:// to /api/files/serve/ for display.
  // =========================================================================

  var _origCreateObjectURL = URL.createObjectURL.bind(URL);
  var _origRevokeObjectURL = URL.revokeObjectURL.bind(URL);

  // Map from file:// URL to real blob URL (for revocation)
  var _blobUrlMap = {};

  URL.createObjectURL = function (obj) {
    if (obj instanceof File && obj.size > 0) {
      var safeName = (obj.name || "file").replace(/[/\\]/g, "_");
      var uniqueName = Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8) + "_" + safeName;
      var serverPath = FILE_ROOT + "/uploads/" + uniqueName;
      var fileUrl = "file://" + serverPath;

      // Upload in background — by the time user clicks Generate, upload is done
      var formData = new FormData();
      formData.append("file", obj);
      formData.append("path", serverPath);
      fetch("/api/files/upload-binary", { method: "POST", body: formData })
        .then(function (r) { return r.json(); })
        .then(function (result) {
          if (!result.success) console.error("[LTX Web] Background file upload failed:", result.error);
        })
        .catch(function (e) { console.error("[LTX Web] Background file upload error:", e); });

      // Also create a real blob URL so the image preview works immediately
      // (before upload completes). Store mapping for cleanup.
      var realBlobUrl = _origCreateObjectURL(obj);
      _blobUrlMap[fileUrl] = realBlobUrl;

      return fileUrl;
    }
    return _origCreateObjectURL(obj);
  };

  URL.revokeObjectURL = function (url) {
    if (_blobUrlMap[url]) {
      _origRevokeObjectURL(_blobUrlMap[url]);
      delete _blobUrlMap[url];
    } else {
      _origRevokeObjectURL(url);
    }
  };

  // =========================================================================
  // Helpers
  // =========================================================================

  /** Convert a server filesystem path to a serve URL */
  function pathToServeUrl(serverPath) {
    if (serverPath && serverPath.startsWith(FILE_ROOT + "/")) {
      var relative = serverPath.substring(FILE_ROOT.length + 1);
      return "/api/files/serve/" + relative.split("/").map(encodeURIComponent).join("/");
    }
    return serverPath;
  }

  /** Trigger a hidden file input and return selected files uploaded to server. */
  function openFilePicker(accept, multiple) {
    return new Promise(function (resolve) {
      var input = document.createElement("input");
      input.type = "file";
      if (accept) input.accept = accept;
      if (multiple) input.multiple = true;
      input.style.display = "none";
      document.body.appendChild(input);
      var resolved = false;
      input.addEventListener("change", async function () {
        resolved = true;
        var files = Array.from(input.files || []);
        if (files.length === 0) { cleanup(); resolve(null); return; }
        var paths = [];
        for (var i = 0; i < files.length; i++) {
          var form = new FormData();
          form.append("file", files[i]);
          try {
            var resp = await fetch("/api/files/upload", { method: "POST", body: form });
            var json = await resp.json();
            paths.push(json.path);
          } catch (e) {
            console.error("[LTX Web] Upload failed:", e);
          }
        }
        cleanup();
        resolve(paths.length > 0 ? paths : null);
      });
      function cleanup() { try { document.body.removeChild(input); } catch(e) {} }
      // Detect cancel via focus return
      window.addEventListener("focus", function onFocus() {
        window.removeEventListener("focus", onFocus);
        setTimeout(function () { if (!resolved) { cleanup(); resolve(null); } }, 500);
      });
      input.click();
    });
  }

  /** Build accept string from Electron-style filters. */
  function filtersToAccept(filters) {
    if (!filters || filters.length === 0) return "";
    var exts = [];
    filters.forEach(function (f) {
      (f.extensions || []).forEach(function (ext) { exts.push("." + ext); });
    });
    return exts.join(",");
  }

  /** Trigger browser download for a server-side file. */
  function triggerDownload(serverPath) {
    var url = pathToServeUrl(serverPath);
    if (!url) return;
    var a = document.createElement("a");
    a.href = url;
    a.download = serverPath.split("/").pop() || "download";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  // =========================================================================
  // Health polling
  // =========================================================================

  var healthCallbacks = [];
  var lastHealthStatus = { status: "alive" };

  function pollHealth() {
    fetch("/health")
      .then(function (r) {
        lastHealthStatus = r.ok ? { status: "alive" } : { status: "dead" };
      })
      .catch(function () {
        lastHealthStatus = { status: "dead" };
      })
      .finally(function () {
        healthCallbacks.forEach(function (cb) { try { cb(lastHealthStatus); } catch(e) {} });
      });
  }

  setInterval(pollHealth, 5000);

  // =========================================================================
  // The shim
  // =========================================================================

  window.electronAPI = {
    // Platform
    platform: "linux",

    // Backend connectivity — same origin, no auth
    getBackend: function () {
      return Promise.resolve({ url: "", token: "" });
    },
    getBackendHealthStatus: function () {
      return Promise.resolve(lastHealthStatus);
    },
    onBackendHealthStatus: function (cb) {
      healthCallbacks.push(cb);
      return function () {
        healthCallbacks = healthCallbacks.filter(function (c) { return c !== cb; });
      };
    },

    // Setup/bootstrap — bypass everything (backend already running)
    checkPythonReady: function () { return Promise.resolve({ ready: true }); },
    startPythonBackend: function () { return Promise.resolve(); },
    startPythonSetup: function () { return Promise.resolve(); },
    checkFirstRun: function () { return Promise.resolve({ needsSetup: false, needsLicense: false }); },
    acceptLicense: function () { return Promise.resolve(true); },
    completeSetup: function () { return Promise.resolve(true); },
    fetchLicenseText: function () { return Promise.resolve("Apache-2.0 License"); },
    getNoticesText: function () { return Promise.resolve(""); },
    onPythonSetupProgress: function () {},
    removePythonSetupProgress: function () {},

    // App info
    getAppInfo: function () {
      return Promise.resolve({
        version: "web",
        isPackaged: true,
        modelsPath: FILE_ROOT + "/LTXDesktop/models",
        userDataPath: FILE_ROOT + "/LTXDesktop",
      });
    },
    getModelsPath: function () { return Promise.resolve(FILE_ROOT + "/LTXDesktop/models"); },
    getDownloadsPath: function () { return Promise.resolve(FILE_ROOT + "/downloads"); },
    getResourcePath: function () { return Promise.resolve(null); },
    getProjectAssetsPath: function () { return Promise.resolve(FILE_ROOT + "/project-assets"); },
    getLogPath: function () {
      return Promise.resolve({ logPath: FILE_ROOT + "/logs/ltx.log", logDir: FILE_ROOT + "/logs" });
    },
    checkGpu: function () {
      return fetch("/api/runtime-policy")
        .then(function (r) { return r.json(); })
        .then(function (data) {
          return { available: !data.force_api_generations, name: "NVIDIA GPU", vram: 0 };
        })
        .catch(function () { return { available: true }; });
    },

    // File dialogs → HTML5 file picker + upload to server
    showOpenFileDialog: function (options) {
      var accept = filtersToAccept(options && options.filters);
      var multi = options && options.properties && options.properties.indexOf("multiSelections") >= 0;
      return openFilePicker(accept, multi);
    },
    showSaveDialog: function (options) {
      // Generate a server-side path; user downloads the result after it's written
      var name = (options && options.defaultPath) || "export_" + Date.now();
      var filename = name.split("/").pop() || name;
      return Promise.resolve(FILE_ROOT + "/downloads/" + filename);
    },
    showOpenDirectoryDialog: function () {
      // No directory picker in browsers — return default data dir
      return Promise.resolve(FILE_ROOT + "/LTXDesktop");
    },

    // File operations → backend endpoints
    readLocalFile: function (filePath) {
      // Strip file:// prefix — frontend often passes file:// URLs (e.g. AudioWaveform)
      if (filePath.startsWith("file://")) filePath = decodeURIComponent(filePath.slice(7));
      return fetch("/api/files/read?path=" + encodeURIComponent(filePath))
        .then(function (r) { return r.json(); });
    },
    saveFile: function (filePath, data, encoding) {
      return fetch("/api/files/write", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: filePath, data: data, encoding: encoding || "utf-8" }),
      }).then(function (r) { return r.json(); }).then(function (result) {
        // Auto-download files saved to /downloads/ (user-facing exports like subtitles)
        if (result.success && filePath.startsWith(FILE_ROOT + "/downloads/")) {
          triggerDownload(filePath);
        }
        return result;
      });
    },
    saveBinaryFile: function (filePath, data) {
      // Upload binary data as multipart form (avoids base64 bloat and stack overflow)
      var blob = new Blob([data]);
      var form = new FormData();
      form.append("file", blob, filePath.split("/").pop() || "binary");
      form.append("path", filePath);
      return fetch("/api/files/upload-binary", { method: "POST", body: form })
        .then(function (r) { return r.json(); })
        .then(function (result) {
          if (result.success && filePath.startsWith(FILE_ROOT + "/downloads/")) {
            triggerDownload(filePath);
          }
          return result;
        });
    },
    copyToProjectAssets: function (srcPath, projectId) {
      return fetch("/api/files/copy-to-assets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ srcPath: srcPath, projectId: projectId }),
      }).then(function (r) { return r.json(); });
    },
    checkFilesExist: function (filePaths) {
      return fetch("/api/files/check-exist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths: filePaths }),
      }).then(function (r) { return r.json(); });
    },
    searchDirectoryForFiles: function (dir, filenames) {
      var paths = filenames.map(function (f) { return dir + "/" + f; });
      return fetch("/api/files/check-exist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths: paths }),
      })
        .then(function (r) { return r.json(); })
        .then(function (exists) {
          var result = {};
          filenames.forEach(function (f) {
            var full = dir + "/" + f;
            if (exists[full]) result[f] = full;
          });
          return result;
        });
    },

    // Video frame extraction
    extractVideoFrame: function (videoUrl, seekTime, width, quality) {
      // Convert file:// URL or serve URL back to a server filesystem path
      var videoPath = videoUrl;
      if (videoUrl.startsWith("file://")) {
        videoPath = decodeURIComponent(videoUrl.slice(7));
      } else if (videoUrl.startsWith("/api/files/serve/")) {
        videoPath = FILE_ROOT + "/" + decodeURIComponent(videoUrl.replace("/api/files/serve/", ""));
      }
      return fetch("/api/files/extract-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ videoPath: videoPath, seekTime: seekTime, width: width, quality: quality }),
      }).then(function (r) { return r.json(); });
    },

    // Export — server-side ffmpeg via backend endpoint
    exportNative: function (data) {
      return fetch("/api/files/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      })
        .then(function (r) { return r.json(); })
        .then(function (result) {
          if (result.success && result.download && data.outputPath) {
            triggerDownload(data.outputPath);
          }
          return result;
        })
        .catch(function (e) {
          return { success: false, error: "Export failed: " + e.message };
        });
    },
    exportCancel: function () { return Promise.resolve({ ok: true }); },

    // Directory change dialogs — not available in browser
    openModelsDirChangeDialog: function () {
      return Promise.resolve({ success: false, error: "Not available in web UI" });
    },
    openProjectAssetsPathChangeDialog: function () {
      return Promise.resolve({ success: false, error: "Not available in web UI" });
    },

    // Open external links
    openLtxApiKeyPage: function () { window.open("https://ltx.video", "_blank"); return Promise.resolve(true); },
    openFalApiKeyPage: function () { window.open("https://fal.ai", "_blank"); return Promise.resolve(true); },

    // Navigation — no-op in browser
    showItemInFolder: function () { return Promise.resolve(); },
    openParentFolderOfFile: function () { return Promise.resolve(); },
    openLogFolder: function () { return Promise.resolve(true); },

    // Logging
    writeLog: function (level, message) {
      console[level === "ERROR" ? "error" : "log"]("[LTX]", message);
      return Promise.resolve();
    },
    getLogs: function () {
      return Promise.resolve({ logPath: "", lines: ["Logs not available in web UI"], error: undefined });
    },

    // Analytics — no-op
    getAnalyticsState: function () { return Promise.resolve({ analyticsEnabled: false, installationId: "web" }); },
    setAnalyticsEnabled: function () { return Promise.resolve(); },
    sendAnalyticsEvent: function () { return Promise.resolve(); },
  };

  // Expose triggerDownload for explicit user exports
  window.__ltxWebDownload = triggerDownload;

  console.log("[LTX Web] electronAPI shim loaded — file:// URL interception active");
})();
