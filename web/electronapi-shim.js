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
      return fetch("/api/files/read?path=" + encodeURIComponent(filePath))
        .then(function (r) { return r.json(); });
    },
    saveFile: function (filePath, data, encoding) {
      return fetch("/api/files/write", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: filePath, data: data, encoding: encoding || "utf-8" }),
      }).then(function (r) { return r.json(); });
      // NOTE: No auto-download here. Internal saves (project JSON, settings)
      // should not trigger browser downloads. Use triggerDownload() explicitly
      // for user-facing exports.
    },
    saveBinaryFile: function (filePath, data) {
      // Convert ArrayBuffer to base64 using chunked approach for large files
      var bytes = new Uint8Array(data);
      var chunkSize = 32768;
      var parts = [];
      for (var i = 0; i < bytes.length; i += chunkSize) {
        var chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
        parts.push(String.fromCharCode.apply(null, chunk));
      }
      var b64 = btoa(parts.join(""));
      return fetch("/api/files/write", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: filePath, data: b64, encoding: "base64" }),
      }).then(function (r) { return r.json(); });
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

    // Export — v1: not supported, users download raw clips
    exportNative: function () {
      return Promise.resolve({ success: false, error: "Timeline export not available in web UI. Download clips individually." });
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
