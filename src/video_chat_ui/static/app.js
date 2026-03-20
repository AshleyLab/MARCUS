(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const inputMode = document.getElementById("inputMode");
  const dropzoneText = document.getElementById("dropzoneText");
  const dropzoneHint = document.getElementById("dropzoneHint");
  const videoPreview = document.getElementById("videoPreview");
  const previewVideo = document.getElementById("previewVideo");
  const previewImage = document.getElementById("previewImage");
  const videoFilename = document.getElementById("videoFilename");
  const chatLog = document.getElementById("chatLog");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const newChatBtn = document.getElementById("newChatBtn");
  const errorMessage = document.getElementById("errorMessage");
  const loadingMessage = document.getElementById("loadingMessage");

  let state = {
    videoId: null,
    videoFilename: null,
    messages: [],
    mediaKind: "video",
  };

  const MODE_CONFIG = {
    video: {
      accept: ".mp4,.mkv,.avi,.mov",
      text: "Drop video here or click to upload",
      hint: "MP4, MKV, AVI, MOV (max 500MB)",
      placeholder: "Ask a question about the video...",
    },
    cmr: {
      accept: ".tgz,.tar.gz",
      text: "Drop CMR study .tgz here or click",
      hint: "Tar.gz DICOM study (max 500MB); preprocessing may take several minutes.",
      placeholder: "Ask about the CMR grid video...",
    },
    echo: {
      accept: ".tgz,.tar.gz",
      text: "Drop Echo study .tgz here or click",
      hint: "Tar.gz DICOM study (max 500MB); requires ffmpeg on server.",
      placeholder: "Ask about the echo grid video...",
    },
    ecg: {
      accept: ".npy",
      text: "Drop 12-lead ECG .npy here or click",
      hint: "NumPy array shape (12, N); max 500MB.",
      placeholder: "Ask a question about the ECG...",
    },
  };

  function applyModeUI() {
    const m = inputMode.value;
    const c = MODE_CONFIG[m] || MODE_CONFIG.video;
    fileInput.accept = c.accept;
    dropzoneText.textContent = c.text;
    dropzoneHint.textContent = c.hint;
    chatInput.placeholder = c.placeholder;
  }

  inputMode.addEventListener("change", () => {
    applyModeUI();
    newChatBtn.click();
  });
  applyModeUI();

  function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.hidden = false;
  }
  function clearError() {
    errorMessage.hidden = true;
    errorMessage.textContent = "";
  }
  function setLoading(loading, label) {
    loadingMessage.hidden = !loading;
    if (label) loadingMessage.textContent = label;
    else loadingMessage.textContent = "Working...";
    sendBtn.disabled = loading;
  }
  function enableChat() {
    chatInput.disabled = false;
    sendBtn.disabled = false;
  }
  function disableChat() {
    chatInput.disabled = true;
    sendBtn.disabled = true;
  }

  function networkErrorHint(msg) {
    if (!msg || msg === "Failed to fetch" || msg.includes("NetworkError")) {
      return "Connection failed. Open this page at the same address as the server and ensure it is running.";
    }
    return msg;
  }

  function parseError(r, t) {
    try {
      const d = JSON.parse(t);
      return d.detail || (Array.isArray(d.detail) ? d.detail.map((x) => x.msg).join(" ") : r.statusText);
    } catch (_) {
      return t || r.statusText;
    }
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
  });
  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) handleFile(file);
  });

  function handleFile(file) {
    const mode = inputMode.value;
    clearError();
    if (mode === "video") {
      const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
      if (![".mp4", ".mkv", ".avi", ".mov"].includes(ext)) {
        showError("Invalid file type. Use MP4, MKV, AVI, or MOV.");
        return;
      }
      const formData = new FormData();
      formData.append("video", file);
      setLoading(true, "Uploading...");
      fetch("/upload", { method: "POST", body: formData })
        .then((r) => {
          if (!r.ok) return r.text().then((t) => Promise.reject(new Error(parseError(r, t))));
          return r.json();
        })
        .then((data) => {
          state.videoId = data.id;
          state.videoFilename = data.filename;
          state.messages = [];
          state.mediaKind = "video";
          dropzone.hidden = true;
          videoPreview.hidden = false;
          previewVideo.hidden = false;
          previewImage.hidden = true;
          previewVideo.src = URL.createObjectURL(file);
          previewImage.src = "";
          videoFilename.textContent = "Video: " + data.filename;
          chatLog.innerHTML = "";
          enableChat();
        })
        .catch((err) => showError(networkErrorHint(err.message || "Upload failed.")))
        .finally(() => setLoading(false));
      return;
    }

    const fn = file.name.toLowerCase();
    if (mode === "ecg" && !fn.endsWith(".npy")) {
      showError("ECG mode requires a .npy file.");
      return;
    }
    if ((mode === "cmr" || mode === "echo") && !fn.endsWith(".tgz") && !fn.endsWith(".tar.gz")) {
      showError("CMR/Echo mode requires a .tgz archive.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("expert", mode);
    setLoading(true, "Preprocessing (may take minutes)...");
    fetch("/preprocess", { method: "POST", body: formData })
      .then((r) => {
        if (!r.ok) return r.text().then((t) => Promise.reject(new Error(parseError(r, t))));
        return r.json();
      })
      .then((data) => {
        state.videoId = data.id;
        state.videoFilename = file.name;
        state.messages = [];
        state.mediaKind = data.kind === "image" ? "image" : "video";
        dropzone.hidden = true;
        videoPreview.hidden = false;
        if (state.mediaKind === "image") {
          previewVideo.hidden = true;
          previewVideo.src = "";
          previewImage.hidden = false;
          previewImage.src = "/media/" + encodeURIComponent(data.id);
          videoFilename.textContent = "ECG: " + file.name;
        } else {
          previewImage.hidden = true;
          previewImage.src = "";
          previewVideo.hidden = false;
          previewVideo.src = "/media/" + encodeURIComponent(data.id);
          videoFilename.textContent = (mode === "echo" ? "Echo grid: " : "CMR grid: ") + file.name;
        }
        chatLog.innerHTML = "";
        enableChat();
      })
      .catch((err) => showError(networkErrorHint(err.message || "Preprocess failed.")))
      .finally(() => setLoading(false));
  }

  function appendMessage(role, content, isStreaming) {
    const div = document.createElement("div");
    div.className = "message " + role;
    const roleLabel = document.createElement("div");
    roleLabel.className = "role-label";
    roleLabel.textContent = role === "user" ? "You" : "Model";
    const contentEl = document.createElement("div");
    contentEl.className = "content";
    contentEl.textContent = content;
    div.appendChild(roleLabel);
    div.appendChild(contentEl);
    if (role === "user" && state.videoFilename && state.messages.length === 0) {
      const tag = document.createElement("div");
      tag.className = "video-tag";
      tag.textContent =
        state.mediaKind === "image"
          ? "Image: " + state.videoFilename
          : "Video: " + state.videoFilename;
      div.appendChild(tag);
    }
    div.dataset.streaming = isStreaming ? "1" : "0";
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
    return contentEl;
  }

  function updateLastAssistantContent(text) {
    const last = chatLog.querySelector(".message.assistant:last-child");
    if (last) {
      const content = last.querySelector(".content");
      if (content) content.textContent = text;
      chatLog.scrollTop = chatLog.scrollHeight;
    }
  }

  newChatBtn.addEventListener("click", () => {
    state.videoId = null;
    state.videoFilename = null;
    state.messages = [];
    state.mediaKind = "video";
    dropzone.hidden = false;
    videoPreview.hidden = true;
    previewVideo.src = "";
    previewImage.src = "";
    videoFilename.textContent = "";
    fileInput.value = "";
    chatLog.innerHTML = "";
    chatInput.value = "";
    clearError();
    disableChat();
  });

  function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    if (!state.videoId && state.messages.length === 0) {
      showError("Upload or preprocess a file first.");
      return;
    }
    clearError();
    state.messages.push({ role: "user", content: text });
    appendMessage("user", text, false);
    chatInput.value = "";

    const body = {
      video_id: state.videoId,
      media_kind: state.mediaKind,
      messages: state.messages,
    };
    setLoading(true, "Expert model is thinking...");
    appendMessage("assistant", "", true);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000);
    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
      .then((r) => {
        if (!r.ok) {
          return r.text().then((t) => {
            try {
              const d = JSON.parse(t);
              return Promise.reject(new Error(d.detail || r.statusText));
            } catch (_) {
              return Promise.reject(new Error(t || r.statusText));
            }
          });
        }
        return r.body.getReader();
      })
      .then((reader) => {
        const decoder = new TextDecoder();
        let buffer = "";
        let full = "";
        function read() {
          return reader.read().then(({ done, value }) => {
            if (done) {
              clearTimeout(timeoutId);
              setLoading(false);
              const last = chatLog.querySelector(".message.assistant:last-child");
              if (last) last.dataset.streaming = "0";
              state.messages.push({ role: "assistant", content: full });
              return;
            }
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";
            for (const line of lines) {
              if (!line.startsWith("data: ")) continue;
              const data = line.slice(6).trim();
              if (data === "[DONE]") continue;
              try {
                const obj = JSON.parse(data);
                const delta = obj.choices?.[0]?.delta?.content;
                if (delta) {
                  full += delta;
                  updateLastAssistantContent(full);
                }
              } catch (_) {}
            }
            return read();
          });
        }
        return read();
      })
      .catch((err) => {
        clearTimeout(timeoutId);
        setLoading(false);
        const last = chatLog.querySelector(".message.assistant[data-streaming='1']");
        if (last) last.remove();
        const msg = err.name === "AbortError" ? "Request timed out." : err.message || "Request failed.";
        showError(networkErrorHint(msg));
      });
  }

  sendBtn.addEventListener("click", sendMessage);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
})();
