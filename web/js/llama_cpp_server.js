/**
 * Frontend for the "Llama-cpp Server" node.
 *
 * The node itself only shows the few things that change often (preset, model,
 * mmproj, thinking switch) plus a status line and two buttons: Start/Stop and
 * Settings. Everything else stays a real widget (so workflows keep serialising
 * it) but is hidden from the canvas and edited through a modal settings panel.
 * All actions go through the HTTP routes in nodes_server.py (/llama_cpp_vlm/server/...).
 */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "llama_cpp_server";
const ROUTE = "/llama_cpp_vlm/server";
const CUSTOM_PRESET = "Custom";
const POLL_MS = 4000;

// Widgets that stay on the node. Everything else moves into the settings panel.
const VISIBLE = new Set(["preset", "model", "mmproj", "enable_thinking"]);

const SECTIONS = [
    { title: "Connection", keys: ["mode", "base_url", "server_exe", "api_key"] },
    { title: "Launch (mode = launch)", keys: ["cuda_devices", "n_gpu_layers", "n_ctx", "image_max_tokens", "extra_args", "startup_timeout"] },
    { title: "Generation", keys: ["reasoning_effort", "max_tokens", "temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty", "timeout"] },
    { title: "Cache", keys: ["cache_size"] },
];

let presetCache = null;
let cssInjected = false;

function toast(severity, summary, detail = "", life = 4000) {
    try {
        app.extensionManager.toast.add({ severity, summary, detail, life });
    } catch (e) {
        console.log(`[llama-cpp_vlm] ${severity}: ${summary} ${detail}`);
    }
}

async function apiJson(path, options = {}) {
    const response = await api.fetchApi(`${ROUTE}${path}`, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });
    let data = {};
    try { data = await response.json(); } catch (e) { /* ignore */ }
    if (!response.ok || data.ok === false) {
        throw new Error(data.error || `${response.status} ${response.statusText}`);
    }
    return data;
}

async function loadPresets(force = false) {
    if (presetCache && !force) return presetCache;
    const data = await apiJson("/presets");
    presetCache = { builtin: data.builtin || {}, user: data.user || {}, keys: data.keys || [] };
    return presetCache;
}

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function collectValues(node) {
    const values = {};
    for (const w of node.widgets || []) {
        if (w.__llamaExtra) continue;
        values[w.name] = w.value;
    }
    return values;
}

function pickOption(options, match) {
    if (!match || !Array.isArray(options)) return null;
    const needle = String(match).toLowerCase();
    return options.find((o) => String(o).toLowerCase() === needle)
        || options.find((o) => String(o).toLowerCase().includes(needle))
        || null;
}

function promptText(title, defaultValue, callback) {
    if (app.canvas && typeof app.canvas.prompt === "function") {
        app.canvas.prompt(title, defaultValue, callback, null);
    } else {
        const value = window.prompt(title, defaultValue);
        if (value !== null) callback(value);
    }
}

function hideWidget(w) {
    w.hidden = true;                       // litegraph: not drawn, takes no space
    w.options = { ...(w.options || {}), hidden: true }; // side panel
}

function injectCss() {
    if (cssInjected) return;
    cssInjected = true;
    const style = document.createElement("style");
    style.textContent = `
.llama-srv-overlay{position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:10000;display:flex;align-items:center;justify-content:center}
.llama-srv-dialog{background:var(--comfy-menu-bg,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);border-radius:10px;width:560px;max-width:92vw;max-height:88vh;display:flex;flex-direction:column;box-shadow:0 10px 40px rgba(0,0,0,.45);font-size:13px}
.llama-srv-head{display:flex;align-items:center;justify-content:space-between;padding:14px 18px;border-bottom:1px solid var(--border-color,#444)}
.llama-srv-head h3{margin:0;font-size:15px;font-weight:600}
.llama-srv-body{overflow:auto;padding:6px 18px 14px}
.llama-srv-section{margin-top:12px}
.llama-srv-section h4{margin:0 0 6px;font-size:12px;letter-spacing:.04em;text-transform:uppercase;opacity:.65}
.llama-srv-row{display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center;padding:4px 0}
.llama-srv-row label{opacity:.9;cursor:help}
.llama-srv-row input[type=text],.llama-srv-row input[type=number],.llama-srv-row select{width:100%;box-sizing:border-box;background:var(--comfy-input-bg,#1a1a1a);color:var(--input-text,#ddd);border:1px solid var(--border-color,#444);border-radius:6px;padding:5px 8px;font:inherit}
.llama-srv-row input[type=checkbox]{width:16px;height:16px}
.llama-srv-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
.llama-srv-btn{background:var(--comfy-input-bg,#1a1a1a);color:var(--input-text,#ddd);border:1px solid var(--border-color,#444);border-radius:6px;padding:6px 12px;cursor:pointer;font:inherit}
.llama-srv-btn:hover{border-color:var(--p-primary-color,#6fa8dc)}
.llama-srv-btn.danger:hover{border-color:#d9534f;color:#f28b85}
.llama-srv-foot{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:12px 18px;border-top:1px solid var(--border-color,#444)}
.llama-srv-foot .hint{opacity:.6;font-size:12px}
`;
    document.head.appendChild(style);
}

function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
        else node.setAttribute(k, v);
    }
    for (const c of children) node.appendChild(c);
    return node;
}

/** Build one form control bound to a node widget. */
function controlFor(node, w, spec, onChanged) {
    const opts = w.options || {};
    const type = spec ? spec[0] : w.type;
    let input;
    const commit = (value) => {
        w.value = value;
        w.callback?.(value, app.canvas, node);
        onChanged?.(w);
        node.setDirtyCanvas(true, true);
    };
    if (Array.isArray(type) || Array.isArray(opts.values)) {
        input = el("select", { onchange: () => commit(input.value) });
        for (const v of (Array.isArray(type) ? type : opts.values)) {
            input.appendChild(el("option", { value: v, text: v }));
        }
        input.value = w.value;
    } else if (type === "BOOLEAN") {
        input = el("input", { type: "checkbox", onchange: () => commit(input.checked) });
        input.checked = !!w.value;
    } else if (type === "INT" || type === "FLOAT") {
        input = el("input", { type: "number", onchange: () => {
            let v = type === "INT" ? parseInt(input.value, 10) : parseFloat(input.value);
            if (Number.isNaN(v)) v = w.value;
            if (opts.min !== undefined) v = Math.max(opts.min, v);
            if (opts.max !== undefined) v = Math.min(opts.max, v);
            input.value = v;
            commit(v);
        } });
        if (opts.min !== undefined) input.min = opts.min;
        if (opts.max !== undefined) input.max = opts.max;
        input.step = type === "INT" ? (opts.step || 1) : (opts.round || opts.step || 0.01);
        input.value = w.value;
    } else {
        input = el("input", { type: "text", onchange: () => commit(input.value) });
        input.value = w.value ?? "";
    }
    return input;
}

function openSettings(node, nodeData) {
    injectCss();
    const required = nodeData.input?.required || {};
    const presetWidget = widget(node, "preset");
    const markCustom = (w) => {
        if (w.name !== "preset" && presetWidget && presetWidget.value !== CUSTOM_PRESET) {
            presetWidget.value = CUSTOM_PRESET;
        }
    };

    const body = el("div", { class: "llama-srv-body" });
    for (const section of SECTIONS) {
        const sec = el("div", { class: "llama-srv-section" }, [el("h4", { text: section.title })]);
        for (const key of section.keys) {
            const w = widget(node, key);
            if (!w) continue;
            const spec = required[key];
            const tooltip = spec?.[1]?.tooltip || "";
            const label = el("label", { text: key, title: tooltip });
            sec.appendChild(el("div", { class: "llama-srv-row" }, [label, controlFor(node, w, spec, markCustom)]));
        }
        body.appendChild(sec);
    }

    // ---- maintenance actions ------------------------------------------------
    const actions = el("div", { class: "llama-srv-actions" });
    actions.appendChild(el("button", { class: "llama-srv-btn", text: "💾 Save as preset…", onclick: () => {
        const current = presetWidget?.value;
        promptText("Preset name", current && current !== CUSTOM_PRESET ? current : "", async (name) => {
            name = (name || "").trim();
            if (!name) return;
            try {
                const result = await apiJson("/presets", { method: "POST", body: JSON.stringify({ name, values: collectValues(node) }) });
                await loadPresets(true);
                if (presetWidget) { presetWidget.options.values = result.names; presetWidget.value = name; }
                toast("success", `Preset "${name}" saved`);
            } catch (e) { toast("error", "Save preset failed", e.message); }
        });
    } }));
    actions.appendChild(el("button", { class: "llama-srv-btn danger", text: "🗑 Delete current preset", onclick: async () => {
        const name = presetWidget?.value;
        if (!name || name === CUSTOM_PRESET) return toast("warn", "Select a user preset first");
        try {
            const result = await apiJson("/presets/delete", { method: "POST", body: JSON.stringify({ name }) });
            await loadPresets(true);
            presetWidget.options.values = result.names;
            presetWidget.value = CUSTOM_PRESET;
            toast("info", `Preset "${name}" deleted`);
        } catch (e) { toast("error", "Delete preset failed", e.message); }
    } }));
    actions.appendChild(el("button", { class: "llama-srv-btn", text: "♻ Clear result cache", onclick: async () => {
        try { await apiJson("/clear_cache", { method: "POST", body: "{}" }); toast("info", "Inference result cache cleared"); node.__llamaRefresh(); }
        catch (e) { toast("error", "Clear cache failed", e.message); }
    } }));
    actions.appendChild(el("button", { class: "llama-srv-btn", text: "🧹 Clear chat history", onclick: async () => {
        try { await apiJson("/clear_states", { method: "POST", body: "{}" }); toast("info", "Saved conversations cleared"); }
        catch (e) { toast("error", "Clear failed", e.message); }
    } }));
    body.appendChild(el("div", { class: "llama-srv-section" }, [el("h4", { text: "Presets & maintenance" }), actions]));

    const overlay = el("div", { class: "llama-srv-overlay" });
    const close = () => overlay.remove();
    const dialog = el("div", { class: "llama-srv-dialog" }, [
        el("div", { class: "llama-srv-head" }, [
            el("h3", { text: "Llama-cpp Server settings" }),
            el("button", { class: "llama-srv-btn", text: "✕", onclick: close }),
        ]),
        body,
        el("div", { class: "llama-srv-foot" }, [
            el("span", { class: "hint", text: "Changes apply immediately and are saved with the workflow. Restart the server for launch settings to take effect." }),
            el("button", { class: "llama-srv-btn", text: "Close", onclick: close }),
        ]),
    ]);
    overlay.appendChild(dialog);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    const onKey = (e) => { if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); } };
    document.addEventListener("keydown", onKey);
    document.body.appendChild(overlay);
}

app.registerExtension({
    name: "llama_cpp_vlm.server",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;
            node.__llamaApplyingPreset = false;

            // ---- hide the rarely-changed widgets ---------------------------------
            for (const w of node.widgets || []) {
                if (!VISIBLE.has(w.name)) hideWidget(w);
            }

            // ---- status line ---------------------------------------------------------
            const status = node.addWidget("text", "status", "○ unknown", () => {}, { serialize: false });
            status.__llamaExtra = true;
            status.disabled = true;
            const setStatus = (text, color) => {
                status.value = text;
                node.__llamaStatusColor = color;
                node.setDirtyCanvas(true, false);
            };

            // ---- start / stop toggle ------------------------------------------------
            const toggle = node.addWidget("button", "▶ Start server", null, async () => {
                const values = collectValues(node);
                if (node.__llamaRunning) {
                    try { await apiJson("/stop", { method: "POST", body: "{}" }); toast("info", "llama-server stopped"); }
                    catch (e) { toast("error", "Stop failed", e.message); }
                } else if (values.mode === "connect") {
                    toast("info", "Connect mode", "Nothing to start: this node only connects to base_url. Switch mode in ⚙ Settings.");
                } else {
                    try {
                        const result = await apiJson("/start", { method: "POST", body: JSON.stringify(values) });
                        toast(result.started ? "success" : "info", result.started ? "llama-server starting…" : result.message);
                    } catch (e) { toast("error", "Start failed", e.message, 8000); }
                }
                setTimeout(() => node.__llamaRefresh(), 500);
            }, { serialize: false });
            toggle.__llamaExtra = true;

            // ---- settings panel ------------------------------------------------------
            const settings = node.addWidget("button", "⚙ Settings", null, () => openSettings(node, nodeData), { serialize: false });
            settings.__llamaExtra = true;

            // ---- preset application ----------------------------------------------
            const applyPreset = async (name) => {
                if (!name || name === CUSTOM_PRESET) return;
                let presets;
                try { presets = await loadPresets(); } catch (e) { return toast("error", "Cannot load presets", e.message); }
                const preset = presets.user[name] || presets.builtin[name];
                if (!preset) return;
                node.__llamaApplyingPreset = true;
                try {
                    for (const key of presets.keys) {
                        if (!(key in preset)) continue;
                        const w = widget(node, key);
                        if (w) w.value = preset[key];
                    }
                    for (const key of ["model", "mmproj"]) {
                        const match = preset[`${key}_match`];
                        if (!match) continue;
                        const w = widget(node, key);
                        const found = pickOption(w?.options?.values, match);
                        if (found) w.value = found;
                        else toast("warn", `No ${key} matching "${match}"`, "Download it into models/LLM or pick one manually.", 6000);
                    }
                } finally {
                    node.__llamaApplyingPreset = false;
                }
                node.setDirtyCanvas(true, true);
            };

            const presetWidget = widget(node, "preset");
            if (presetWidget) {
                const original = presetWidget.callback;
                presetWidget.callback = function (value) {
                    original?.apply(this, arguments);
                    applyPreset(value);
                };
            }

            // Any manual edit on the canvas flips the preset combo back to "Custom".
            for (const w of node.widgets) {
                if (w.__llamaExtra || w.name === "preset") continue;
                const original = w.callback;
                w.callback = function () {
                    original?.apply(this, arguments);
                    if (!node.__llamaApplyingPreset && presetWidget && presetWidget.value !== CUSTOM_PRESET) {
                        presetWidget.value = CUSTOM_PRESET;
                    }
                };
            }

            // ---- status polling --------------------------------------------------------
            node.__llamaRefresh = async () => {
                const values = collectValues(node);
                try {
                    const params = new URLSearchParams({ base_url: values.base_url || "" });
                    const data = await apiJson(`/status?${params}`);
                    const managed = data.managed;
                    node.__llamaRunning = managed;
                    const props = data.props || {};
                    const model = props.model ? props.model.replace(/\.gguf$/i, "") : "";
                    const caps = props.vision ? "vision" : (props.model ? "text-only" : "");
                    const ctx = props.n_ctx ? `${props.n_ctx} ctx` : "";
                    const cache = data.cache && data.cache.max_entries > 0 ? `cache ${data.cache.entries}/${data.cache.max_entries}` : "";
                    const detail = [model, caps, ctx, cache].filter(Boolean).join(" · ");

                    if (managed && data.healthy) {
                        setStatus(`● running · ${detail}`, "#5fbf5f");
                    } else if (managed) {
                        setStatus("◌ loading model…", "#e0b04a");
                    } else if (data.healthy) {
                        setStatus(`● external · ${detail}`, "#5f9fdf");
                    } else if (data.reachable) {
                        setStatus("◌ server loading (external)", "#e0b04a");
                    } else if (data.error) {
                        setStatus(`✖ ${data.error}`, "#df5f5f");
                        if (data.log_tail && node.__llamaLastError !== data.error) {
                            node.__llamaLastError = data.error;
                            toast("error", data.error, data.log_tail, 10000);
                        }
                    } else {
                        setStatus(`○ stopped · ${values.mode === "connect" ? values.base_url : "launch mode"}`, "#888");
                    }
                    toggle.label = managed ? "■ Stop server" : "▶ Start server";
                } catch (e) {
                    setStatus("? status unavailable", "#888");
                }
                node.setDirtyCanvas(true, false);
            };

            node.__llamaTimer = setInterval(() => {
                if (!node.graph) { clearInterval(node.__llamaTimer); return; }
                node.__llamaRefresh();
            }, POLL_MS);
            setTimeout(() => node.__llamaRefresh(), 300);
            // Hidden widgets shrink the node: recompute the size once.
            setTimeout(() => { node.setSize(node.computeSize()); node.setDirtyCanvas(true, true); }, 0);
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this.__llamaTimer) clearInterval(this.__llamaTimer);
            onRemoved?.apply(this, arguments);
        };

        // Colour the status text by state.
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            const status = widget(this, "status");
            if (status && this.__llamaStatusColor) {
                status.options = status.options || {};
                status.options.color = this.__llamaStatusColor;
            }
        };
    },
});
