/**
 * Frontend for the "Llama-cpp Server" node.
 *
 * Adds a Start/Stop toggle button, a live status line, preset save/delete
 * buttons and a "clear chat history" button. All actions go through the HTTP
 * routes registered in nodes_server.py (/llama_cpp_vlm/server/...).
 */
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "llama_cpp_server";
const ROUTE = "/llama_cpp_vlm/server";
const CUSTOM_PRESET = "Custom";
const POLL_MS = 4000;

let presetCache = null;

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

app.registerExtension({
    name: "llama_cpp_vlm.server",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;
            node.__llamaApplyingPreset = false;

            // ---- status line ------------------------------------------------
            const status = node.addWidget("text", "status", "○ unknown", () => {}, { serialize: false });
            status.__llamaExtra = true;
            status.disabled = true;

            const setStatus = (text, color) => {
                status.value = text;
                node.__llamaStatusColor = color;
                node.setDirtyCanvas(true, false);
            };

            // ---- start / stop toggle -----------------------------------------
            const toggle = node.addWidget("button", "▶ Start server", null, async () => {
                const values = collectValues(node);
                if (node.__llamaRunning) {
                    try {
                        await apiJson("/stop", { method: "POST", body: "{}" });
                        toast("info", "llama-server stopped");
                    } catch (e) {
                        toast("error", "Stop failed", e.message);
                    }
                } else if (values.mode === "connect") {
                    toast("info", "Connect mode", "Nothing to start: this node only connects to base_url.");
                } else {
                    try {
                        const result = await apiJson("/start", { method: "POST", body: JSON.stringify(values) });
                        toast(result.started ? "success" : "info", result.started ? "llama-server starting…" : result.message);
                    } catch (e) {
                        toast("error", "Start failed", e.message, 8000);
                    }
                }
                setTimeout(() => node.__llamaRefresh(), 500);
            }, { serialize: false });
            toggle.__llamaExtra = true;

            // ---- preset buttons --------------------------------------------------
            const savePreset = node.addWidget("button", "💾 Save preset", null, () => {
                const current = widget(node, "preset")?.value;
                promptText("Preset name", current && current !== CUSTOM_PRESET ? current : "", async (name) => {
                    name = (name || "").trim();
                    if (!name) return;
                    try {
                        const result = await apiJson("/presets", {
                            method: "POST",
                            body: JSON.stringify({ name, values: collectValues(node) }),
                        });
                        await loadPresets(true);
                        const presetWidget = widget(node, "preset");
                        if (presetWidget) {
                            presetWidget.options.values = result.names;
                            presetWidget.value = name;
                        }
                        toast("success", `Preset "${name}" saved`);
                    } catch (e) {
                        toast("error", "Save preset failed", e.message);
                    }
                });
            }, { serialize: false });
            savePreset.__llamaExtra = true;

            const deletePreset = node.addWidget("button", "🗑 Delete preset", null, async () => {
                const presetWidget = widget(node, "preset");
                const name = presetWidget?.value;
                if (!name || name === CUSTOM_PRESET) return toast("warn", "Select a user preset first");
                try {
                    const result = await apiJson("/presets/delete", { method: "POST", body: JSON.stringify({ name }) });
                    await loadPresets(true);
                    presetWidget.options.values = result.names;
                    presetWidget.value = CUSTOM_PRESET;
                    toast("info", `Preset "${name}" deleted`);
                } catch (e) {
                    toast("error", "Delete preset failed", e.message);
                }
            }, { serialize: false });
            deletePreset.__llamaExtra = true;

            const clearStates = node.addWidget("button", "🧹 Clear chat history", null, async () => {
                try {
                    await apiJson("/clear_states", { method: "POST", body: "{}" });
                    toast("info", "Saved conversations cleared");
                } catch (e) {
                    toast("error", "Clear failed", e.message);
                }
            }, { serialize: false });
            clearStates.__llamaExtra = true;

            const clearCache = node.addWidget("button", "♻ Clear result cache", null, async () => {
                try {
                    await apiJson("/clear_cache", { method: "POST", body: "{}" });
                    toast("info", "Inference result cache cleared");
                    node.__llamaRefresh();
                } catch (e) {
                    toast("error", "Clear cache failed", e.message);
                }
            }, { serialize: false });
            clearCache.__llamaExtra = true;

            // ---- preset application -------------------------------------------------
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

            // Any manual edit flips the preset combo back to "Custom".
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

            // ---- status polling ------------------------------------------------------
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
                        setStatus("○ stopped", "#888");
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
