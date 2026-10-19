"use strict";

// Capture timing: ~0.75 s looking straight (baseline), then ~3 s turning. The server accepts 6-48 frames.
const FRAME_INTERVAL_MS = 125;
const BASELINE_FRAMES = 6;
const TOTAL_FRAMES = 30;
const FRAME_WIDTH = 640;

const $ = (selector, root = document) => root.querySelector(selector);
const video = $("#camera");
const canvas = document.createElement("canvas");
const progress = $("#progress");
const prompt = $("#prompt");
const result = $("#result");
let stream = null;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function startCamera() {
  if (stream) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser cannot access a camera (it needs HTTPS or localhost).");
  }
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  $("#camera-off").hidden = true;
}

function stopCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  $("#camera-off").hidden = false;
}

function grabFrame() {
  const scale = Math.min(1, FRAME_WIDTH / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height); // raw, not mirrored
  return canvas.toDataURL("image/jpeg", 0.85);
}

function showPrompt(step, direction) {
  prompt.hidden = false;
  prompt.dataset.step = step;
  prompt.dataset.direction = direction || "";
  $("#prompt-arrow").textContent = step === "turn" ? (direction === "left" ? "←" : "→") : "●";
  $("#prompt-text").textContent =
    step === "ready" ? "Get ready…" :
    step === "straight" ? "Look straight at the camera" :
    `Slowly turn your head to the ${direction}`;
}

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    const message = Array.isArray(detail)
      ? detail.map((e) => `${e.loc?.at(-1) ?? "input"}: ${e.msg}`).join("; ")
      : detail?.message ?? `Request failed (${response.status}).`;
    throw new Error(message);
  }
  return data;
}

async function captureChallenge() {
  await startCamera();
  const challenge = await api("/api/challenges");
  progress.hidden = false;
  progress.value = 0;
  showPrompt("ready");
  await sleep(900);
  const frames = [];
  for (let i = 0; i < TOTAL_FRAMES; i++) {
    if (i < BASELINE_FRAMES) showPrompt("straight");
    else showPrompt("turn", challenge.direction);
    frames.push(grabFrame());
    progress.value = (i + 1) / TOTAL_FRAMES;
    await sleep(FRAME_INTERVAL_MS);
  }
  prompt.hidden = true;
  progress.hidden = true;
  return { challenge_id: challenge.challenge_id, frames };
}

function showResult(ok, title, message) {
  result.replaceChildren();
  const alert = document.createElement("div");
  alert.setAttribute("role", "alert");
  alert.dataset.variant = ok ? "success" : "error";
  const strong = document.createElement("strong");
  strong.textContent = title;
  alert.append(strong, document.createTextNode(` ${message}`));
  result.append(alert);
  window.ot?.toast(message, title, { variant: ok ? "success" : "danger" });
}

function formValues(form) {
  return Object.fromEntries(new FormData(form).entries());
}

function validate(form) {
  let ok = true;
  for (const input of form.querySelectorAll("input")) {
    const field = input.closest("[data-field]");
    const valid = input.checkValidity();
    input.toggleAttribute("aria-invalid", !valid);
    if (field) field.toggleAttribute("aria-invalid", !valid);
    ok &&= valid;
  }
  return ok;
}

function handle(formId, run) {
  const form = $(formId);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!validate(form)) {
      showResult(false, "Check the form.", "Some fields are missing or invalid.");
      return;
    }
    const button = event.submitter ?? form.querySelector("button[type=submit]");
    const buttons = form.querySelectorAll("button");
    buttons.forEach((b) => (b.disabled = true));
    button.setAttribute("aria-busy", "true");
    button.dataset.spinner = "small";
    try {
      const capture = await captureChallenge();
      const [title, message] = await run(formValues(form), capture, button);
      showResult(true, title, message);
    } catch (error) {
      showResult(false, "Not completed.", error.message);
    } finally {
      stopCamera(); // the camera is only on while a check is running
      prompt.hidden = true;
      progress.hidden = true;
      button.removeAttribute("aria-busy");
      buttons.forEach((b) => (b.disabled = false));
    }
  });
}

handle("#pay-form", async (v, capture) => {
  const r = await api("/api/pay", {
    email: v.email, pin: v.pin, amount: v.amount, merchant: v.merchant || "Demo merchant", capture,
  });
  return ["Approved.", r.message];
});

handle("#enrol-form", async (v, capture) => {
  const r = await api("/api/enrol", { email: v.email, pin: v.pin, consent: v.consent === "on", capture });
  return ["Enrolled.", r.message];
});

handle("#identify-form", async (_v, capture) => {
  const r = await api("/api/identify", { capture });
  return [r.email ? "Match found." : "No match.", r.message];
});

handle("#account-form", async (v, capture, button) => {
  const path = button.value === "delete" ? "/api/account/delete" : "/api/account/face";
  const r = await api(path, { email: v.email, pin: v.pin, capture });
  return ["Done.", r.message];
});

$("#theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.style.colorScheme === "dark" ||
    (!root.style.colorScheme && matchMedia("(prefers-color-scheme: dark)").matches);
  root.style.colorScheme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.style.colorScheme); } catch { /* storage unavailable */ }
});

try {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.style.colorScheme = saved;
} catch { /* storage unavailable */ }
