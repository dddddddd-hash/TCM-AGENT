const messagesEl = document.querySelector("#messages");
const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const sendBtn = document.querySelector("#sendBtn");
const statusEl = document.querySelector("#status");
const evalOutput = document.querySelector("#evalOutput");

const metricEls = {
  confidence: document.querySelector("#confidence"),
  safetyRoute: document.querySelector("#safetyRoute"),
  iterations: document.querySelector("#iterations"),
  evidenceCount: document.querySelector("#evidenceCount"),
  citations: document.querySelector("#citations"),
};

const buttons = {
  new: document.querySelector("#newBtn"),
  rebuild: document.querySelector("#rebuildBtn"),
  eval: document.querySelector("#evalBtn"),
  safety: document.querySelector("#safetyBtn"),
};

function clearEmptyState() {
  const empty = messagesEl.querySelector(".empty-state");
  if (empty) empty.remove();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInline(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return html;
}

function renderMarkdown(text) {
  const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let listItems = [];
  let listType = "ul";
  let codeLines = [];
  let inCode = false;

  function flushParagraph() {
    if (!paragraph.length) return;
    html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (!listItems.length) return;
    html.push(`<${listType}>${listItems.map((item) => `<li>${renderInline(item)}</li>`).join("")}</${listType}>`);
    listItems = [];
    listType = "ul";
  }

  function flushCode() {
    html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    codeLines = [];
  }

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("```")) {
      if (inCode) {
        flushCode();
        inCode = false;
      } else {
        flushParagraph();
        flushList();
        inCode = true;
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length, 4);
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    const unordered = trimmed.match(/^[-*]\s+(.+)$/);
    const numbered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (unordered || numbered) {
      flushParagraph();
      const currentType = numbered ? "ol" : "ul";
      if (listItems.length && listType !== currentType) flushList();
      listType = currentType;
      listItems.push((unordered || numbered)[1]);
      continue;
    }

    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();
  if (inCode) flushCode();
  return html.join("");
}

function addMessage(role, text) {
  clearEmptyState();
  const message = document.createElement("article");
  message.className = `message ${role}`;

  const label = document.createElement("div");
  label.className = "role";
  label.textContent = role === "user" ? "You" : "Agent";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    bubble.classList.add("markdown-body");
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }

  message.append(label, bubble);
  messagesEl.append(message);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setBusy(isBusy, label = "Send") {
  sendBtn.disabled = isBusy;
  sendBtn.textContent = isBusy ? "Working" : label;
  Object.values(buttons).forEach((btn) => {
    btn.disabled = isBusy;
  });
}

async function api(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function updateMetrics(data) {
  metricEls.confidence.textContent = `${Math.round((data.confidence || 0) * 100)}%`;
  metricEls.safetyRoute.textContent = `${data.safety_action || "proceed"} / ${data.safety_risk_level || "low"}`;
  metricEls.iterations.textContent = data.retrieval_iterations ?? "-";
  metricEls.evidenceCount.textContent = data.evidence_count ?? "-";

  const citations = data.citations || [];
  metricEls.citations.innerHTML = "";
  if (!citations.length) {
    metricEls.citations.className = "citations muted";
    metricEls.citations.textContent = "No citations yet";
    return;
  }

  metricEls.citations.className = "citations";
  citations.forEach((citation, index) => {
    const item = document.createElement("div");
    item.className = "citation";
    item.textContent = `${index + 1}. ${citation}`;
    metricEls.citations.append(item);
  });
}

function formatSummary(summary) {
  if (!summary) return "No result";
  const lines = [];
  if ("avg_faithfulness" in summary) {
    lines.push(`Samples: ${summary.total_samples}`);
    lines.push(`Answered: ${summary.answer_samples}`);
    lines.push(`Faithfulness: ${Number(summary.avg_faithfulness).toFixed(3)}`);
    lines.push(`Answer Relevancy: ${Number(summary.avg_answer_relevancy).toFixed(3)}`);
    lines.push(`Route Accuracy: ${(Number(summary.route_accuracy) * 100).toFixed(1)}%`);
    lines.push(`Clarify Success: ${(Number(summary.clarify_success_rate) * 100).toFixed(1)}%`);
    lines.push(`Avg Contexts: ${Number(summary.avg_evidence_count).toFixed(2)}`);
    lines.push(`Report: ${summary.report_path}`);
  } else if ("accuracy" in summary) {
    lines.push(`Total: ${summary.total}`);
    lines.push(`Passed: ${summary.passed}`);
    lines.push(`Failed: ${summary.failed}`);
    lines.push(`Accuracy: ${(Number(summary.accuracy) * 100).toFixed(1)}%`);
  }
  return lines.join("\n");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage("user", message);
  input.value = "";
  input.style.height = "auto";
  setBusy(true);

  try {
    const data = await api("/api/chat", { message });
    addMessage("assistant", data.answer);
    updateMetrics(data);
  } catch (error) {
    addMessage("assistant", `Request failed: ${error.message}`);
  } finally {
    setBusy(false);
    input.focus();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

buttons.new.addEventListener("click", async () => {
  setBusy(true);
  try {
    await api("/api/new");
    messagesEl.innerHTML = "";
    addMessage("assistant", "New conversation started. Short-term memory has been cleared.");
  } catch (error) {
    addMessage("assistant", `Failed to start a new conversation: ${error.message}`);
  } finally {
    setBusy(false);
  }
});

buttons.rebuild.addEventListener("click", async () => {
  evalOutput.textContent = "Rebuilding knowledge index. This may take a while...";
  setBusy(true);
  try {
    await api("/api/rebuild");
    evalOutput.textContent = "Knowledge index rebuilt.";
  } catch (error) {
    evalOutput.textContent = `Rebuild failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
});

buttons.eval.addEventListener("click", async () => {
  evalOutput.textContent = "Running batch RAG evaluation...";
  setBusy(true);
  try {
    const data = await api("/api/eval");
    evalOutput.textContent = formatSummary(data.summary);
  } catch (error) {
    evalOutput.textContent = `Evaluation failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
});

buttons.safety.addEventListener("click", async () => {
  evalOutput.textContent = "Running safety evaluation...";
  setBusy(true);
  try {
    const data = await api("/api/safetyeval");
    evalOutput.textContent = formatSummary(data.summary);
  } catch (error) {
    evalOutput.textContent = `Safety evaluation failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
});

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    statusEl.textContent = data.ok ? "Agent ready" : `Status: ${data.status}`;
    statusEl.className = data.ok ? "status-dot ready" : "status-dot";
  } catch (error) {
    statusEl.textContent = "Disconnected";
    statusEl.className = "status-dot error";
  }
}

checkHealth();
setInterval(checkHealth, 5000);
