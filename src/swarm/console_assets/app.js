(function () {
  "use strict";
  var sites = [], current = null, timer = null;
  //: What has been typed, per site. Switching tabs used to redraw the form
  //: from the site definition and silently discard it - which is worst in the
  //: one flow this tool exists for: read the plan, switch to the worker, and
  //: find the objective you wanted to copy from is gone.
  var typed = {};
  var $ = function (id) { return document.getElementById(id); };

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;   // never as markup
    return node;
  }
  function card(title, body, cls) {
    var box = el("div", "card" + (cls ? " " + cls : ""));
    if (title) box.appendChild(el("h2", null, title));
    box.appendChild(body);
    return box;
  }
  function pre(text) { return el("pre", null, text); }

  function api(path, payload) {
    var opts = { headers: { "Content-Type": "application/json" } };
    if (payload) { opts.method = "POST"; opts.body = JSON.stringify(payload); }
    return fetch(path, opts).then(function (r) {
      return r.json().then(function (b) { return { ok: r.ok, status: r.status, body: b }; });
    });
  }

  function values() {
    var out = typed[current.key] || {};
    current.fields.forEach(function (f) {
      var node = document.querySelector('[name="' + f.name + '"]');
      if (node) out[f.name] = node.value;
    });
    typed[current.key] = out;
    return out;
  }

  function drawForm() {
    var form = $("form");
    var kept = typed[current.key] || {};
    form.textContent = "";
    $("blurb").textContent = current.blurb;
    current.fields.forEach(function (f) {
      form.appendChild(el("label", null, f.label));
      var node = el(f.kind === "area" ? "textarea" : "input");
      node.name = f.name;
      node.placeholder = f.placeholder || "";
      node.value = kept[f.name] !== undefined ? kept[f.name] : (f.value || "");
      node.oninput = function () { values(); };
      form.appendChild(node);
    });
  }

  function drawTabs() {
    var tabs = $("tabs");
    tabs.textContent = "";
    sites.forEach(function (site) {
      var b = el("button", "tab", site.label);
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", String(site.key === current.key));
      b.onclick = function () {
        values();               // keep what is on screen before replacing it
        current = site;
        drawTabs();
        drawForm();
      };
      tabs.appendChild(b);
    });
  }

  function show(nodes) {
    var out = $("out");
    out.textContent = "";
    nodes.filter(Boolean).forEach(function (n) { out.appendChild(n); });
  }

  function promptCard(p) {
    var body = el("div");
    body.appendChild(el("h2", null, "system"));
    body.appendChild(pre(p.system));
    body.appendChild(el("h2", null, "human"));
    body.appendChild(pre(p.human));
    return card("prompt · " + p.chars + " chars", body);
  }

  function errorCard(e) {
    var body = el("div");
    body.appendChild(pre(e.type + ": " + e.message));
    if (e.fix) {
      var fix = el("p", "fix");
      fix.appendChild(el("strong", null, "Try: "));
      fix.appendChild(el("code", null, e.fix));
      body.appendChild(fix);
    }
    if (e.traceback) {
      var d = el("details");
      d.appendChild(el("summary", null, "traceback"));
      d.appendChild(pre(e.traceback));
      body.appendChild(d);
    }
    return card("failed", body, "err");
  }

  function captureCard(c) {
    var body = el("div");
    var pills = el("div");
    [["model", c.model], ["schema", c.schema_name],
     ["total", c.total_s !== null ? c.total_s + "s" : "?"],
     ["load", c.load_s !== null ? c.load_s + "s" : "?"],
     ["in", c.prompt_tokens], ["out", c.output_tokens]].forEach(function (pair) {
      if (pair[1] === null || pair[1] === undefined || pair[1] === "") return;
      var p = el("span", "pill", pair[0] + " " + pair[1]);
      p.style.marginRight = "6px";
      pills.appendChild(p);
    });
    body.appendChild(pills);
    if (c.response && c.response.text) {
      body.appendChild(el("h2", null, "raw response"));
      body.appendChild(pre(c.response.text));
    }
    return card("the call", body);
  }

  function resultCard(site, r) {
    var body = el("div");
    if (site === "stack") {
      body.appendChild(el("pre", null, r.stack));
      return card("answer", body);
    }
    if (site === "planner") {
      if (r.reasoning) {
        body.appendChild(el("h2", null, "reasoning"));
        body.appendChild(pre(r.reasoning));
      }
      (r.tasks || []).forEach(function (task, i) {
        var d = el("details", "file");
        d.open = true;
        d.appendChild(el("summary", null, (i + 1) + ". " + task.id));
        d.appendChild(pre(
          task.goal
          + "\n\nfiles:       " + (task.files.join(", ") || "(none)")
          + "\ndepends on:  " + (task.depends_on.join(", ") || "(nothing)")
        ));
        body.appendChild(d);
      });
      return card("plan · " + (r.tasks || []).length + " task(s)", body);
    }
    if (!r.edits || !r.edits.length) {
      body.appendChild(el("p", "empty", "the model returned no edits"));
      return card("answer", body);
    }
    r.edits.forEach(function (edit) {
      var d = el("details", "file");
      d.appendChild(el("summary", null, edit.path + "  ·  " + edit.chars + " chars"));
      d.appendChild(pre(edit.content));
      body.appendChild(d);
    });
    return card("answer · " + r.edits.length + " file(s)", body);
  }

  function waiting(job) {
    var body = el("div");
    body.appendChild(el("p", null, "calling the model — " + job.elapsed_s + "s elapsed"));
    body.appendChild(el("p", "empty",
      "A cold model loads before it generates; the first call of the day is the slow one."));
    return card("running", body);
  }

  function poll(id, promptNode) {
    api("/status?id=" + encodeURIComponent(id)).then(function (res) {
      var job = res.body;
      if (job.state === "running") {
        show([waiting(job), promptNode]);
        timer = setTimeout(function () { poll(id, promptNode); }, 1000);
        return;
      }
      $("go").disabled = false;
      show([
        job.state === "error" ? errorCard(job.error) : resultCard(job.site, job.result),
        job.capture ? captureCard(job.capture) : null,
        promptNode
      ]);
    });
  }

  function fire() {
    $("go").disabled = true;
    clearTimeout(timer);
    var payload = { site: current.key, values: values() };
    api("/prompt", payload).then(function (p) {
      var promptNode = p.ok ? promptCard(p.body) : null;
      if (!p.ok) { $("go").disabled = false; show([errorCard({ type: "bad input",
                    message: p.body.error, fix: p.body.fix })]); return; }
      show([card("running", el("p", null, "starting…")), promptNode]);
      api("/run", payload).then(function (r) {
        if (!r.ok) {
          $("go").disabled = false;
          show([errorCard({ type: "refused", message: r.body.error, fix: r.body.fix }), promptNode]);
          return;
        }
        poll(r.body.id, promptNode);
      });
    });
  }

  function peek() {
    api("/prompt", { site: current.key, values: values() }).then(function (p) {
      show([p.ok ? promptCard(p.body)
                 : errorCard({ type: "bad input", message: p.body.error, fix: p.body.fix })]);
    });
  }

  $("go").onclick = function (e) { e.preventDefault(); fire(); };
  $("peek").onclick = function (e) { e.preventDefault(); peek(); };

  api("/sites").then(function (res) {
    sites = res.body.sites;
    //: Never clobber a tab the operator has already chosen. This resolves once
    //: at load, but a slow response and a fast click put the selection back on
    //: the first site while leaving the other one's form on screen.
    current = current || sites[0];
    var m = res.body.models;
    $("models").textContent = "worker " + m.worker + "  ·  orchestrator " + m.orchestrator
                            + "  ·  " + m.base_url;
    $("hint").textContent = "Captures are written for every call, successful or not.";
    drawTabs();
    drawForm();
  });
})();
