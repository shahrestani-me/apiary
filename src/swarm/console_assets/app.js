(function () {
  "use strict";
  var sites = [], current = null, timer = null;
  //: What has been typed, per site. Switching tabs used to redraw the form
  //: from the site definition and silently discard it - which is worst in the
  //: one flow this tool exists for: read the plan, switch to the worker, and
  //: find the objective you wanted to copy from is gone.
  var typed = {};
  //: The swarm tab wears two faces. "describe" borrows the intake site's
  //: plain-language questions and hides every technical field; "advanced" is
  //: the raw run form. A variable rather than storage: the choice should
  //: survive tab switches, not page reloads.
  var wizardMode = "describe";
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
      if (!node) return;
      out[f.name] = node.type === "checkbox" ? (node.checked ? "1" : "") : node.value;
    });
    typed[current.key] = out;
    return out;
  }

  //: The intake site is a normal model-call site, but the swarm tab embeds its
  //: questions as the "Describe it" wizard. Absent from /sites (an older
  //: backend), the wizard simply never appears and the swarm tab is unchanged.
  function intakeSite() {
    for (var i = 0; i < sites.length; i++) {
      if (sites[i].key === "intake") return sites[i];
    }
    return null;
  }

  //: `values()` reads the selected site's fields; in describe mode the fields
  //: on screen belong to the intake site instead, so they get their own reader
  //: and their own slot in `typed` - a mode switch loses nothing either way.
  function intakeValues() {
    var site = intakeSite();
    var out = typed[site.key] || {};
    site.fields.forEach(function (f) {
      var node = document.querySelector('[name="' + f.name + '"]');
      if (!node) return;
      out[f.name] = node.type === "checkbox" ? (node.checked ? "1" : "") : node.value;
    });
    typed[site.key] = out;
    return out;
  }

  function modeToggle() {
    var wrap = el("div", "modes");
    wrap.appendChild(el("span", "modehint", "mode"));
    [["describe", "Describe it"], ["advanced", "Advanced"]].forEach(function (m) {
      var b = el("button", "mode" + (wizardMode === m[0] ? " on" : ""), m[1]);
      b.type = "button";
      b.setAttribute("aria-pressed", String(wizardMode === m[0]));
      b.onclick = function (e) {
        e.preventDefault();
        if (wizardMode === m[0]) return;
        (wizardMode === "describe" ? intakeValues : values)();   // keep what is on screen
        wizardMode = m[0];
        drawForm();
      };
      wrap.appendChild(b);
    });
    return wrap;
  }

  function drawForm() {
    var form = $("form");
    form.textContent = "";
    var swarm = current.kind === "swarm";
    var intake = swarm ? intakeSite() : null;
    var describing = Boolean(intake) && wizardMode === "describe";
    //: In describe mode the form shows the intake site's questions in the
    //: swarm tab's clothes; everywhere else, the selected site's own fields.
    var spec = describing ? intake : current;
    var keep = describing ? intakeValues : values;
    var kept = typed[spec.key] || {};
    $("blurb").textContent = spec.blurb;
    if (intake) form.appendChild(modeToggle());
    spec.fields.forEach(function (f) {
      if (f.kind === "check") {
        var wrap = el("label", "checkline");
        var box = el("input");
        box.type = "checkbox";
        box.name = f.name;
        box.checked = (kept[f.name] !== undefined ? kept[f.name] : f.value) === "1";
        box.onchange = function () { keep(); };
        wrap.appendChild(box);
        wrap.appendChild(el("span", null, f.label));
        form.appendChild(wrap);
        return;
      }
      form.appendChild(el("label", null, f.label));
      var node = el(f.kind === "area" ? "textarea" : "input");
      node.name = f.name;
      node.placeholder = f.placeholder || "";
      node.value = kept[f.name] !== undefined ? kept[f.name] : (f.value || "");
      node.oninput = function () { keep(); };
      form.appendChild(node);
    });
    //: The swarm tab fires a run, not a prompt; there is no prompt to peek at.
    $("peek").style.display = swarm ? "none" : "";
    $("go").textContent = describing ? "Propose a setup"
                          : swarm ? "Run the swarm" : "Fire";
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
        if (site.kind === "swarm") swarmShow();
        else { clearTimeout(boardTimer); clearTimeout(extTimer); }
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

  // ---- the swarm tab: the board, and a whole run streamed ------------------

  var boardTimer = null, runTimer = null, runRepo = "";
  //: Runs this console process spawned, by their swarm run id (parsed from the
  //: log). The external view skips these - the same run must not appear twice,
  //: once from memory and once from its artifacts.
  var extTimer = null, extView = null, ownRunIds = {};

  //: Lifecycle order, mirroring `console_board.COLUMNS`. swarm:failed is a
  //: strip below rather than a column: a ticket needing a human must not hide.
  var BOARD_COLUMNS = [
    ["backlog", "Backlog"], ["ready", "Ready"], ["in_progress", "In progress"],
    ["review", "Review"], ["merged", "Merged"], ["verified", "Verified"]
  ];

  function panel() {
    if (!panel.built) {
      var boardBody = el("div");
      boardBody.appendChild(el("p", "empty",
        "Name a repository (or start a run) and the board follows its swarm:* labels, live."));
      panel.built = {
        board: card("board — read from GitHub, the labels are the truth", boardBody),
        boardBody: boardBody,
        runBox: el("div", "stack")
      };
    }
    return panel.built;
  }

  function swarmShow() {
    show([panel().board, panel().runBox]);
    boardTick();
    //: Adopt a run this page did not start - one fired before a reload, or
    //: from another session. Only when the run area is empty, so switching
    //: tabs never clobbers a view that is already following a run.
    if (!panel().runBox.childNodes.length) {
      api("/swarm/latest").then(function (res) {
        if (!res.ok || panel().runBox.childNodes.length) { extTick(); return; }
        var view = swarmView(res.body);
        if (res.body.state === "running") {
          $("go").disabled = true;
          pollSwarm(res.body.id, view);
        }
        extTick();
      });
    } else {
      extTick();
    }
  }

  // ---- runs the console did not start, read from their artifacts -----------

  function externalCard(b) {
    var state = el("span", "pill", "");
    var age = el("span", "pill", "");
    var idPill = el("span", "pill", b.run_id);
    var strip = el("div", "pills");
    [state, age, idPill].forEach(function (p) { strip.appendChild(p); });

    var links = el("p", "links");
    if (b.repo_url) {
      var a = el("a", null, b.repo);
      a.href = b.repo_url; a.target = "_blank"; a.rel = "noopener";
      links.appendChild(a);
    }
    var note = el("p", "blurb", "");
    var head = el("div");
    head.appendChild(strip);
    head.appendChild(links);
    head.appendChild(note);

    var log = pre("");
    log.className = "log";
    var body = el("div");
    body.appendChild(el("p", "blurb",
      "Started outside this console (a terminal, or before a restart). What follows is "
      + "the run's own recorded cycle log from its artifacts - the board above is its live truth."));
    body.appendChild(log);

    var box = panel().runBox;
    box.textContent = "";
    box.appendChild(card("the run — from its artifacts", head));
    box.appendChild(card("cycle log — as recorded", body));

    return { id: b.run_id, next: 0, root: box.firstChild,
             state: state, age: age, note: note, log: log };
  }

  function absorbExternal(b) {
    if (ownRunIds[b.run_id]) return;              // already on screen from memory
    var box = panel().runBox;
    var ours = extView && box.contains(extView.root);
    if (box.childNodes.length && !ours) return;   // never clobber an own view
    if (!ours || extView.id !== b.run_id) {
      extView = externalCard(b);
    }
    extView.state.textContent = b.state === "quiet"
      ? "quiet — a long model call, or interrupted" : b.state;
    extView.state.className = "pill" + (b.state === "finished" ? " ok" : "");
    if (b.last_event_s !== null && b.state !== "finished") {
      extView.age.textContent = "last event " + Math.round(b.last_event_s) + "s ago";
      extView.age.style.display = "";
    } else {
      extView.age.style.display = "none";
    }
    extView.note.textContent = b.note || "";
    if (b.lines && b.lines.length) {
      extView.log.textContent += b.lines.join("\n") + "\n";
      extView.log.scrollTop = extView.log.scrollHeight;
    }
    extView.next = b.next;
    if (b.repo) runRepo = b.repo;                 // the board follows it too
  }

  function extTick() {
    clearTimeout(extTimer);
    if (!current || current.kind !== "swarm") return;
    //: `run` names the run the counter belongs to: when a newer run has taken
    //: the latest slot, the server answers from line zero instead of slicing
    //: the new run's log with the old run's offset.
    var since = extView ? extView.next : 0;
    var run = extView ? "&run=" + encodeURIComponent(extView.id) : "";
    api("/swarm/external?since=" + since + run).then(function (res) {
      if (!current || current.kind !== "swarm") return;
      if (res.ok) absorbExternal(res.body);
      //: 404 means nothing ever ran - not an error worth a card. Poll faster
      //: while an orchestrator is demonstrably cycling, slower otherwise.
      var wait = res.ok && res.body.state === "active" ? 3000 : 15000;
      extTimer = setTimeout(extTick, wait);
    });
  }

  function boardRepo() {
    if (runRepo) return runRepo;
    var kept = typed["swarm"] || {};
    return (kept.repo || "").trim();
  }

  function boardHint(text) {
    var body = panel().boardBody;
    if (body.querySelector(".board")) return;   // never wipe a rendered board
    body.textContent = "";
    body.appendChild(el("p", "empty", text));
  }

  function boardTick() {
    clearTimeout(boardTimer);
    if (!current || current.kind !== "swarm") return;
    var kept = typed["swarm"] || {};
    if (kept.local === "1" && !runRepo) {
      boardHint("local run — the board follows GitHub's swarm:* labels, and a local run "
                + "never touches GitHub. The log below is the whole story.");
      boardTimer = setTimeout(boardTick, 3000);
      return;
    }
    var repo = boardRepo();
    if (!repo || repo.indexOf("/") < 1) {
      boardTimer = setTimeout(boardTick, 3000);
      return;
    }
    api("/swarm/board?repo=" + encodeURIComponent(repo)).then(function (res) {
      if (current && current.kind === "swarm") {
        if (res.ok) renderBoard(res.body);
        else renderBoardTrouble(res.body);
        boardTimer = setTimeout(boardTick, 5000);
      }
    });
  }

  function ticket(c) {
    var t = el("div", "tcard");
    var head = el("div", "thead");
    var a = el("a", null, "#" + c.number);
    a.href = c.url; a.target = "_blank"; a.rel = "noopener";
    head.appendChild(a);
    if (c.pr) {
      var p = el("a", null, "PR #" + c.pr);
      p.href = c.pr_url; p.target = "_blank"; p.rel = "noopener";
      head.appendChild(p);
    }
    if (c.ci) head.appendChild(el("span", "pill" + (c.ci === "red" ? " bad" : ""),
                                 "post-merge CI: " + c.ci));
    if (c.attempt) head.appendChild(el("span", "pill", "attempt " + c.attempt));
    t.appendChild(head);
    t.appendChild(el("div", "ttitle", c.title));
    return t;
  }

  function renderBoard(b) {
    var body = panel().boardBody;
    body.textContent = "";
    var meta = el("p", "links");
    var a = el("a", null, b.repo);
    a.href = b.repo_url; a.target = "_blank"; a.rel = "noopener";
    meta.appendChild(a);
    body.appendChild(meta);
    var grid = el("div", "board");
    BOARD_COLUMNS.forEach(function (col) {
      var cards = b.columns[col[0]] || [];
      var box = el("div", "col");
      box.appendChild(el("h3", null, col[1] + " · " + cards.length));
      cards.forEach(function (c) { box.appendChild(ticket(c)); });
      grid.appendChild(box);
    });
    body.appendChild(grid);
    if ((b.failed || []).length) {
      var strip = el("div", "failedstrip");
      strip.appendChild(el("h3", null, "Failed — needs a human · " + b.failed.length));
      b.failed.forEach(function (c) { strip.appendChild(ticket(c)); });
      body.appendChild(strip);
    }
    if ((b.errors || []).length) {
      body.appendChild(el("p", "empty",
        b.errors.length + " issue(s) the ledger could not parse; they are never dispatched"));
    }
    (b.notes || []).forEach(function (note) {
      body.appendChild(el("p", "empty", "⚠ " + note));
    });
  }

  function renderBoardTrouble(err) {
    var body = panel().boardBody;
    //: Never wipe a board that was rendering to show a transient error;
    //: a blank board reading "GitHub 502" is worse than a stale one.
    if (body.querySelector(".board")) return;
    body.textContent = "";
    body.appendChild(el("p", "empty", (err && err.error) || "the board could not be read"));
    if (err && err.fix) body.appendChild(el("p", "empty", "Try: " + err.fix));
  }

  function swarmView(job) {
    //: Built once and mutated by every poll, so the log node keeps its scroll
    //: position and the browser is not relaying out a growing <pre> per tick.
    var state = el("span", "pill", "running");
    var strip = el("div", "pills");
    strip.appendChild(state);
    var cycle = el("span", "pill", "");
    var prs = el("span", "pill", "");
    var elapsed = el("span", "pill", "");
    [cycle, prs, elapsed].forEach(function (p) { p.style.display = "none"; strip.appendChild(p); });

    var links = el("p", "links");
    var note = el("p", "blurb", "");
    var stop = el("button", "ghost", "Stop the run");
    stop.onclick = function (e) {
      e.preventDefault();
      stop.disabled = true;
      api("/swarm/stop", { id: job.id });
    };

    var head = el("div");
    head.appendChild(strip);
    head.appendChild(links);
    head.appendChild(note);
    head.appendChild(stop);

    var log = pre("");
    log.className = "log";
    var body = el("div");
    body.appendChild(el("p", "blurb", "$ " + job.command));
    body.appendChild(log);

    var box = panel().runBox;
    box.textContent = "";
    extView = null;   // an own view owns the box now; the external card is gone
    box.appendChild(card("the run", head));
    box.appendChild(card("log — the run's own output, live", body));

    var view = {
      next: 0,
      absorb: function (j) {
        state.textContent = j.state + (j.state === "failed" && j.returncode !== null
                                       ? " · exit " + j.returncode : "");
        state.className = "pill " + (j.state === "done" ? "ok"
                                     : j.state === "running" ? "" : "bad");
        elapsed.textContent = j.elapsed_s + "s";
        elapsed.style.display = "";
        var p = j.progress || {};
        if (p.run_id) ownRunIds[p.run_id] = 1;   // the external view must skip it
        if (p.repo) runRepo = p.repo;   // the board follows the run's repository
        if (p.cycle !== null && p.cycle !== undefined) {
          cycle.textContent = "cycle " + p.cycle;
          cycle.style.display = "";
        }
        if ((p.prs || []).length) {
          prs.textContent = (p.prs.length === 1 ? "PR #" : "PRs #") + p.prs.join(", #");
          prs.style.display = "";
        }
        note.textContent = p.met ? "objective met" : (p.note || "");
        if (p.repo_url && !links.childNodes.length) {
          [["repository", ""], ["issues", "/issues"], ["pull requests", "/pulls"]]
            .forEach(function (pair) {
              var a = el("a", null, pair[0]);
              a.href = p.repo_url + pair[1];      // server-built from the validated slug
              a.target = "_blank";
              a.rel = "noopener";
              links.appendChild(a);
            });
        }
        if (j.lines && j.lines.length) {
          log.textContent += j.lines.join("\n") + "\n";
          log.scrollTop = log.scrollHeight;
        }
        this.next = j.next;
        if (j.state !== "running") {
          stop.style.display = "none";
          $("go").disabled = false;
        }
      }
    };
    view.absorb(job);
    return view;
  }

  function pollSwarm(id, view) {
    api("/swarm/status?id=" + encodeURIComponent(id) + "&since=" + view.next)
      .then(function (res) {
        if (!res.ok) { $("go").disabled = false; return; }
        view.absorb(res.body);
        if (res.body.state === "running") {
          runTimer = setTimeout(function () { pollSwarm(id, view); }, 1000);
        }
      });
  }

  function swarmFire() {
    $("go").disabled = true;
    clearTimeout(runTimer);
    api("/swarm/start", { values: values() }).then(function (r) {
      if (!r.ok) {
        $("go").disabled = false;
        var box = panel().runBox;
        box.textContent = "";
        extView = null;
        box.appendChild(errorCard({ type: "refused", message: r.body.error, fix: r.body.fix }));
        return;
      }
      pollSwarm(r.body.id, swarmView(r.body));
    });
  }

  // ---- "Describe it": the intake wizard inside the swarm tab ---------------

  //: The proposal becomes the swarm form's draft, verbatim. Both buttons on
  //: the card go through here, so what runs and what is shown for adjusting
  //: are the same values by construction.
  function adopt(r) {
    typed["swarm"] = { objective: r.brief, repo: r.repo, stack: r.stack,
                       verify: r.verify, public: "1", auto_merge: "1", local: "" };
  }

  function proposalCard(r) {
    var body = el("div");
    var line = el("p", "proposal");
    line.appendChild(el("span", null, "Create "));
    line.appendChild(el("code", null, r.repo));
    line.appendChild(el("span", null, ", a "
      + (r.public === "1" ? "public" : "private") + " repository built on the "));
    line.appendChild(el("code", null, r.stack));
    line.appendChild(el("span", null, " stack, tested by "));
    line.appendChild(el("code", null, r.verify));
    line.appendChild(el("span", null, "."));
    body.appendChild(line);
    if (r.reason) {
      var why = el("p", "why");
      why.appendChild(el("strong", null, "Why: "));
      why.appendChild(el("span", null, r.reason));
      body.appendChild(why);
    }
    var d = el("details", "file");
    d.appendChild(el("summary", null, "the brief the swarm will receive"));
    d.appendChild(pre(r.brief));
    body.appendChild(d);

    var row = el("div", "row");
    var run = el("button", "go", "Run the swarm");
    run.type = "button";
    run.onclick = function (e) { e.preventDefault(); adopt(r); swarmFire(); };
    var adjust = el("button", "ghost", "Adjust first");
    adjust.type = "button";
    adjust.onclick = function (e) {
      e.preventDefault();
      adopt(r);
      wizardMode = "advanced";
      drawForm();               // every field visible and editable, prefilled
    };
    row.appendChild(run);
    row.appendChild(adjust);
    body.appendChild(row);
    return card("proposed setup — nothing is created until you run it", body);
  }

  //: Fires the intake site through the ordinary /run machinery, but renders
  //: into the swarm tab's run area so the board stays on screen throughout.
  function pollIntake(id) {
    api("/status?id=" + encodeURIComponent(id)).then(function (res) {
      var job = res.body;
      var box = panel().runBox;
      if (job.state === "running") {
        box.textContent = "";
        box.appendChild(waiting(job));
        timer = setTimeout(function () { pollIntake(id); }, 1000);
        return;
      }
      $("go").disabled = false;
      box.textContent = "";
      box.appendChild(job.state === "error" ? errorCard(job.error)
                                            : proposalCard(job.result));
    });
  }

  function intakeFire() {
    $("go").disabled = true;
    clearTimeout(timer);
    var box = panel().runBox;
    box.textContent = "";
    box.appendChild(card("running", el("p", null, "starting…")));
    api("/run", { site: "intake", values: intakeValues() }).then(function (r) {
      if (!r.ok) {              // 409 while another call runs, or a bad input
        $("go").disabled = false;
        box.textContent = "";
        box.appendChild(errorCard({ type: "refused", message: r.body.error, fix: r.body.fix }));
        return;
      }
      pollIntake(r.body.id);
    });
  }

  function fire() {
    if (current.kind === "swarm") {
      if (wizardMode === "describe" && intakeSite()) intakeFire();
      else swarmFire();
      return;
    }
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
    //: First, so it is the default: running the swarm is what the console is
    //: opened for; the model-call tabs are the debugger behind it.
    if (res.body.swarm) sites.unshift(res.body.swarm);
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
    if (current.kind === "swarm") swarmShow();
  });
})();
