(function () {
  "use strict";
  var sites = [], current = null, timer = null;
  //: The swarm tab is the product; the model-call tabs (run / plan / propose /
  //: choose / describe) are its debugger. Open the console with ?debug=1 to
  //: get the full tab strip back.
  var debugMode = /(^|[?&])debug=1(&|$)/.test(location.search);
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
  //: The Start building form, as served under its own key on /sites. Null on
  //: an older backend, in which case the plan card renders exactly as it did
  //: and the button never appears.
  var buildSite = null;
  //: What has been typed into it, kept outside the card: the card is redrawn
  //: on every poll tick of a running build, and a redraw that discarded the
  //: owner would be the same bug `typed` exists to prevent one layer down.
  var buildTyped = {};
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
    //: A project already in the store was founded from its requirement, and
    //: the requirement is immutable - but it lives in the prompt history
    //: below the fields now, not in a read-only textarea. The objective box
    //: becomes the NEW prompt for the next run, empty and editable; only the
    //: repository, the project's identity, stays read-only. "Selected"
    //: engages only when the on-screen values are that project's own: a
    //: selection exists, the store knows it, and the form's repo agrees with
    //: the selection. The describe wizard, "Adjust first" on a proposal, and
    //: New project all fail one of those and stay editable.
    var selected = swarm && !describing && Boolean(selectedProject)
                 && Boolean(findProject(selectedProject))
                 && (kept.repo || "").trim() === selectedProject;
    $("blurb").textContent = spec.blurb;
    //: The project bar sits above both faces of the swarm tab: which project
    //: the form is about is a fact that outlives the mode toggle below it.
    if (swarm) form.appendChild(projectBar());
    if (intake) form.appendChild(modeToggle());
    spec.fields.forEach(function (f) {
      //: Setup-time choices describe a repository that does not exist yet.
      //: A selected project's repository exists, so offering to create it
      //: public, to run it locally, or to change its stack would be the form
      //: arguing with the facts - those fields sit this state out. What is
      //: not on screen still travels: `values()` folds only on-screen nodes
      //: into `typed`, so the stack the selection loaded is what a fired run
      //: submits.
      if (selected && (f.name === "local" || f.name === "public" || f.name === "stack")) return;
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
      //: For a selected project the objective box is a prompt box, and its
      //: label must say so - the field submits the run's objective either way.
      var isPrompt = selected && f.name === "objective";
      //: The repo label promises creation, which is exactly what a selected
      //: project must not be promised - its repository is a fact, not a plan.
      form.appendChild(el("label", null, isPrompt
        ? "New prompt — what the swarm should do next for this project"
        : (selected && f.name === "repo")
          ? "Repository — the project's home on GitHub"
          : f.label));
      var node = el(f.kind === "area" ? "textarea" : "input");
      node.name = f.name;
      node.placeholder = isPrompt
        ? "the founding requirement, and every prompt since, sit in the history below"
        : (f.placeholder || "");
      node.value = kept[f.name] !== undefined ? kept[f.name] : (f.value || "");
      node.oninput = function () { keep(); };
      //: The one field that IS the project. Read-only rather than disabled:
      //: a disabled field drops out of form reads and would leave `values()`
      //: answering from whatever `typed` held before, while a read-only one
      //: still reports the stored value the run must use.
      if (selected && f.name === "repo") {
        node.readOnly = true;
        node.className = "locked";
        node.title = "the repository is the project's identity — "
                     + "switching it means picking another project above";
      }
      form.appendChild(node);
    });
    //: The project's past prompts sit under the fields, so the card reads
    //: top-down as: who this project is, what to ask next, what was asked.
    if (selected) form.appendChild(historyBlock(selectedProject));
    //: The swarm tab fires a run, not a prompt; there is no prompt to peek at.
    $("peek").style.display = swarm ? "none" : "";
    $("go").textContent = describing ? "Propose a setup"
                          : swarm ? "Run the swarm" : "Fire";
  }

  //: Whether the strip stays off screen. Only when the swarm view exists to
  //: take its place: a backend that serves no swarm descriptor would leave a
  //: page with no view at all, so the tabs fall back to being the console.
  function hideTabs() {
    if (debugMode) return false;
    for (var i = 0; i < sites.length; i++) {
      if (sites[i].kind === "swarm") return true;
    }
    return false;
  }

  function drawTabs() {
    var tabs = $("tabs");
    tabs.textContent = "";
    //: One view means no strip - a bar with a single tab is clutter. The node
    //: hides too, because an empty .tabs div still carries its bottom margin.
    if (hideTabs()) { tabs.style.display = "none"; return; }
    tabs.style.display = "";
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
    //: Doctor refuses to construct a failing check without a fix, so every
    //: entry here has one. Listed rather than folded into the headline: the
    //: fix for "no docker daemon" and the fix for "no worker image" are
    //: different commands, and a preflight that showed only the first would
    //: send the operator back for the second one at a time.
    (e.checks || []).forEach(function (c) {
      var line = el("p", "fix");
      line.appendChild(el("strong", null, c.name + ": "));
      line.appendChild(el("span", null, c.detail + " — "));
      line.appendChild(el("code", null, c.fix));
      body.appendChild(line);
    });
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

  //: Only ever a github.com URL over TLS, parsed rather than prefix-matched.
  //: The console builds these server-side from a validated slug and an
  //: integer, so nothing here can fire today - but `href` is the one sink on
  //: this page that executes a string, and a payload that ever carried a
  //: `javascript:` URL would find every other defence spent on textContent.
  //: Parsing is what makes it a real check: a prefix test says yes to
  //: `github.com.example.net`, and `URL` does not.
  function link(text, url) {
    var parsed = null;
    try { parsed = new URL(String(url)); } catch (e) { parsed = null; }
    if (!parsed || parsed.protocol !== "https:" || parsed.hostname !== "github.com") {
      return el("span", null, text);
    }
    var a = el("a", null, text);
    a.href = parsed.href;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    return a;
  }

  //: The plan's own fields, drawn inside the plan card rather than in the tab
  //: form: what is being approved is *this* decomposition, and a form for it
  //: living anywhere else would be a button that outlived the thing it acts on.
  function buildForm() {
    var box = el("div", "buildform");
    buildSite.fields.forEach(function (f) {
      var kept = buildTyped[f.name];
      if (f.kind === "check") {
        var wrap = el("label", "checkline");
        var check = el("input");
        check.type = "checkbox";
        check.name = "build_" + f.name;
        check.checked = (kept !== undefined ? kept : f.value) === "1";
        check.onchange = function () { buildTyped[f.name] = check.checked ? "1" : ""; };
        wrap.appendChild(check);
        wrap.appendChild(el("span", null, f.label));
        box.appendChild(wrap);
        return;
      }
      box.appendChild(el("label", null, f.label));
      var node = el("input");
      node.name = "build_" + f.name;
      node.placeholder = f.placeholder || "";
      node.value = kept !== undefined ? kept : (f.value || "");
      node.oninput = function () { buildTyped[f.name] = node.value; };
      box.appendChild(node);
    });
    return box;
  }

  function buildValues() {
    var out = {};
    buildSite.fields.forEach(function (f) {
      out[f.name] = buildTyped[f.name] !== undefined ? buildTyped[f.name] : (f.value || "");
    });
    //: The objective the plan was drafted from, which is what the repository
    //: is generated from. Read off the planner tab rather than retyped: the
    //: two must be the same string or the repository describes one project
    //: and its backlog decomposes another.
    out.objective = (typed.planner || {}).objective || "";
    return out;
  }

  //: The plan card's action. `planId` is the id of the call that produced the
  //: plan on screen, and it is the whole payload: the server writes the tasks
  //: it returned under that id, so nothing between here and GitHub can put a
  //: different decomposition in front of the one the operator read.
  function startBuilding(planId, button) {
    //: Disabled for the round trip, because the window between the click and
    //: the 202 is exactly long enough to click again - and the second click
    //: would be refused by the server's single flight rather than ignored,
    //: putting a 409 on screen for something the operator did not mean to do.
    button.disabled = true;
    //: The refusal lands *under* the plan rather than replacing it: the fixable
    //: ones are all form fields (an owner, a stack, a token), and a page that
    //: threw the decomposition away to show them would make the operator run
    //: the planner again to get back to the button.
    api("/swarm/build", { plan: planId, values: buildValues() }).then(function (r) {
      if (!r.ok) {
        button.disabled = false;
        $("out").appendChild(errorCard({ type: "refused", message: r.body.error,
                                         fix: r.body.fix, checks: r.body.checks }));
        return;
      }
      clearTimeout(timer);
      pollBuild(r.body.id);
    });
  }

  function pollBuild(id) {
    api("/status?id=" + encodeURIComponent(id)).then(function (res) {
      var job = res.body;
      if (job.state === "running") {
        show([buildWaiting(job)]);
        timer = setTimeout(function () { pollBuild(id); }, 1000);
        return;
      }
      if (job.state === "error") { show([errorCard(job.error)]); return; }
      //: The build is over and the swarm is on the repository it made. The run
      //: gets the swarm tab's own view rather than a second one written here:
      //: same log, same summary strip, same Stop button - and therefore the
      //: same SIGINT, which is what disposes the run's containers.
      var runBox = el("div", "stack");
      show([buildCard(job.result), runBox]);
      var started = job.result && job.result.run;
      if (!started) return;          // no run to latch for, and none was taken
      //: The same latch a swarm-tab run takes, so `absorb`'s `busyRun(false)`
      //: at the end of the run releases something that was actually held. It
      //: is the *second* guard rather than the first: the build itself runs
      //: for minutes before this line, and through all of it the only thing
      //: stopping a second build or a competing run is the server's own gate
      //: in `_swarm_build` / `_swarm_start`. This one keeps the page honest
      //: about what it is already doing; that one keeps two of them apart.
      busyRun(true);
      pollSwarm(started.id, swarmView(started, runBox));
    });
  }

  function buildWaiting(job) {
    var body = el("div");
    body.appendChild(el("p", null, "creating the repository and writing the issues — "
                                  + job.elapsed_s + "s elapsed"));
    body.appendChild(el("p", "empty",
      "The model is not being asked anything; this is GitHub's pace, one issue per task. "
      + "The swarm starts on it as soon as the last issue is written."));
    return card("building", body);
  }

  //: What now exists. The repository first, then one row per issue, then -
  //: with equal weight, which is the point - every task that could not become
  //: one. A build that wrote six issues from eight tasks is not a success with
  //: a footnote, and a page that only listed the six would be the silence #129
  //: names as the failing direction.
  function buildCard(r) {
    var body = el("div");
    var head = el("p");
    head.appendChild(el("strong", null, "repository "));
    head.appendChild(link(r.repo, r.html_url));
    body.appendChild(head);
    body.appendChild(el("p", "empty",
      r.default_branch + " · stack " + r.stack + " · verified by " + r.verify_command));

    (r.issues || []).forEach(function (i) {
      var row = el("div", "file");
      row.appendChild(link("#" + i.number, i.url));
      row.appendChild(el("span", null, "  " + i.task_id + " — " + i.title));
      body.appendChild(row);
    });

    (r.rejected || []).forEach(function (t) {
      var row = el("p", "fix");
      row.appendChild(el("strong", null, "not written: "));
      row.appendChild(el("span", null, t.task_id + " — " + t.reason));
      body.appendChild(row);
    });
    (r.warnings || []).forEach(function (w) {
      body.appendChild(el("p", "empty", w));
    });
    //: The repository and its issues are real even when the loop would not
    //: start, so this is a line on a finished build rather than an error card
    //: replacing it - and it carries the command that picks the work up.
    if (r.run_error) {
      var failed = el("p", "fix");
      failed.appendChild(el("strong", null, "the swarm did not start: "));
      failed.appendChild(el("span", null, r.run_error.error));
      body.appendChild(failed);
      if (r.run_error.fix) body.appendChild(el("p", "fix", "Try: " + r.run_error.fix));
    }

    var title = r.repo + " · " + (r.issues || []).length + " issue(s)";
    if ((r.rejected || []).length) title += ", " + r.rejected.length + " task(s) not written";
    return card(title, body, (r.rejected || []).length ? "err" : null);
  }

  function resultCard(site, r, planId) {
    var body = el("div");
    if (site === "build") return buildCard(r);
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
      //: The action lives on the plan it acts on, and appears only when the
      //: backend offers the form: an older one serves no `build` key and this
      //: card is exactly what it always was.
      if (buildSite && planId) {
        body.appendChild(el("h2", null, "start building"));
        body.appendChild(el("p", "empty", buildSite.blurb));
        body.appendChild(buildForm());
        var go = el("button", "tab", "Start building");
        go.onclick = function (e) { e.preventDefault(); startBuilding(planId, go); };
        body.appendChild(go);
      }
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
        job.state === "error" ? errorCard(job.error)
                              : resultCard(job.site, job.result, job.id),
        job.capture ? captureCard(job.capture) : null,
        promptNode
      ]);
    });
  }

  // ---- the swarm tab: the board, and a whole run streamed ------------------

  var boardTimer = null, runTimer = null, runRepo = "";
  //: Which view owns the run area, as a number that only ever goes up. #130
  //: lets a run view be drawn on the planner tab (under a finished build) as
  //: well as in the swarm tab's run area, and until this existed the page
  //: owned both at once: switching tabs calls `show()`, which detaches the
  //: planner's view, and `swarmShow` then adopts the same run and starts a
  //: second `pollSwarm`. Two chains assigning the single `runTimer`, so
  //: `clearTimeout` could only ever cancel the later - the earlier polled a
  //: node with no reachable Stop until the run ended.
  //:
  //: A generation rather than a job id, because both chains poll the *same*
  //: id and an id cannot tell them apart. And a generation rather than the
  //: `clearTimeout` alone, because that closes only the sleeping half of the
  //: window: a request already in flight when the new view is drawn still
  //: resolves afterwards and reschedules itself, which is the same two chains
  //: through a narrower door. `pollSwarm` checks its own generation before it
  //: touches anything.
  var runGeneration = 0;
  //: Runs this console process spawned, by their swarm run id (parsed from the
  //: log). The external view skips these - the same run must not appear twice,
  //: once from memory and once from its artifacts.
  var extTimer = null, extView = null, ownRunIds = {};
  //: The console's project memory, as served by /projects: most recently
  //: active first. `selectedProject` is a repo slug, or "" while the operator
  //: is describing a project that does not exist yet. `runBusy` is the one
  //: switch for "a swarm run is in flight" - the fire button, the selector
  //: and the New project button all follow it together, because switching a
  //: project's values under a run that is still reading them would make the
  //: form lie about what is running.
  var projects = [], selectedProject = "", runBusy = false;
  var projBar = { select: null, button: null };
  //: Which repository the console's own run view (swarmFire / adoption) is
  //: about, recorded whatever the run's state - so followSelection can tell
  //: whether the card in the run area belongs to the project on the selector.
  var ownRepo = "";

  //: How a finished run is named on the page, keyed by `progress.outcome`.
  //: Four endings, and the ticket asks for the page to survive all four and
  //: say which: `swarm run` exits 0 both when the objective was met and when
  //: `--max-cycles` ran out, so "done" answers the question wrongly half the
  //: time. Absent (an older backend, or a run still going) falls back to the
  //: state, which is what the pill always said.
  //: Keyed by `RunJob.progress.outcome`; `tests/test_console_run.py` pins this
  //: table against the vocabulary `conclude` can actually write, because a
  //: sixth ending added on the server would otherwise fall through the
  //: `|| j.state` below and read "done" - quietly, which is how the wrong
  //: answer this table exists to prevent would come back.
  var OUTCOMES = {
    met: "objective met", capped: "cycle cap reached",
    exhausted: "plan exhausted", stopped: "stopped",
    failed: "failed", done: "done"
  };

  //: Lifecycle order, mirroring `console_board.COLUMNS` - the internal
  //: vocabulary, derived from the world rather than read off labels.
  //: needs-human is a strip below rather than a column: a ticket needing a
  //: human must not hide.
  var BOARD_COLUMNS = [
    ["blocked", "Blocked"], ["eligible", "Eligible"], ["claimed", "Claimed"],
    ["review", "Review"], ["landed", "Landed"], ["verified", "Verified"]
  ];

  function panel() {
    if (!panel.built) {
      var boardBody = el("div");
      boardBody.appendChild(el("p", "empty",
        "Name a repository (or start a run) and the board derives every ticket's state "
        + "from GitHub, live."));
      panel.built = {
        board: card("board — derived from the world, read-only", boardBody),
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
        //: Adoption is for a run this page is *not* already following - one
        //: fired before a reload, or by another session. A run whose view was
        //: just detached by `show()` on the way to this tab is still ours, and
        //: re-adopting it is how the page came to poll one run twice. Drawing
        //: it again is right (the operator is looking at an empty run area);
        //: `swarmView` cancels the poller it replaces, so there is still one.
        //: The same rule the external view follows: a run still going is
        //: adopted whatever the selection, a finished one only beside its own
        //: selected project - otherwise the page opens looking like a project
        //: was chosen, and nobody chose it.
        var prog = res.body.progress || {};
        if (res.body.state === "running") {
          busyRun(true);
          pollSwarm(res.body.id, swarmView(res.body));
        } else if (prog.repo && prog.repo === selectedProject) {
          swarmView(res.body);
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

    return { id: b.run_id, repo: b.repo || "", next: 0, root: box.firstChild,
             state: state, age: age, note: note, log: log };
  }

  function absorbExternal(b) {
    if (ownRunIds[b.run_id]) return;              // already on screen from memory
    var box = panel().runBox;
    var ours = extView && box.contains(extView.root);
    //: A finished run is shown only beside its own project. With nothing
    //: selected - or another project selected - a card and a cycle log on
    //: screen read as a choice the operator never made, which is exactly the
    //: complaint that shaped the selector. A run still going is different:
    //: it is a fact about this machine, and it shows regardless.
    if (b.state === "finished" && b.repo !== selectedProject) {
      if (ours) { box.textContent = ""; extView = null; }
      return;
    }
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
    //: A finished run has an ending recorded, and this is where it is read
    //: (#134). Only when finished: `active` and `quiet` mean the summary is
    //: not on disk, which is a run to keep watching rather than one to draw a
    //: terminal state for.
    if (b.state === "finished") drawOutcome(extView, b.run_id, box);
    //: Only a run that is still going speaks for the page: a finished run's
    //: card stays as an account of what happened, but it must not pick the
    //: board's repo - or, through boot, the project - for an operator who has
    //: chosen nothing yet. And never once a project is selected; from then on
    //: the selection owns the board.
    if (b.repo && !selectedProject && b.state !== "finished") runRepo = b.repo;
  }

  // ---- how the run ended, read from its own summary ------------------------
  //
  //: Step 6 of the epic (#134). A build that stops without saying so reads as
  //: a build that hung, and "finished" has four meanings here - the objective
  //: was met, the round cap was hit, every task reached a terminal state, or a
  //: human is needed. This panel says which, quotes the sentence the *run*
  //: recorded for it, and names the tasks that need a person.
  //:
  //: Drawn from `/swarm/outcome`, which reads `summary.json`, rather than from
  //: the job in the console's memory: an ending read off a child process's
  //: stdout dies with the console, and this one has to survive a reload. It is
  //: also the only account available for a run this console did not start.

  //: Wall clock as a human reads it. A run is minutes to hours long, and
  //: seconds alone stop being legible somewhere around the four-minute mark.
  function hms(s) {
    if (s === null || s === undefined) return "";
    var m = Math.floor(s / 60), h = Math.floor(m / 60);
    if (h) return h + "h " + (m % 60) + "m";
    if (m) return m + "m " + Math.round(s % 60) + "s";
    return Math.round(s) + "s";
  }

  //: Which pill an ending wears. A cap is neutral on purpose - work left open
  //: is neither a success nor a failure - and `exhausted` reads ok because the
  //: goal gate was off, so the run did exactly the work that was planned.
  //: `stopped` is neutral too: the operator pressed Stop, and calling their
  //: own decision a failure is the one reading that is never true.
  function outcomeClass(kind) {
    return kind === "met" || kind === "exhausted" ? " ok" : kind === "failed" ? " bad" : "";
  }

  function fact(list, term, text) {
    list.appendChild(el("dt", null, term));
    list.appendChild(el("dd", null, text));
  }

  //: An issue as a link when the repository is known, and as plain text when
  //: it is not - the same rule every other href on this page follows, because
  //: `repo_url` is built by the server from a validated slug or not at all.
  function issueRef(o, number) {
    if (!o.repo_url) return el("span", null, "#" + number);
    var a = el("a", null, "#" + number);
    a.href = o.repo_url + "/issues/" + number;
    a.target = "_blank";
    a.rel = "noopener";
    return a;
  }

  function outcomeCard(o) {
    var body = el("div");
    var strip = el("div", "pills");
    strip.appendChild(el("span", "pill" + outcomeClass(o.outcome),
                         OUTCOMES[o.outcome] || "ended"));
    strip.appendChild(el("span", "pill",
                         o.cap ? o.cycles + "/" + o.cap + " cycles"
                               : o.cycles + (o.cycles === 1 ? " cycle" : " cycles")));
    if (o.wall_s !== null && o.wall_s !== undefined) {
      strip.appendChild(el("span", "pill", hms(o.wall_s)));
    }
    body.appendChild(strip);

    //: The run's own words, quoted. `close_the_loop` and the goal gate compose
    //: these sentences and the run records the one it ended on, so the page
    //: has no second vocabulary for an ending - which is how "objective met"
    //: and "stopped after 40 cycles" came to read identically as "done".
    body.appendChild(el("p", "proposal", o.reason || "this run recorded no ending"));
    if (o.note) body.appendChild(el("p", "why", o.note));

    var tasks = o.tasks || {}, prs = o.prs || {};
    var merged = (tasks.merged || []).length;
    var human = (tasks.needs_human || []).length;
    var abandoned = (tasks.abandoned || []).length;
    var facts = el("dl", "facts");
    fact(facts, "tasks", merged + " merged · " + human + " needing a human · "
                         + abandoned + " abandoned");
    fact(facts, "pull requests", prs.opened + " opened, " + prs.merged + " merged");
    fact(facts, "cycles", o.cap ? o.cycles + " of " + o.cap
                                : o.cycles + " (no cap was set)");
    //: Reported only when it was measured. Nothing increments the inference
    //: clock yet, and printing "0% inference" off an unmeasured field would be
    //: a claim about the model rather than about the recording.
    fact(facts, "wall clock", (hms(o.wall_s) || "not recorded") + (
      o.inference_share === null || o.inference_share === undefined
        ? " · inference not recorded"
        : " · " + Math.round(o.inference_share * 100) + "% inference"));
    body.appendChild(facts);

    //: Separate from the counts and never folded into them: a task waiting for
    //: a person is the one thing on this panel that must not scroll past, and
    //: "8 tasks" hides it behind the ones that merged.
    if (human) {
      var stripe = el("div", "failedstrip");
      stripe.appendChild(el("h3", null, "needs a human"));
      var row = el("p", "links");
      (tasks.needs_human || []).forEach(function (n) { row.appendChild(issueRef(o, n)); });
      stripe.appendChild(row);
      body.appendChild(stripe);
    }

    var links = el("p", "links");
    if (o.repo_url) {
      var a = el("a", null, o.repo || "repository");
      a.href = o.repo_url;
      a.target = "_blank";
      a.rel = "noopener";
      links.appendChild(a);
    }
    body.appendChild(links);
    //: Text, not a link: a browser will not follow a `file:` href out of an
    //: `http:` document, so the useful form is the path to paste beside
    //: `swarm show` - which is where the events are.
    body.appendChild(el("p", "runpath", "events: " + o.path));
    return card("how it ended — from the run's own summary", body, "outcome");
  }

  //: One fetch per view, inserted above the log so it is the first thing read
  //: when a run stops. A 404 means the summary is not on disk yet - the child
  //: writes it on its way out - so exactly one retry follows, and then the
  //: page stops asking rather than polling a run that never wrote one.
  function drawOutcome(view, runId, box) {
    if (!runId || view.outcomeAsked) return;
    view.outcomeAsked = true;
    var ask = function (retry) {
      api("/swarm/outcome?run=" + encodeURIComponent(runId)).then(function (res) {
        //: Both ways a view can stop owning the run area while this request is
        //: in flight, and neither is reachable from the other: the console's
        //: own views carry a generation, and the artifacts view is identified
        //: by the card it drew. Without the second one, a run fired in the
        //: half-second after this fetch left gets the *previous* run's ending
        //: inserted into its card stack.
        if (view.generation !== undefined && view.generation !== runGeneration) return;
        if (view.root && !box.contains(view.root)) return;
        if (!box.childNodes.length) return;            // the view was replaced
        if (!res.ok) {
          if (res.status === 404 && retry) setTimeout(function () { ask(false); }, 1500);
          return;
        }
        //: `remove()` rather than `removeChild`: `querySelector` searches
        //: descendants, and handing `removeChild` a node that is not a direct
        //: child throws rather than replacing anything.
        var old = box.querySelector(".outcome");
        if (old) old.remove();
        box.insertBefore(outcomeCard(res.body), box.childNodes[1] || null);
      });
    };
    ask(true);
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

  // ---- projects: the selector, and what selecting one means ----------------

  //: One project's prompts as served by /projects/history - newest first,
  //: derived from the run artifacts, so a run fired from a terminal counts
  //: too. Cached per repo for the session because drawForm runs on every
  //: redraw and refetching each time would flicker the list; invalidated at
  //: the two refetch points around a fired run (swarmFire, absorb).
  var promptHistory = {};
  //: The history box currently on screen, refreshed in place the way projBar
  //: is - so a refetch never redraws the whole form under a typist.
  var histBox = null;

  function historyWhen(iso) {
    var d = new Date(iso);
    return isNaN(d) ? iso : d.toLocaleString();
  }

  function renderHistory(box, prompts) {
    box.textContent = "";
    box.appendChild(el("h2", null, "prompt history · " + prompts.length));
    if (!prompts.length) {
      box.appendChild(el("p", "empty",
        "no recorded runs yet — the first prompt founds the history"));
      return;
    }
    prompts.forEach(function (p, i) {
      var d = el("details", "file");
      var s = el("summary");
      s.appendChild(el("span", null, historyWhen(p.started_at)));
      s.appendChild(el("span", "pill" + (p.finished ? " ok" : ""),
                       p.finished ? "finished" : "unfinished"));
      //: The oldest prompt is the requirement the project was founded from;
      //: the history is where it stays visible - and immutable, like the rest.
      if (i === prompts.length - 1) s.appendChild(el("span", "pill", "requirement"));
      d.appendChild(s);
      d.appendChild(pre(p.objective));
      box.appendChild(d);
    });
  }

  function fetchHistory(repo, box) {
    api("/projects/history?repo=" + encodeURIComponent(repo)).then(function (res) {
      if (!res.ok || !res.body || !res.body.prompts) {
        //: A quiet one-line note, never a broken form - the fields above it
        //: still work, and the next selection simply tries again.
        box.textContent = "";
        box.appendChild(el("p", "empty", "the prompt history could not be read"
          + (res.body && res.body.error ? " — " + res.body.error : "")));
        return;
      }
      promptHistory[repo] = res.body.prompts;
      renderHistory(box, res.body.prompts);
    });
  }

  function historyBlock(repo) {
    var box = el("div", "history");
    histBox = box;
    if (promptHistory[repo]) {
      renderHistory(box, promptHistory[repo]);
    } else {
      box.appendChild(el("p", "empty", "reading the prompt history…"));
      fetchHistory(repo, box);
    }
    return box;
  }

  //: Drop the cache and refresh the on-screen list in place. Only for the
  //: selected repo: any other repo has no list on screen, and dropping its
  //: cache alone means the next selection refetches.
  function refreshHistory(repo) {
    if (!repo) return;
    delete promptHistory[repo];
    if (repo === selectedProject && histBox && document.body.contains(histBox)) {
      fetchHistory(repo, histBox);
    }
  }

  function findProject(repo) {
    for (var i = 0; i < projects.length; i++) {
      if (projects[i].repo === repo) return projects[i];
    }
    return null;
  }

  //: One switch for "a run is in flight". `$("go")` was already the page's
  //: single-flight latch; the project controls join it here rather than each
  //: caller remembering three nodes to toggle.
  function busyRun(on) {
    runBusy = on;
    $("go").disabled = on;
    syncBar();
  }

  //: The selector and its button are rebuilt with every form redraw; this
  //: refreshes the one currently on screen in place, so a /projects response
  //: or a busy-state flip lands without redrawing the form under a typist.
  function syncBar() {
    if (!projBar.select) return;
    var pick = projBar.select;
    pick.textContent = "";
    if (!selectedProject) {
      var blank = el("option", null,
                     projects.length ? "— pick a project —" : "— no projects yet —");
      blank.value = "";
      pick.appendChild(blank);
    }
    projects.forEach(function (p) {
      var o = el("option", null, p.repo);
      o.value = p.repo;
      pick.appendChild(o);
    });
    pick.value = selectedProject;
    var tip = "a run is in flight — watch it below, or stop it, before switching projects";
    pick.disabled = runBusy;
    pick.title = runBusy ? tip : "";
    projBar.button.disabled = runBusy;
    projBar.button.title = runBusy
      ? tip : "start a fresh project from a plain-language description";
  }

  function projectBar() {
    var wrap = el("div", "projbar");
    wrap.appendChild(el("span", "modehint", "project"));
    var pick = el("select");
    pick.onchange = function () { if (pick.value) selectProject(pick.value); };
    var fresh = el("button", "ghost", "New project");
    fresh.type = "button";
    fresh.onclick = function (e) { e.preventDefault(); newProject(); };
    wrap.appendChild(pick);
    wrap.appendChild(fresh);
    projBar.select = pick;
    projBar.button = fresh;
    syncBar();
    return wrap;
  }

  //: Selecting a project loads its stored values into the same `typed` slot
  //: the operator's own typing lives in - the advanced form then simply shows
  //: them. The objective is deliberately NOT loaded: the project's past
  //: prompts (its founding requirement first among them) live in the history
  //: below the form, and the box is for the next prompt - so it opens empty.
  //: The checkboxes and cycle cap are left as they are: how a run is fired is
  //: a per-fire choice, not a fact about the project.
  function selectProject(repo) {
    var p = findProject(repo);
    if (!p) return;
    var kept = typed["swarm"] || {};
    //: `local` is forced off, not carried: a stored project is a GitHub
    //: project, and a leftover tick from an earlier local experiment would
    //: silently read the repo slug as a directory path. `public` is dropped
    //: for the same reason it is not rendered - existing repositories are
    //: untouched by it either way.
    typed["swarm"] = { objective: "", repo: p.repo, stack: p.stack,
                       verify: p.verify, local: "",
                       auto_merge: kept.auto_merge, no_goal_check: kept.no_goal_check,
                       max_cycles: kept.max_cycles };
    selectedProject = repo;
    //: The board follows the selection at once - but never wrestles the repo
    //: away from a run that is still writing its own progress.
    if (!runBusy) runRepo = "";
    wizardMode = "advanced";
    if (current && current.kind === "swarm") {
      drawForm();
      boardTick();
      followSelection();
    } else {
      syncBar();
    }
  }

  //: The run area follows the selection the way the board does. A finished
  //: run's card belonging to another project is removed - leaving it would
  //: read as a choice this click just unmade - and an immediate external poll
  //: brings the selected project's own last run back without the 15s wait.
  //: A live run is never touched: it is a fact, not a preference.
  function followSelection() {
    if (runBusy) return;
    var box = panel().runBox;
    if (box.childNodes.length) {
      var ours = extView && box.contains(extView.root);
      var about = ours ? extView.repo : ownRepo;
      if (about !== selectedProject) {
        box.textContent = "";
        extView = null;
      }
    }
    extTick();
  }

  //: "New project" is the intake questionnaire: the front door for a project
  //: that does not exist yet, with a blank swarm slate behind it for the
  //: proposal to land on.
  function newProject() {
    typed["swarm"] = { objective: "", repo: "", stack: "", verify: "" };
    selectedProject = "";
    if (!runBusy) runRepo = "";
    wizardMode = "describe";
    drawForm();
    boardTick();
    followSelection();
  }

  //: A run the page adopted owns the run area; the selector still lands on
  //: the run's repo when that repo is a project this console knows. Guarded
  //: against the run the operator just fired: an equal selection returns
  //: without touching the form, so nothing typed for *this* run is reloaded
  //: from the store mid-flight.
  function reflectRun(repo) {
    if (!repo || repo === selectedProject) return;
    if (findProject(repo)) selectProject(repo);
  }

  //: Re-fetched after every successful start: the backend has just upserted
  //: the project, and a brand-new repository must appear in the selector,
  //: selected, without a reload.
  function loadProjects(prefer) {
    api("/projects").then(function (res) {
      if (!res.ok || !res.body || !res.body.projects) return;
      projects = res.body.projects;
      //: The moment a described project lands in the store it stops being a
      //: draft: select it properly, so the repo locks, the prompt box empties
      //: (the submitted brief is the history's founding entry now) and the
      //: history renders. An unchanged selection only refreshes the bar.
      if (prefer && findProject(prefer) && prefer !== selectedProject) {
        selectProject(prefer);
      } else {
        syncBar();
      }
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
      boardHint("local run — the board derives its columns from GitHub, and a local run "
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
    //: The resolver's own sentence for why this card sits where it does -
    //: textContent-safe prose, surfaced as a hover rather than more pixels.
    if (c.because) t.title = c.because;
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
    //: From apiary's own store, and only when there is one: a renewed budget is
    //: why a task with more attempts than the cap is still running.
    if (c.renewals) head.appendChild(el("span", "pill", "budget renewed " + c.renewals + "x"));
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
    if ((b.needs_human || []).length) {
      var strip = el("div", "failedstrip");
      strip.appendChild(el("h3", null, "Needs a human · " + b.needs_human.length));
      b.needs_human.forEach(function (c) { strip.appendChild(ticket(c)); });
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

  //: `into` is where the view is drawn, and it defaults to the swarm tab's run
  //: area. #130 passes one: a run chained off Start building is followed from
  //: the planner tab, and a second copy of this view written there would be a
  //: second answer to "what is this run doing" - the one that goes stale.
  function swarmView(job, into) {
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

    var box = into || panel().runBox;
    box.textContent = "";
    //: Whoever draws last owns the run. Claiming here rather than at each call
    //: site is what makes that true by construction: every path that shows a
    //: run - swarmFire, adoption, a build's chained run - goes through this
    //: function, and none of them can forget to.
    clearTimeout(runTimer);
    runTimer = null;
    var generation = ++runGeneration;
    //: Only when this view is taking over the swarm tab's run area. Nulling it
    //: while drawing into a build card would drop the external view's handle
    //: on a card still sitting in `panel().runBox`, which then stops updating.
    if (!into) extView = null;
    box.appendChild(card("the run", head));
    box.appendChild(card("log — the run's own output, live", body));

    var view = {
      next: 0,
      generation: generation,
      absorb: function (j) {
        var p = j.progress || {};
        //: `state` cannot say this on its own: a run that met its objective
        //: and a run that ran out of cycles both end "done" with exit 0, and
        //: the second has unfinished work sitting in the repository. An empty
        //: `outcome` is a run still going, or a backend older than #130.
        var ended = OUTCOMES[p.outcome] || j.state;
        state.textContent = ended + (j.state === "failed" && j.returncode !== null
                                     ? " · exit " + j.returncode : "");
        //: Neutral for a cap, because unfinished work is not a failure and
        //: not a success. `exhausted` falls through to "ok" deliberately: the
        //: gate was off, so the run did exactly the work that was planned and
        //: there is nothing left it was asked for.
        state.className = "pill " + (j.state === "running" ? ""
                                     : p.outcome === "capped" ? ""
                                     : j.state === "done" ? "ok" : "bad");
        elapsed.textContent = j.elapsed_s + "s";
        elapsed.style.display = "";
        if (p.run_id) ownRunIds[p.run_id] = 1;   // the external view must skip it
        //: While the run lives, the board follows its repository and the
        //: selector lands on it when it is a known project. A run that has
        //: already ended claims neither - its card is a record, not a choice
        //: made for the operator.
        if (p.repo) ownRepo = p.repo;
        if (p.repo && j.state === "running") {
          runRepo = p.repo;
          reflectRun(p.repo);
        }
        if (p.cycle !== null && p.cycle !== undefined) {
          cycle.textContent = "cycle " + p.cycle;
          cycle.style.display = "";
        }
        if ((p.prs || []).length) {
          prs.textContent = (p.prs.length === 1 ? "PR #" : "PRs #") + p.prs.join(", #");
          prs.style.display = "";
        }
        //: The pill says which ending; this says what the run said about it.
        //: Repeating "objective met" in both - which is what this did before
        //: the pill could name an outcome - spends the one line that could
        //: have carried the verdict's reason.
        note.textContent = p.note || "";
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
          busyRun(false);
          //: The run just wrote (or failed to write) its summary: refetch so
          //: the newest history entry's pill reads finished truthfully.
          refreshHistory(selectedProject);
          //: ...and the same summary carries the ending (#134). Drawn from the
          //: artifacts rather than from `p` beside it, because that is the
          //: account that is still there after a reload - and the one a run
          //: this console did not start has too.
          drawOutcome(this, p.run_id, box);
        }
      }
    };
    view.absorb(job);
    return view;
  }

  function pollSwarm(id, view) {
    api("/swarm/status?id=" + encodeURIComponent(id) + "&since=" + view.next)
      .then(function (res) {
        //: A newer view took the run area while this request was in flight.
        //: Absorbing would write into a node that is no longer on the page,
        //: and rescheduling would put a second chain back on the one timer -
        //: which is the bug the generation exists to close, arriving through
        //: the half `clearTimeout` cannot reach.
        if (view.generation !== runGeneration) return;
        if (!res.ok) { busyRun(false); return; }
        view.absorb(res.body);
        if (res.body.state === "running") {
          runTimer = setTimeout(function () { pollSwarm(id, view); }, 1000);
        }
      });
  }

  function swarmFire() {
    busyRun(true);
    clearTimeout(runTimer);
    var vals = values();
    api("/swarm/start", { values: vals }).then(function (r) {
      if (!r.ok) {
        busyRun(false);
        var box = panel().runBox;
        box.textContent = "";
        extView = null;
        box.appendChild(errorCard({ type: "refused", message: r.body.error, fix: r.body.fix }));
        return;
      }
      pollSwarm(r.body.id, swarmView(r.body));
      //: The backend has just written this project down; re-fetching is what
      //: makes a brand-new repository appear in the selector, selected. A
      //: local run records a directory path, which is not a project.
      var fired = (vals.repo || "").trim();
      ownRepo = fired;   // the run view is about this repo from its first tick
      if (vals.local !== "1") loadProjects(fired);
      //: The prompt just fired becomes history the moment the run records its
      //: run.json, which happens within moments of the spawn - so one refetch
      //: ~3s in shows it without waiting for the run to end. The run-end
      //: refetch in `absorb` then flips its pill to finished.
      setTimeout(function () { refreshHistory(fired); }, 3000);
      //: The prompt was submitted: an emptied box says so, and the refetch
      //: above is where its text reappears - as history. Cleared in place,
      //: not by redraw, so the run view below is untouched.
      if (fired && fired === selectedProject) {
        var slot = typed["swarm"] || {};
        slot.objective = "";
        typed["swarm"] = slot;
        var promptNode = document.querySelector('[name="objective"]');
        if (promptNode) promptNode.value = "";
      }
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
    buildSite = res.body.build || null;
    //: Never clobber a tab the operator has already chosen. This resolves once
    //: at load, but a slow response and a fast click put the selection back on
    //: the first site while leaving the other one's form on screen.
    current = current || sites[0];
    //: With the strip hidden the swarm view is the only view; `current` must
    //: never rest on a site whose tab has no button to leave it by. sites[0]
    //: is the swarm whenever hideTabs() answers yes - it was just unshifted.
    if (hideTabs()) current = sites[0];
    var m = res.body.models;
    $("models").textContent = "worker " + m.worker + "  ·  orchestrator " + m.orchestrator
                            + "  ·  " + m.base_url;
    $("hint").textContent = "Captures are written for every call, successful or not.";
    drawTabs();
    drawForm();
    if (current.kind === "swarm") swarmShow();
    //: The remembered projects, most recently active first - listed, never
    //: presumed: the selector opens on "pick a project" and the operator
    //: chooses. The one exception is a run already in flight that the page
    //: adopted: selecting its repo, when the store knows it (either here or
    //: in `reflectRun`, whichever response arrives second), reflects what is
    //: actually running - reality, not a default.
    if (res.body.swarm) {
      api("/projects").then(function (pr) {
        if (!pr.ok || !pr.body || !pr.body.projects) return;
        projects = pr.body.projects;
        if (findProject(runRepo)) selectProject(runRepo);
        else syncBar();
      });
    }
  });
})();
