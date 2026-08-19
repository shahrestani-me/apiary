"""The capability contract: what a valid tracker block is, and what it refuses.

The ticket's "done when" has four clauses and this file is three of them (the
`doctor` check is the fourth, in `test_doctor.py`).

**A malformed block fails at load, with the field named.** Table-driven, one
case per way a block can be wrong, and every case asserts on the *message*
rather than only on the exception type. That is not thoroughness for its own
sake: the entire argument for validating here rather than letting the first
cycle raise is that a `KeyError` an hour into a run does not say which line of
which file to edit, so a refusal that also does not say it has moved the
failure without improving it. `test_every_refusal_names_the_field_and_a_fix`
holds that over the whole table at once.

**`args` is opaque, and stays opaque.** Three tests, and they are the ones to
be most careful about deleting. #143's Jira constraint is that intake args must
never be narrowed to a schema validated against GitHub's and Linear's parameter
names, because Jira's intake is a JQL string and its create needs `cloudId`,
`projectKey` and `issueTypeName` - constants with no counterpart on either
priority tracker. So the tests assert that nonsense arguments, unknown keys and
a free-text query *all survive validation and arrive at the call unchanged*.
A test suite that demanded well-formed arguments would be the mechanism by
which that constraint got broken.

**Jira must remain reachable from here.** `test_a_jira_shaped_block_validates`
is written entirely out of #143's "must not be precluded" table - JQL intake,
`commentBody`, Basic auth, `issueIdOrKey` - against no Jira profile and no Jira
code. It is a canary: whatever narrows this contract to the two trackers that
exist will fail it, which is a cheaper way to find out than adding Jira.

The two priority profiles are checked against the tool shapes #143 recorded
from the live servers, including the two that would otherwise be forgotten:
GitHub's `issue_write` needs `method: create` pinned, and its comment takes
`issue_number` where Linear's takes `issueId`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from swarm.config import TRACKER_CONFIG_ENV, Settings
from swarm.mcp.client import STDIO_SCHEME, McpClient
from swarm.mcp.contract import (
    CAPABILITIES,
    COMMENT,
    CREATE,
    INTAKE,
    PROFILES,
    Auth,
    ContractError,
    TrackerContract,
    client_for,
    load_tracker,
    main,
    parse_tracker,
)

GITHUB_BLOCK = """
tracker:
  mcp: github
  args: { owner: shahrestani-me, repo: apiary }
  intake: { args: { labels: [agent-ready] } }
"""

LINEAR_BLOCK = """
tracker:
  mcp: linear
  args: { teamId: TEAM-1 }
"""


def contract(text: str, *, source: str = "tracker.yaml") -> TrackerContract:
    return parse_tracker(text, source=source)


# --------------------------------------------------------------------------
# The two priority profiles
# --------------------------------------------------------------------------


def test_the_github_profile_can_make_the_calls_the_spike_recorded():
    """#143's whole finding, as an assertion.

    ADR 0001's `{ tool, args }` could not call this server: `issue_write` is
    create and update behind a discriminator, so the tool name alone is not a
    call, and the comment is addressed by `issue_number` rather than by
    whatever apiary calls a task ref.
    """
    block = contract(GITHUB_BLOCK)

    assert block.create.tool == "issue_write"
    assert block.arguments(CREATE, {"title": "T", "body": "B"}) == {
        "method": "create",
        "owner": "shahrestani-me",
        "repo": "apiary",
        "title": "T",
        "body": "B",
    }
    assert block.arguments(COMMENT, {"ref": 417, "body": "the PR is up"}) == {
        "owner": "shahrestani-me",
        "repo": "apiary",
        "issue_number": 417,
        "body": "the PR is up",
    }
    assert block.arguments(INTAKE) == {
        "owner": "shahrestani-me",
        "repo": "apiary",
        "labels": ["agent-ready"],
    }
    assert block.task_ref({"number": 417, "title": "something"}) == "417"


def test_the_linear_profile_renames_the_body_and_the_ref():
    """The residual divergence, in the one place it is allowed to live.

    `description` on create and `issueId` on comment - two entries in a field
    map, against a GitHub profile that needs neither. Nothing else about the
    two blocks differs in kind.
    """
    block = contract(LINEAR_BLOCK)

    assert block.create.tool == "create_issue"
    assert block.arguments(CREATE, {"title": "T", "body": "B"}) == {
        "teamId": "TEAM-1",
        "title": "T",
        "description": "B",
    }
    assert block.arguments(COMMENT, {"ref": "abc-uuid", "body": "hi"}) == {
        "teamId": "TEAM-1",
        "issueId": "abc-uuid",
        "body": "hi",
    }
    assert block.task_ref({"id": "abc-uuid", "identifier": "ENG-123"}) == "ENG-123"


def test_the_comment_body_is_never_hard_coded():
    """The trap #143 names: both priority trackers spell it `body`.

    A design that read the coincidence as a fact would work on GitHub, work on
    Linear, and be unfixable on Jira, whose comment field is `commentBody`. The
    assertion is that the mapping is *reachable* on the comment capability, not
    only on create.
    """
    block = contract(
        """
        mcp: atlassian
        endpoint: https://api.atlassian.com/v1/mcp
        auth: { value_env: APIARY_JIRA_TOKEN, scheme: basic }
        intake:  { tool: searchJiraIssuesUsingJql, ref: key }
        comment: { tool: addCommentToJiraIssue, fields: { body: commentBody, ref: issueIdOrKey } }
        create:  { tool: createJiraIssue, fields: { title: summary } }
        """
    )
    assert block.arguments(COMMENT, {"ref": "PROJ-12", "body": "up"}) == {
        "issueIdOrKey": "PROJ-12",
        "commentBody": "up",
    }


def test_both_shipped_profiles_are_valid_on_their_own():
    """A profile a customer cannot use without editing is documentation, not a default.

    Each needs exactly the constants only that organization knows - GitHub's
    `owner`/`repo`, Linear's `teamId` - and nothing else.
    """
    assert set(PROFILES) == {"github", "linear"}
    for name, extra in (("github", "{ owner: o, repo: r }"), ("linear", "{ teamId: T }")):
        block = contract(f"mcp: {name}\nargs: {extra}")
        assert set(block.capabilities) == set(CAPABILITIES)
        assert all(block.capability(capability).tool for capability in CAPABILITIES)
        assert block.ref_rule(INTAKE)


def test_the_github_profile_is_a_local_server_reusing_the_existing_token():
    """#143's recommendation 3, and the reason it is not a header.

    The remote GitHub MCP server advertises the classic OAuth scopes that
    `security.assert_scoped_token` refuses outright, so the profile is the local
    stdio binary - which reads its own credential from its own environment.
    That is a different delivery, not a different token: `GITHUB_TOKEN` is the
    one apiary already holds.
    """
    block = contract(GITHUB_BLOCK)
    assert block.is_stdio
    assert block.endpoint == f"{STDIO_SCHEME}github-mcp-server"
    assert block.auth.value_env == "GITHUB_TOKEN"
    assert block.auth.server_env == "GITHUB_PERSONAL_ACCESS_TOKEN"
    assert "github.com/settings/personal-access-tokens" in block.auth.mint


# --------------------------------------------------------------------------
# `args` is opaque - the Jira constraint
# --------------------------------------------------------------------------


def test_intake_args_are_passed_through_whatever_they_are():
    """The single most important constraint on #143's page.

    Every one of these is meaningless to apiary and none of them is checked
    against anything. A validator that understood `labels` well enough to
    reject `jql` is a validator Jira cannot be added behind.
    """
    block = contract(
        """
        mcp: acme
        endpoint: https://mcp.acme.test/mcp
        auth: { value_env: ACME_TOKEN }
        intake:
          tool: search
          ref: key
          args:
            jql: "project = ENG AND labels = agent-ready ORDER BY created"
            cloudId: 8f3e-not-a-uuid
            maxResults: 50
            expand: [names, renderedFields]
            weird: { nested: { deeply: true } }
        comment: { tool: c }
        create:  { tool: k }
        """
    )
    assert block.arguments(INTAKE) == {
        "jql": "project = ENG AND labels = agent-ready ORDER BY created",
        "cloudId": "8f3e-not-a-uuid",
        "maxResults": 50,
        "expand": ["names", "renderedFields"],
        "weird": {"nested": {"deeply": True}},
    }


def test_a_jira_shaped_block_validates_against_no_jira_code():
    """The canary for #143's "must not be precluded" table.

    JQL intake, `cloudId` / `projectKey` / `issueTypeName` as create constants,
    `summary`/`description` and `commentBody` as fields, Basic auth, and
    `issueIdOrKey` as the ref. Six requirements, no Jira profile, no branch
    anywhere that knows Jira exists.
    """
    block = contract(
        """
        mcp: atlassian
        endpoint: https://api.atlassian.com/v1/mcp
        auth: { value_env: APIARY_JIRA_TOKEN, scheme: basic }
        args: { cloudId: 11111111-2222-3333 }
        intake:
          tool: searchJiraIssuesUsingJql
          ref: key
          args: { jql: "labels = agent-ready" }
        comment:
          tool: addCommentToJiraIssue
          fields: { ref: issueIdOrKey, body: commentBody }
        create:
          tool: createJiraIssue
          args: { projectKey: ENG, issueTypeName: Task }
          fields: { title: summary, body: description }
        """
    )
    assert block.auth.scheme == "basic"
    assert block.arguments(CREATE, {"title": "T", "body": "B"}) == {
        "cloudId": "11111111-2222-3333",
        "projectKey": "ENG",
        "issueTypeName": "Task",
        "summary": "T",
        "description": "B",
    }
    assert block.task_ref({"key": "PROJ-12"}, CREATE) == "PROJ-12"
    assert Auth(value_env="X", scheme="basic").header_value("me@x:tok").startswith("Basic ")


def test_a_capability_can_pin_a_constant_the_other_trackers_have_no_word_for():
    """`method: create` and `issueTypeName` are the same mechanism.

    Which is the point of #143's finding that the scope constants need no new
    field: `args` was always a static dict, and the only thing ADR 0001 got
    wrong about it was calling it the customer's query language.
    """
    assert contract(GITHUB_BLOCK).create.args["method"] == "create"


# --------------------------------------------------------------------------
# Refusals - one case per way a block can be wrong
# --------------------------------------------------------------------------

#: (id, block, what the message must name)
BAD_BLOCKS: list[tuple[str, str, str]] = [
    ("not-a-mapping", "- one\n- two", "must be a mapping"),
    ("no-mcp", "intake: { tool: t }", "tracker.mcp is missing"),
    ("unknown-top-level-key", "mcp: github\ncomments: { tool: t }", "comments"),
    ("typo-suggests-the-real-key", "mcp: github\ncomments: { tool: t }", "did you mean comment"),
    (
        "no-endpoint",
        "mcp: acme\nauth: { value_env: T }\nintake: { tool: t, ref: id }\n"
        "comment: { tool: c }\ncreate: { tool: k }",
        "no endpoint and no command",
    ),
    ("unknown-profile-is-named", "mcp: gtihub", "matches no built-in profile"),
    (
        "endpoint-is-not-a-url",
        "mcp: acme\nendpoint: mcp.acme.test\nauth: { value_env: T }\n"
        "intake: { tool: t, ref: id }\ncomment: { tool: c }\ncreate: { tool: k }",
        "not a URL",
    ),
    (
        "endpoint-and-command-both",
        "mcp: acme\nendpoint: https://mcp.acme.test/mcp\ncommand: [acme-mcp]",
        "two alternative ways",
    ),
    ("command-is-not-argv", "mcp: acme\ncommand: acme-mcp", "as a list"),
    ("missing-capability", "mcp: acme\nendpoint: https://mcp.acme.test/mcp\n"
     "auth: { value_env: T }\nintake: { tool: t, ref: id }\ncomment: { tool: c }",
     "tracker.create is missing"),
    ("capability-is-not-a-mapping", "mcp: github\ncomment: list_issues", "must be a mapping"),
    ("capability-has-no-tool", "mcp: acme\nendpoint: https://x.test/mcp\nauth: { value_env: T }\n"
     "intake: { ref: id }\ncomment: { tool: c }\ncreate: { tool: k }",
     "tracker.intake.tool is missing"),
    ("unknown-capability-key", "mcp: github\ncreate: { tool: t, feilds: { body: b } }", "feilds"),
    ("args-not-a-mapping", "mcp: github\nintake: { args: [1, 2] }", "tracker.intake.args must be"),
    ("shared-args-not-a-mapping", "mcp: github\nargs: [1, 2]", "tracker.args must be"),
    ("fields-not-a-mapping", "mcp: github\ncreate: { fields: 3 }", "tracker.create.fields must"),
    (
        "fields-left-hand-side-is-closed",
        "mcp: github\ncreate: { fields: { bdoy: description } }",
        "apiary's vocabulary and is closed",
    ),
    (
        "fields-right-hand-side-must-name-something",
        "mcp: github\ncreate: { fields: { body: 7 } }",
        "must name a field on the right-hand side",
    ),
    ("intake-has-no-ref", "mcp: linear\nintake: { ref: null }", "tracker.intake.ref is missing"),
    ("ref-is-not-a-name", "mcp: linear\nintake: { ref: 7 }", "tracker.intake.ref must name"),
    ("auth-not-a-mapping", "mcp: linear\nauth: bearer", "tracker.auth must be a mapping"),
    ("auth-unknown-key", "mcp: linear\nauth: { value_env: T, bearer: yes }", "bearer"),
    (
        "auth-has-no-variable",
        "mcp: acme\nendpoint: https://x.test/mcp\nintake: { tool: t, ref: id }\n"
        "comment: { tool: c }\ncreate: { tool: k }",
        "tracker.auth.value_env is missing",
    ),
    ("auth-unknown-scheme", "mcp: linear\nauth: { scheme: digest }", "must be one of"),
    (
        "stdio-without-a-server-variable",
        "mcp: acme\ncommand: [acme-mcp, stdio]\nauth: { value_env: T }\n"
        "intake: { tool: t, ref: id }\ncomment: { tool: c }\ncreate: { tool: k }",
        "reads its own credential from its own environment",
    ),
    (
        "semantic-intake-on-github",
        "mcp: github\nintake: { tool: search_issues, ref: number }",
        "semantic matching",
    ),
    ("not-yaml", "mcp: github\n  intake: [", "not valid YAML"),
]


@pytest.mark.parametrize("case,block,expected", BAD_BLOCKS, ids=[c[0] for c in BAD_BLOCKS])
def test_a_malformed_block_is_refused_by_name(case: str, block: str, expected: str):
    with pytest.raises(ContractError) as raised:
        contract(block)
    assert expected in str(raised.value), str(raised.value)


def test_every_refusal_names_the_file_and_something_to_do():
    """The house rule from `doctor.Check`, held over config validation too.

    A refusal that does not name the file cannot be acted on from a log, and
    one that does not show the shape it wanted sends the reader to the source.
    """
    for case, block, _ in BAD_BLOCKS:
        with pytest.raises(ContractError) as raised:
            contract(block)
        message = str(raised.value)
        assert message.startswith("tracker.yaml:"), f"{case}: {message}"
        assert len(message) > 60, f"{case}: too terse to act on: {message}"


def test_jql_is_not_refused_for_looking_like_a_search():
    """The inadmissible-intake rule is about one tool on one server.

    GitHub's `search_issues` is refused because that server's implementation is
    a semantic match. Jira's `searchJiraIssuesUsingJql` is a search and is
    perfectly deterministic, and a rule written against the word "search" would
    have taken it out with no way to say so.
    """
    block = contract(
        """
        mcp: atlassian
        endpoint: https://api.atlassian.com/v1/mcp
        auth: { value_env: T }
        intake:  { tool: searchJiraIssuesUsingJql, ref: key }
        comment: { tool: c }
        create:  { tool: k }
        """
    )
    assert block.intake.tool == "searchJiraIssuesUsingJql"


# --------------------------------------------------------------------------
# The ref rule
# --------------------------------------------------------------------------


def test_a_ref_that_cannot_be_a_branch_name_is_refused():
    """ADR 0001 derives execution state from `apiary/<ref>-attempt-N`.

    So a ref is not a display string. Pointing the rule at a title validates
    fine - the field exists, and no schema here knows what a title is - and the
    damage would be a branch that parses back as a different task, which is why
    the refusal is at the moment of extraction rather than at load.
    """
    block = contract("mcp: linear\nargs: { teamId: T }\nintake: { ref: title }")
    with pytest.raises(ContractError, match="branch name"):
        block.task_ref({"title": "add a thing / with a slash"})


def test_a_missing_ref_field_names_what_was_expected():
    block = contract(GITHUB_BLOCK)
    with pytest.raises(ContractError, match="has no such field"):
        block.task_ref({"id": 1, "title": "t"})


def test_a_capability_without_its_own_rule_inherits_intakes():
    """The ref rule is a fact about the server, not about the call."""
    block = contract(GITHUB_BLOCK)
    assert block.ref_rule(CREATE) == "number"
    assert block.task_ref({"number": 9}, CREATE) == "9"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_no_tracker_file_is_not_an_error(tmp_path: Path):
    """apiary runs on the label control plane until #152.

    An installation with no contract is a normal one, and a preflight that
    failed on it would be a preflight everybody learns to ignore.
    """
    settings = Settings(tracker_config=str(tmp_path / "absent.yaml"))
    assert load_tracker(settings=settings, env={}) is None


def test_a_named_file_that_is_absent_is_an_error(tmp_path: Path):
    """Naming a file is how an operator says they meant to configure a tracker.

    The failure this separates out is the worst one available: a typo'd path
    that silently means "no tracker", producing a run that reads nothing and
    reports nothing, with every check passing.
    """
    missing = tmp_path / "typo.yaml"
    with pytest.raises(ContractError, match="no such file"):
        load_tracker(settings=Settings(), env={TRACKER_CONFIG_ENV: str(missing)})


def test_a_file_may_be_the_block_or_carry_it_under_tracker(tmp_path: Path):
    wrapped = tmp_path / "wrapped.yaml"
    wrapped.write_text(GITHUB_BLOCK, encoding="utf-8")
    bare = tmp_path / "bare.yaml"
    bare.write_text(GITHUB_BLOCK.replace("tracker:\n", "").replace("  ", ""), encoding="utf-8")

    assert load_tracker(wrapped, env={}).mcp == "github"
    assert load_tracker(bare, env={}).mcp == "github"


def test_the_settings_path_is_where_the_default_comes_from(tmp_path: Path):
    written = tmp_path / "tracker.yaml"
    written.write_text(LINEAR_BLOCK, encoding="utf-8")
    block = load_tracker(settings=Settings(tracker_config=str(written)), env={})
    assert block is not None and block.mcp == "linear"
    assert block.source == str(written)


# --------------------------------------------------------------------------
# Reaching the server it names
# --------------------------------------------------------------------------


def test_a_remote_contract_sends_the_credential_as_the_scheme_says():
    block = contract(LINEAR_BLOCK)
    client = client_for(block, env={"APIARY_LINEAR_TOKEN": "lin_api_x"})
    assert isinstance(client, McpClient)
    assert client._headers["Authorization"] == "Bearer lin_api_x"
    #: So that `McpAuthError` names the variable this tracker's token lives in
    #: rather than the default one.
    assert client.token_env == "APIARY_LINEAR_TOKEN"


def test_a_stdio_contract_hands_the_credential_to_the_subprocess():
    """And hands it nothing else.

    `over_stdio` passes through a named list, so a credential added to the
    orchestrator later does not silently become a third-party binary's.
    """
    block = contract(GITHUB_BLOCK)
    client = client_for(
        block, env={"GITHUB_TOKEN": "github_pat_x", "APIARY_PROVISION_TOKEN": "secret"}
    )
    child = client._transport.env
    assert child["GITHUB_PERSONAL_ACCESS_TOKEN"] == "github_pat_x"
    assert "APIARY_PROVISION_TOKEN" not in child
    assert "Authorization" not in client._headers


def test_a_missing_credential_says_which_variable_and_how_to_mint_one():
    block = contract(LINEAR_BLOCK)
    with pytest.raises(ContractError) as raised:
        client_for(block, env={})
    message = str(raised.value)
    assert "APIARY_LINEAR_TOKEN" in message
    assert "linear.app/settings/api" in message


def test_doctor_can_build_a_client_without_a_credential():
    """The reason `require_credential` exists.

    A probe that refused to exist without a token could not tell "nothing is
    listening" from "you have not exported one yet", and those have different
    fixes.
    """
    client = client_for(contract(LINEAR_BLOCK), env={}, require_credential=False)
    assert "Authorization" not in client._headers


# --------------------------------------------------------------------------
# The command every refusal names
# --------------------------------------------------------------------------


def test_the_validate_command_prints_the_merged_block(tmp_path: Path, capsys):
    """`python -m swarm.mcp.contract` shows what the profile filled in.

    Printing the *merged* contract rather than echoing the file is the whole
    value: a four-line block that leans on a profile is exactly the one whose
    resolved tool names nobody can see.
    """
    path = tmp_path / "tracker.yaml"
    path.write_text(GITHUB_BLOCK, encoding="utf-8")

    assert main([str(path)]) == 0
    printed = capsys.readouterr().out
    assert "issue_write" in printed and "method" in printed
    assert "GITHUB_PERSONAL_ACCESS_TOKEN" in printed


def test_the_validate_command_reports_a_bad_block_and_exits_one(tmp_path: Path, capsys):
    path = tmp_path / "tracker.yaml"
    path.write_text("mcp: github\ncomments: { tool: t }\n", encoding="utf-8")

    assert main([str(path)]) == 1
    assert "did you mean comment" in capsys.readouterr().err
