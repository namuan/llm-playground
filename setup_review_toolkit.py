#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = [
#   "pandas",
#   "persistent-cache@git+https://github.com/namuan/persistent-cache"
# ]
# ///
"""
Bootstrap a repo-local AI review toolkit for OpenCode, Pi, or both.

Run this from the root of a Git repository.

Shared scaffolding:
- .review/            human-owned review policy/context/knowledge
- AGENTS.md           created only when one does not already exist

OpenCode adapter:
- .opencode/agents/
- .opencode/commands/

Pi adapter:
- .pi/agents/         custom agents for pi-subagents
- .pi/skills/         /skill:review, /skill:review-brief, etc.
- .pi/settings.json   minimal project config marker

Pi orchestration expects the `pi-subagents` package. Scaffold generation does not
require network access. Use --install-pi-deps to run:

    pi install npm:pi-subagents

Usage:
./setup_review_toolkit.py -h
./setup_review_toolkit.py --harness opencode
./setup_review_toolkit.py --harness pi
./setup_review_toolkit.py --harness both
./setup_review_toolkit.py --harness pi --install-pi-deps
./setup_review_toolkit.py --dry-run
./setup_review_toolkit.py --force
./setup_review_toolkit.py -v
./setup_review_toolkit.py -vv
"""
import json
import logging
import shutil
import subprocess
import textwrap
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path


TOOLKIT_VERSION = "0.2.0"
VALID_HARNESSES = ("opencode", "pi", "both")


def setup_logging(verbosity):
    logging_level = logging.WARNING
    if verbosity == 1:
        logging_level = logging.INFO
    elif verbosity >= 2:
        logging_level = logging.DEBUG

    logging.basicConfig(
        handlers=[logging.StreamHandler()],
        format="%(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging_level,
    )
    logging.captureWarnings(capture=True)


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbose",
        help="Increase verbosity of logging output",
    )
    parser.add_argument(
        "--harness",
        choices=VALID_HARNESSES,
        default="both",
        help="Harness adapter(s) to generate. Default: both",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite toolkit files that already exist. AGENTS.md is never overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without writing files or installing dependencies.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to scaffold. Defaults to current directory.",
    )
    parser.add_argument(
        "--install-pi-deps",
        action="store_true",
        help="If Pi is selected, install pi-subagents using `pi install npm:pi-subagents`.",
    )
    return parser.parse_args()


def run_command(command, *, cwd=None, check=True):
    logging.debug("Running command: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or f"Command failed: {' '.join(command)}"
        )
    return result


def run_git(root, *args):
    result = run_command(["git", "-C", str(root), *args])
    return result.stdout.strip()


def validate_repo(root):
    root = root.resolve()
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Repository path does not exist or is not a directory: {root}")

    try:
        toplevel = Path(run_git(root, "rev-parse", "--show-toplevel")).resolve()
    except (RuntimeError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"{root} is not inside a Git repository. Run this script from a project repo."
        ) from exc

    if root != toplevel:
        raise RuntimeError(
            "Run this from the Git repository root.\n"
            f"Current: {root}\n"
            f"Git root: {toplevel}\n"
            f"Or pass --root {toplevel}"
        )
    return root


def dedent(value):
    return textwrap.dedent(value).lstrip("\n").rstrip() + "\n"


REVIEW_CONTRACT = dedent("""
    ## Required output contract

    Return a concise review using this structure:

    ```yaml
    reviewer: <reviewer-name>
    verdict: pass | concern | block
    findings:
      - severity: high | medium | low
        file: <path>
        lines: <line or range if known>
        claim: <concrete problem>
        evidence: <why the code supports this claim>
        suggested_check: <specific verification or remediation>
    uncertainties:
      - <material fact you could not establish>
    confidence: <0.0-1.0>
    ```

    Rules:
    - Prefer zero findings over speculative findings.
    - Do not report style preferences unless they materially affect correctness or maintenance.
    - A `block` means concrete evidence indicates the change is unsafe or violates intent.
    - A `concern` means a material issue or uncertainty needs resolution.
    - If evidence is insufficient, put it in `uncertainties` rather than inventing a finding.
    - Do not edit files.
""")


def shared_files():
    files = {}

    files[".review/README.md"] = dedent("""
        # AI Review Toolkit

        Harness-independent review policy and durable engineering context.

        ## Workflow

        1. Independent specialist reviewers inspect the proposed change.
        2. A Review Adjudicator resolves overlap, disagreement, and uncertainty.
        3. Humans receive only unresolved engineering decisions.
        4. Knowledge-worthy changes are captured selectively.
        5. A weekly "Interesting Changes" session preserves deliberate team learning.

        ## Customize first

        - `policy.md`
        - `architecture.md`
        - `product-context.md`

        Keep these files short, explicit, and repository-specific. They are the shared
        intellectual layer; OpenCode and Pi are only adapters around it.
    """)

    files[".review/policy.md"] = dedent("""
        # Review Policy

        This file is human-owned. AI reviewers apply it but must not silently weaken it.

        ## Principles

        - Evidence beats confidence.
        - Unresolved material uncertainty requires escalation.
        - AI review is read-only; fixing happens after adjudication.
        - Review repository context, not only changed lines.
        - Reuse CI, static-analysis, test, and repository-history evidence.
        - Do not block on personal style preferences.

        ## Always escalate by default

        Customize this list.

        - authentication or authorization semantic changes
        - destructive or difficult-to-reverse data migrations
        - public API compatibility changes
        - cross-service ownership or architectural boundary changes
        - cryptography or secrets handling
        - material concurrency/distributed-consistency changes

        ## Usually low risk

        These still escalate if evidence reveals material uncertainty.

        - documentation-only changes
        - test-only changes with no production behavior change
        - deterministic generated artifacts
        - narrowly scoped mechanical refactors with strong verification

        ## Adjudication

        A change is `CLEAR` only when:
        - no unresolved blocking finding exists;
        - material reviewer disagreement has been resolved with evidence;
        - material uncertainty is resolved or explicitly non-blocking under policy;
        - applicable tests/checks pass.

        Otherwise return `HUMAN_INPUT_REQUIRED` or `BLOCKED`.
    """)

    files[".review/architecture.md"] = dedent("""
        # Architecture Invariants

        Replace these TODOs with rules experienced reviewers currently carry in their heads.

        ## System boundaries

        - TODO: Which modules/services own which responsibilities?
        - TODO: Which dependency directions are forbidden?
        - TODO: What are the persistence/data ownership boundaries?

        ## API and compatibility

        - TODO: Define backwards-compatibility expectations.
        - TODO: Define versioning/deprecation rules.

        ## Reliability and security

        - TODO: Define retry/idempotency expectations.
        - TODO: Define authorization boundaries.
        - TODO: Define sensitive-data handling rules.

        ## Established patterns

        - TODO: Record patterns that new code should normally follow.

        Keep this about invariants and boundaries, not general coding style.
    """)

    files[".review/product-context.md"] = dedent("""
        # Product Context

        Give the Product Intent Reviewer enough context to judge implementation behavior.

        ## Product purpose

        TODO: What user/customer problem does this repository solve?

        ## Important user journeys

        - TODO
        - TODO

        ## Product invariants

        Examples:
        - A user must never lose X.
        - Operation Y must remain backwards compatible.
        - State Z is visible only after condition Q.

        Replace these examples with real invariants.

        ## Current priorities / deliberate constraints

        - TODO
        - TODO
    """)

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "AI Review Result",
        "type": "object",
        "required": ["reviewer", "verdict", "findings", "uncertainties", "confidence"],
        "properties": {
            "reviewer": {"type": "string"},
            "verdict": {"enum": ["pass", "concern", "block"]},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["severity", "claim", "evidence", "suggested_check"],
                    "properties": {
                        "severity": {"enum": ["high", "medium", "low"]},
                        "file": {"type": "string"},
                        "lines": {"type": "string"},
                        "claim": {"type": "string"},
                        "evidence": {"type": "string"},
                        "suggested_check": {"type": "string"},
                    },
                },
            },
            "uncertainties": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    files[".review/schemas/review-result.schema.json"] = json.dumps(schema, indent=2) + "\n"

    files[".review/knowledge/README.md"] = dedent("""
        # Knowledge Records

        Store only durable, knowledge-worthy records here.

        Good candidates:
        - architecture decisions or changed boundaries
        - new invariants or established patterns
        - important incident/bug lessons
        - non-obvious trade-offs future engineers should understand

        Do not create a record for every change.
    """)

    files["AGENTS.md"] = dedent("""
        # Repository Agent Instructions

        This repository uses a harness-independent AI review toolkit.

        Human-owned context:
        - `.review/policy.md`
        - `.review/architecture.md`
        - `.review/product-context.md`

        Review methodology:
        - independent specialist reviewers
        - one evidence-based adjudicator
        - human escalation only for unresolved judgment or material uncertainty
        - selective knowledge capture
        - weekly "Interesting Changes" learning session

        Before significant implementation or review work, read the relevant `.review/` files.

        Do not weaken explicit review policy merely to make a change pass.
    """)

    return files


def reviewer_prompt(kind):
    prompts = {
        "correctness": dedent("""
            You are the Correctness Reviewer.

            Inspect the current proposed change in repository context. Do not limit yourself to
            changed lines: inspect relevant callers, callees, state transitions, error paths,
            retry behavior, concurrency assumptions, and nearby invariants.

            Mindset: assume the implementation may be wrong and look for a concrete execution
            path that demonstrates the failure.

            Focus on:
            - logic and semantic errors
            - edge/boundary cases
            - incorrect state transitions
            - error handling and partial failure
            - retries/idempotency
            - compatibility regressions
            - race/concurrency issues where applicable

            Use `.review/policy.md` and `.review/architecture.md` when relevant.
        """),
        "tests": dedent("""
            You are the Test Reviewer.

            Infer the behavioral claims made by the change, then determine whether tests and
            verification evidence actually prove those claims. Do not merely count tests.

            Focus on:
            - missing scenarios and boundary conditions
            - weak assertions
            - tests coupled to implementation rather than behavior
            - failure/retry/error-path coverage
            - regressions existing tests would not catch
            - changed production behavior without matching verification
            - flaky/non-deterministic assumptions

            You may run existing targeted tests when useful. Do not modify files.
        """),
        "architecture": dedent("""
            You are the Architecture & Maintainability Reviewer.

            Start with `.review/architecture.md` and `.review/policy.md`. Inspect surrounding
            repository structure and relevant history where needed.

            Focus on:
            - dependency direction and component/service boundaries
            - data/domain ownership
            - public API compatibility
            - misplaced responsibilities
            - abstractions that materially increase future change cost
            - duplication that creates divergence risk
            - repository-specific patterns and invariants

            Do not turn generic aesthetic preferences into findings.
        """),
        "product-intent": dedent("""
            You are the Product Intent Reviewer.

            Read `.review/product-context.md`, available change/PR description, linked/local
            requirements, relevant tests, and the implementation.

            Compare intended behavior with implemented behavior.

            Focus on:
            - requirements implemented differently from stated intent
            - acceptance criteria not covered
            - externally visible behavior introduced without explanation
            - product invariants weakened
            - error/empty/permission states affecting users
            - ambiguity requiring product or engineering judgment

            If requirements context is unavailable, report that as uncertainty.
        """),
    }
    return prompts[kind] + "\n" + REVIEW_CONTRACT


HISTORIAN_PROMPT = dedent("""
    You are the Change Historian.

    Investigate git history, git blame, commits, ADRs, repository docs, tests, and nearby
    comments to determine why unusual code, patterns, or boundaries exist.

    Supply historical evidence to the requesting reviewer/adjudicator. Do not decide whether
    the change passes and do not edit files.
""")


ADJUDICATOR_PROMPT = dedent("""
    You are the Review Adjudicator.

    You are NOT another generic code reviewer. Evaluate specialist evidence and decide what
    remains unresolved.

    Inputs should include:
    - correctness review
    - test review
    - architecture review
    - product-intent review
    - applicable CI/tool evidence

    Responsibilities:
    1. Deduplicate overlapping findings.
    2. Identify reviewer disagreement.
    3. Verify contested claims against repository evidence when feasible.
    4. Distinguish objective defects from decisions requiring human judgment.
    5. Treat material uncertainty as first-class evidence.
    6. Apply `.review/policy.md`.
    7. Never average confidence scores into a merge decision.
    8. Use repository history when it can resolve an architectural or behavioral question.

    Return one recommendation:
    - `CLEAR`
    - `HUMAN_INPUT_REQUIRED`
    - `BLOCKED`

    Then provide:

    ## Summary
    ## Cleared areas
    ## Blocking findings
    ## Human questions
    ## Residual uncertainty

    Each human question must explain:
    - why human judgment is required
    - the narrow decision to make
    - relevant files/lines
    - competing options/constraints when known

    Optimize for turning "review this PR" into one or two specific engineering decisions.
""")


KNOWLEDGE_PROMPT = dedent("""
    You are the Knowledge Curator.

    Preserve the knowledge-sharing value of code review without requiring every engineer to
    synchronously review every change.

    Classify a change as:
    - NONE
    - LOCAL
    - TEAM
    - ARCHITECTURAL

    Prefer NONE. Avoid documentation spam.

    A durable record should contain:
    - concise change summary
    - why it matters
    - decision/trade-off
    - new or changed invariant/pattern
    - who should know
    - relevant files/commits/PR references
    - follow-up questions, if any

    For weekly digests:
    - inspect the requested range of recent/merged changes
    - choose only 3-5 changes worth human discussion
    - prioritize architectural novelty, surprising bugs, changed patterns, reviewer
      disagreements, cross-component effects, or generalizable lessons
    - do not produce a chronological dump
""")


def opencode_agent_frontmatter(description, *, mode="subagent", edit="deny", bash="allow", task=None):
    lines = [
        "---",
        f"description: {description}",
        f"mode: {mode}",
        "temperature: 0.1",
        "permission:",
        f"  edit: {edit}",
        f"  bash: {bash}",
    ]
    if task is not None:
        lines.append(f"  task: {task}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def opencode_files():
    files = {}

    specs = {
        "reviewer-correctness": (
            "Adversarial read-only reviewer for correctness, edge cases, state, errors, retries, and regressions.",
            reviewer_prompt("correctness"),
        ),
        "reviewer-tests": (
            "Read-only reviewer that checks whether tests actually prove the behavioral claims of the change.",
            reviewer_prompt("tests"),
        ),
        "reviewer-architecture": (
            "Read-only reviewer for architecture boundaries, dependencies, maintainability, ownership, and repository invariants.",
            reviewer_prompt("architecture"),
        ),
        "reviewer-product-intent": (
            "Read-only reviewer comparing implementation with product intent, requirements, and product invariants.",
            reviewer_prompt("product-intent"),
        ),
        "change-historian": (
            "Read-only repository historian for git history, blame, previous decisions, and unusual code or boundaries.",
            HISTORIAN_PROMPT,
        ),
        "review-adjudicator": (
            "Read-only adjudicator that synthesizes evidence and narrows human review to specific questions.",
            ADJUDICATOR_PROMPT,
        ),
        "knowledge-curator": (
            "Curates durable engineering knowledge and selects high-value changes for team learning.",
            KNOWLEDGE_PROMPT,
        ),
    }

    for name, (description, prompt) in specs.items():
        files[f".opencode/agents/{name}.md"] = (
            opencode_agent_frontmatter(description) + prompt
        )

    conductor = opencode_agent_frontmatter(
        "Primary orchestration agent that runs independent specialist reviews and adjudicates the result.",
        mode="primary",
        edit="deny",
        bash="allow",
        task="allow",
    ) + dedent("""
        You are the Review Conductor.

        Orchestrate; do not add your own generic findings.

        1. Establish review scope, honoring an explicit base/ref.
        2. Read shared context from `.review/` and `AGENTS.md`.
        3. Gather diff/stat, changed files, tests/CI context, and available task/PR context.
        4. Invoke independently:
           - reviewer-correctness
           - reviewer-tests
           - reviewer-architecture
           - reviewer-product-intent
        5. Do not reveal one specialist's conclusions to another before review completes.
        6. Collect structured results.
        7. Invoke review-adjudicator with all results and shared evidence.
        8. Return adjudication first; include specialist detail only when useful.

        Review is read-only.
    """)
    files[".opencode/agents/review-conductor.md"] = conductor

    commands = {
        "review": """
            ---
            description: Run the independent AI review ensemble and adjudicate the current change
            agent: review-conductor
            ---

            Review the current proposed change.

            Optional base/scope:
            $ARGUMENTS

            Run specialist reviewers independently, then the adjudicator. Do not modify files.
        """,
        "review-brief": """
            ---
            description: Produce a narrow human-review brief for the current change
            agent: review-conductor
            ---

            Review the current proposed change.

            Optional base/scope:
            $ARGUMENTS

            Optimize the final response for a busy staff engineer:
            - recommendation
            - what AI cleared
            - smallest set of unresolved human decisions
            - exact relevant files/lines
            - residual uncertainty
        """,
        "review-fix": """
            ---
            description: Fix accepted review findings, verify them, and prepare for re-review
            agent: build
            ---

            Address accepted/material review findings for the current working change.

            Findings/instructions:
            $ARGUMENTS

            Fix only relevant issues, preserve `.review/` intent, run useful verification,
            summarize changes, and recommend `/review` again.
        """,
        "learn-from-change": """
            ---
            description: Extract durable engineering knowledge from a completed change
            agent: knowledge-curator
            ---

            Analyze this change for durable knowledge:
            $ARGUMENTS

            Classify NONE, LOCAL, TEAM, or ARCHITECTURAL.
            Prefer NONE. Only propose/write a `.review/knowledge/` record when warranted.
        """,
        "weekly-changes": """
            ---
            description: Select 3-5 recent changes for a focused team knowledge-sharing session
            agent: knowledge-curator
            ---

            Prepare an Interesting Changes agenda for:
            $ARGUMENTS

            Select only 3-5 changes. For each include what changed, why it matters,
            engineering lesson, who should care, one discussion question, and ~3-5 min timing.
        """,
    }
    for name, body in commands.items():
        files[f".opencode/commands/{name}.md"] = dedent(body)

    return files


def pi_agent(name, description, prompt, tools="read, grep, find, ls, bash"):
    return dedent(f"""
        ---
        name: {name}
        description: {description}
        tools: {tools}
        inheritProjectContext: true
        systemPromptMode: replace
        acceptanceRole: read-only
        completionGuard: false
        ---
        {prompt}
    """)


def pi_files():
    files = {}

    files[".pi/README.md"] = dedent("""
        # Pi Adapter

        This adapter uses:
        - project skills in `.pi/skills/`
        - project subagent definitions in `.pi/agents/`
        - the `pi-subagents` extension for delegation/orchestration

        Install once:

            pi install npm:pi-subagents

        Then start Pi in this repository and use:

            /skill:review
            /skill:review-brief
            /skill:review-fix
            /skill:learn-from-change
            /skill:weekly-changes
    """)

    files[".pi/settings.json"] = json.dumps(
        {
            "reviewToolkit": {
                "version": TOOLKIT_VERSION,
                "requires": ["npm:pi-subagents"]
            }
        },
        indent=2,
    ) + "\n"

    files[".pi/agents/reviewer-correctness.md"] = pi_agent(
        "reviewer-correctness",
        "Adversarial reviewer for functional correctness, edge cases, state, errors, retries, and regressions.",
        reviewer_prompt("correctness"),
    )
    files[".pi/agents/reviewer-tests.md"] = pi_agent(
        "reviewer-tests",
        "Reviewer that checks whether tests prove the behavioral claims of a change.",
        reviewer_prompt("tests"),
    )
    files[".pi/agents/reviewer-architecture.md"] = pi_agent(
        "reviewer-architecture",
        "Reviewer for architecture boundaries, dependencies, maintainability, ownership, and repository invariants.",
        reviewer_prompt("architecture"),
    )
    files[".pi/agents/reviewer-product-intent.md"] = pi_agent(
        "reviewer-product-intent",
        "Reviewer comparing implementation with product intent, requirements, and product invariants.",
        reviewer_prompt("product-intent"),
    )
    files[".pi/agents/change-historian.md"] = pi_agent(
        "change-historian",
        "Repository historian for git history, blame, previous decisions, and unusual code or boundaries.",
        HISTORIAN_PROMPT,
    )
    files[".pi/agents/review-adjudicator.md"] = pi_agent(
        "review-adjudicator",
        "Adjudicator that synthesizes reviewer evidence and narrows human review to specific questions.",
        ADJUDICATOR_PROMPT,
    )
    files[".pi/agents/knowledge-curator.md"] = pi_agent(
        "knowledge-curator",
        "Curates durable engineering knowledge and selects high-value changes for team learning.",
        KNOWLEDGE_PROMPT,
    )

    skill_review = dedent("""
        ---
        name: review
        description: Run independent specialist code reviewers in parallel, adjudicate their evidence, and return a merge/human-review decision. Use for reviewing the current branch or diff.
        ---

        # Review

        Use the `subagent` capability supplied by `pi-subagents`.

        User arguments, when supplied, define the review base/scope.

        ## Workflow

        1. Establish review scope.
           - Honor an explicit base/ref from the user.
           - Otherwise inspect git branch/status/merge-base and choose a sensible comparison.
           - State the chosen scope.
        2. Read:
           - `AGENTS.md` when present
           - `.review/policy.md`
           - `.review/architecture.md`
           - `.review/product-context.md`
        3. Gather shared diff/stat, changed files, relevant tests/CI, and available task/PR context.
        4. Launch these project subagents independently, preferably in parallel:
           - `reviewer-correctness`
           - `reviewer-tests`
           - `reviewer-architecture`
           - `reviewer-product-intent`
        5. Give each reviewer the same review scope and shared context, but do not include another
           reviewer's conclusions.
        6. Collect all reviewer outputs.
        7. Launch `review-adjudicator` with:
           - the four structured reviews
           - review scope
           - shared repository evidence
           - applicable CI/test results
        8. Return the adjudicator result first.
        9. Include specialist findings only as a compact appendix when useful.

        Do not edit repository files during review.
        If `subagent` is unavailable, explain that `pi-subagents` must be installed with:
        `pi install npm:pi-subagents`.
    """)
    files[".pi/skills/review/SKILL.md"] = skill_review

    files[".pi/skills/review-brief/SKILL.md"] = dedent("""
        ---
        name: review-brief
        description: Run the AI review ensemble and return only the narrow unresolved decisions a busy staff engineer needs to make.
        ---

        # Review Brief

        Follow the same workflow as the `review` skill:
        - independent specialist subagents
        - adjudicator after all specialists complete
        - read-only review

        Optimize the final response for human attention:

        1. Recommendation: CLEAR / HUMAN_INPUT_REQUIRED / BLOCKED
        2. What AI has already cleared
        3. The smallest set of unresolved human decisions
        4. Exact relevant files/lines
        5. Material residual uncertainty

        Do not dump routine reviewer commentary.
        If `subagent` is unavailable, instruct the user to install `pi-subagents`.
    """)

    files[".pi/skills/review-fix/SKILL.md"] = dedent("""
        ---
        name: review-fix
        description: Apply accepted AI review findings to the current working change, run verification, and prepare it for another independent review.
        ---

        # Review Fix

        This is an implementation workflow, not a reviewer.

        1. Inspect the current diff and review findings available in the session.
        2. Apply only accepted/material fixes; avoid unrelated rewrites.
        3. Preserve `.review/policy.md`, `.review/architecture.md`, and product intent.
        4. Run the narrowest useful tests/checks, then broader checks when warranted.
        5. Summarize:
           - fixes made
           - verification run
           - remaining uncertainty
        6. Recommend `/skill:review` again after fixing.
    """)

    files[".pi/skills/learn-from-change/SKILL.md"] = dedent("""
        ---
        name: learn-from-change
        description: Decide whether a completed change contains durable engineering knowledge and selectively capture it for future engineers.
        ---

        # Learn From Change

        Use the `knowledge-curator` subagent when available.

        Analyze the requested change/ref, or infer the most relevant completed/current change.

        Classify:
        - NONE
        - LOCAL
        - TEAM
        - ARCHITECTURAL

        Prefer NONE.

        Only for LOCAL/TEAM/ARCHITECTURAL:
        - propose a concise durable record
        - include why it matters, decision/trade-off, invariant/pattern, audience, references
        - store under `.review/knowledge/` only when the user explicitly requests writing or the
          active workflow clearly authorizes it
    """)

    files[".pi/skills/weekly-changes/SKILL.md"] = dedent("""
        ---
        name: weekly-changes
        description: Select 3-5 recent merged changes worth discussing in a short engineering knowledge-sharing session.
        ---

        # Weekly Interesting Changes

        Use the `knowledge-curator` subagent when useful.

        Inspect the supplied range/period. If none is supplied, use a sensible recent git window.

        Select only 3-5 changes based on:
        - architectural novelty
        - surprising bugs/lessons
        - changed patterns or invariants
        - reviewer disagreement
        - cross-component consequences
        - broadly reusable engineering learning

        For each:
        - what changed
        - why it matters
        - engineering lesson/decision/new pattern
        - who should care
        - one discussion question
        - suggested 3-5 minute discussion time

        Finish with a 20-30 minute agenda.
        Do not list every merged change.
    """)

    return files


def selected_files(harness):
    files = {}
    files.update(shared_files())
    if harness in ("opencode", "both"):
        files.update(opencode_files())
    if harness in ("pi", "both"):
        files.update(pi_files())
    return files


def write_file(path, content, *, force, dry_run, never_overwrite=False):
    exists = path.exists()

    if exists and never_overwrite:
        logging.info("Keeping existing %s", path)
        return "kept"

    if exists and not force:
        logging.warning("Skipping existing file: %s", path)
        return "skipped"

    action = "overwrite" if exists else "create"
    if dry_run:
        print(f"[dry-run] {action}: {path}")
        return action

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logging.info("%s: %s", action.capitalize(), path)
    return action


def install_pi_dependencies(*, root, dry_run):
    command = ["pi", "install", "npm:pi-subagents"]

    if dry_run:
        print("[dry-run] install:", " ".join(command))
        return "dry-run"

    if shutil.which("pi") is None:
        raise RuntimeError(
            "`pi` was not found on PATH. Install Pi first, then run:\n"
            "  pi install npm:pi-subagents"
        )

    print("Installing Pi dependency: pi-subagents")
    result = run_command(command, cwd=root)
    if result.stdout.strip():
        print(result.stdout.strip())
    return "installed"


def print_summary(root, harness, results, dry_run, pi_install_status=None):
    counts = {}
    for result in results.values():
        counts[result] = counts.get(result, 0) + 1

    print()
    print(f"AI review toolkit v{TOOLKIT_VERSION}")
    print(f"Repository: {root}")
    print(f"Harness: {harness}")
    if dry_run:
        print("Mode: dry-run")
    print()

    for key in ("create", "overwrite", "skipped", "kept"):
        if key in counts:
            print(f"{key:>10}: {counts[key]}")

    if pi_install_status:
        print(f"{'pi deps':>10}: {pi_install_status}")

    print()
    print("Shared methodology:")
    print("  .review/policy.md")
    print("  .review/architecture.md")
    print("  .review/product-context.md")
    print("  independent reviewers -> adjudicator -> narrow human decision")
    print("  selective knowledge capture -> Interesting Changes session")

    if harness in ("opencode", "both"):
        print()
        print("OpenCode:")
        print("  /review [base]")
        print("  /review-brief [base]")
        print("  /review-fix [findings]")
        print("  /learn-from-change [ref]")
        print("  /weekly-changes [range]")

    if harness in ("pi", "both"):
        print()
        print("Pi:")
        print("  /skill:review [base]")
        print("  /skill:review-brief [base]")
        print("  /skill:review-fix")
        print("  /skill:learn-from-change [ref]")
        print("  /skill:weekly-changes [range]")
        if not pi_install_status and not dry_run:
            print("  dependency: pi install npm:pi-subagents")

    if not dry_run:
        print()
        print("Customize before relying on the review:")
        print("  1. .review/architecture.md")
        print("  2. .review/product-context.md")
        print("  3. .review/policy.md")
        print()
        print("AGENTS.md is never overwritten if it already exists.")


def main(args):
    logging.debug("Arguments: %r", args)

    if args.install_pi_deps and args.harness == "opencode":
        raise RuntimeError("--install-pi-deps requires --harness pi or --harness both")

    root = validate_repo(args.root)
    files = selected_files(args.harness)

    results = {}
    for relative_path, content in files.items():
        results[relative_path] = write_file(
            root / relative_path,
            content,
            force=args.force,
            dry_run=args.dry_run,
            never_overwrite=(relative_path == "AGENTS.md"),
        )

    pi_install_status = None
    if args.install_pi_deps and args.harness in ("pi", "both"):
        pi_install_status = install_pi_dependencies(root=root, dry_run=args.dry_run)

    print_summary(
        root,
        args.harness,
        results,
        args.dry_run,
        pi_install_status=pi_install_status,
    )


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    try:
        main(args)
    except Exception as exc:
        logging.error("%s", exc)
        raise SystemExit(1)
