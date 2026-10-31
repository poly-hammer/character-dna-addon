"""Name transformations shared by editors, independent of Blender and DNA."""

import logging
import re


logger = logging.getLogger(__name__)

RAW_CONTROL_PREFIX = "CTRL_expressions."


def _swap_side_token(side_token: str) -> str:
    """Swap the ``L`` <-> ``R`` polarity of a regex ``side`` token, preserving
    case and any surrounding characters. Handles the whole-word ``left`` /
    ``right`` convention as well as single ``l`` / ``r`` letters (e.g. ``_L_``,
    ``_L``, ``L``, ``_left``)."""
    lowered = side_token.lower()
    if "left" in lowered:
        return re.sub("left", "right", side_token, flags=re.IGNORECASE)
    if "right" in lowered:
        return re.sub("right", "left", side_token, flags=re.IGNORECASE)
    return side_token.translate(str.maketrans("LlRr", "RrLl"))


def regex_side_mirror(name: str, regex_pattern: str) -> "str | None":
    """Return the ``L`` <-> ``R`` mirrored form of ``name`` resolved by
    ``regex_pattern``, or ``None`` when ``name`` has no detectable side.

    ``regex_pattern`` MUST contain a named group ``side`` capturing the side
    token (``l`` / ``r`` / ``left`` / ``right``). Only the matched ``side`` span
    is swapped, so a side token appearing elsewhere in the name is left intact.
    An invalid pattern or a pattern without a ``side`` group returns ``None``.
    """
    try:
        match = re.search(regex_pattern, name)
    except re.error:
        return None
    if match is None:
        return None
    try:
        side_token = match.group("side") or ""
    except IndexError:  # pattern lacks a 'side' named group
        return None
    if not side_token:
        return None
    swapped = _swap_side_token(side_token)
    if swapped == side_token:
        return None
    start, end = match.span("side")
    return name[:start] + swapped + name[end:]


def classify_side_by_regex(
    name: str,
    regex_pattern: str,
    known_names: "set[str] | None" = None,
) -> str:
    """Classify ``name`` as ``"L"`` / ``"R"`` / ``"C"`` using ``regex_pattern``.

    ``regex_pattern`` MUST contain a named group ``side`` capturing the side
    token (``l`` / ``r`` / ``left`` / ``right``); that token determines the side.
    Names that do not match -- or an invalid pattern / missing ``side`` group --
    classify as ``"C"`` (center).

    When ``known_names`` is provided the classification is **twin-gated**: a name
    is only sided when its regex mirror also appears in ``known_names``. This
    rejects false positives (a center name that incidentally ends in a side
    letter) and mirrors the behavior of the raw-control side classifier. With
    ``known_names`` omitted the classification is pure name parsing.
    """
    mirror = regex_side_mirror(name, regex_pattern)
    if mirror is None:
        return "C"
    if known_names is not None and (mirror == name or mirror not in known_names):
        return "C"
    match = re.search(regex_pattern, name)  # pattern already validated by the mirror
    side_token = match.group("side") if match else ""  # pyright: ignore[reportOptionalMemberAccess]
    letter = next((char for char in (side_token or "") if char in "LlRr"), "")
    if letter in "Ll":
        return "L"
    if letter in "Rr":
        return "R"
    return "C"


def get_mirror_side_replacement(source_side: str) -> str:
    """Return the opposite-side string for ``source_side``, preserving case.

    Supports the four conventions surfaced by the addon-preferences
    ``raw_control_editor_bone_mirror_regex_pattern`` named ``side``
    group: ``_L_`` / ``_R_`` infix (case-insensitive) and ``_L`` /
    ``_R`` suffix (case-insensitive). The case of each ``L``/``R``
    character is preserved in the result so ``FACIAL_L_*`` mirrors to
    ``FACIAL_R_*`` and ``thigh_l`` mirrors to ``thigh_r``. Returns
    ``source_side`` unchanged if no convention matches.
    """
    swap = {"l": "r", "L": "R", "r": "l", "R": "L"}
    # Infix: _L_ / _R_ (or lowercase) -- swap the middle character.
    if len(source_side) == 3 and source_side[0] == "_" and source_side[2] == "_" and source_side[1] in swap:
        return f"_{swap[source_side[1]]}_"
    # Suffix: _L / _R (or lowercase) -- swap the trailing character.
    if len(source_side) == 2 and source_side[0] == "_" and source_side[1] in swap:
        return f"_{swap[source_side[1]]}"
    # Bare single-character L / R (used by raw control names like 'browDownL').
    if len(source_side) == 1 and source_side in swap:
        return swap[source_side]
    return source_side


def get_mirrored_name(name: str, regex_pattern: str) -> "str | None":
    """Return the L↔R mirrored bone name resolved by ``regex_pattern``,
    or ``None`` if ``name`` does not match the pattern or has no
    side marker.

    ``regex_pattern`` MUST contain a named group ``side`` that captures
    the side token (e.g. ``_l`` / ``_r`` / ``_L_`` / ``_R_``). The
    swap is applied to the matched span only (NOT via ``str.replace``)
    so other occurrences of the side token elsewhere in the name are
    left untouched.
    """
    try:
        match = re.match(regex_pattern, name)
    except re.error:
        logger.warning("Invalid bone mirror regex pattern: %s", regex_pattern)
        return None
    if not match:
        return None
    try:
        source_side = match.group("side")
    except IndexError:
        return None
    if source_side is None:
        return None
    target_side = get_mirror_side_replacement(source_side)
    if source_side == target_side:
        return None
    start, end = match.span("side")
    return name[:start] + target_side + name[end:]


_RAW_CONTROL_PH_SUFFIX_RE = re.compile(r"^(?P<prefix>.+?)(?P<side>[LR])(?P<suffix>Ph\d+)$")


_RAW_CONTROL_WORD_SIDE_RE = re.compile(r"^(?P<prefix>.+?)(?P<side>Left|Right)(?P<suffix>[A-Z]?)$")


_RAW_CONTROL_TRAILING_SIDE_RE = re.compile(r"^(?P<prefix>.+?)(?P<side>[LR])$")


_WORD_SIDE_SWAP = {"Left": "Right", "Right": "Left"}


_LETTER_SIDE_SWAP = {"L": "R", "R": "L"}


def mirror_raw_control_name(short_name: str) -> "tuple[str | None, str | None]":
    """Return ``(mirrored_short_name, side)`` for a ``CTRL_expressions.*``
    raw control's short name (the part AFTER the universal prefix), or
    ``(None, None)`` when ``short_name`` has no detectable side.

    ``side`` is ``"L"`` / ``"R"``: ``"L"`` is returned for any rule
    whose matched ``side`` token is ``"L"`` or ``"Left"``, and ``"R"``
    for ``"R"`` or ``"Right"``. Rules are tried in priority order
    (``Ph`` infix first to avoid the trailing-letter rule swallowing
    the ``L`` in ``...LPh1``; then trailing single ``L`` / ``R`` so
    ``eyeLookLeftL`` mirrors via the trailing token to
    ``eyeLookLeftR`` rather than ``eyeLookRightL``; then whole-word
    ``Left`` / ``Right`` for names that end on a non-side character
    like ``mouthLeft``, ``teethLeftU``, ``tongueTipLeft``).

    Pure name transformation -- no DNA lookup, no twin-exists check.
    Use :func:`classify_raw_control_side` for the twin-gated classifier.
    """
    # Rule 1: ``...<L|R>Ph<digit>`` (must run BEFORE trailing-letter)
    match = _RAW_CONTROL_PH_SUFFIX_RE.match(short_name)
    if match is not None:
        side_token = match.group("side")
        mirrored = match.group("prefix") + _LETTER_SIDE_SWAP[side_token] + match.group("suffix")
        return mirrored, side_token

    # Rule 2: trailing single uppercase ``L`` / ``R``. Runs BEFORE the
    # whole-word rule so ``eyeLookLeftL`` (``Left`` is the look
    # direction here, not a side token) mirrors to ``eyeLookLeftR``.
    match = _RAW_CONTROL_TRAILING_SIDE_RE.match(short_name)
    if match is not None:
        side_token = match.group("side")
        mirrored = match.group("prefix") + _LETTER_SIDE_SWAP[side_token]
        return mirrored, side_token

    # Rule 3: whole-word ``Left`` / ``Right`` (optionally followed by a
    # single uppercase letter suffix like ``teethLeftU``).
    match = _RAW_CONTROL_WORD_SIDE_RE.match(short_name)
    if match is not None:
        side_token = match.group("side")
        swapped = _WORD_SIDE_SWAP[side_token]
        mirrored = match.group("prefix") + swapped + match.group("suffix")
        return mirrored, "L" if side_token == "Left" else "R"  # noqa: S105

    return None, None


def classify_raw_control_side(
    short_name: str,
    known_short_names: "set[str]",
) -> str:
    """Classify a raw control short name (post-``CTRL_expressions.``
    prefix) as ``"L"``, ``"R"``, or ``"C"`` (center).

    Returns ``"L"`` / ``"R"`` only when :func:`mirror_raw_control_name`
    detects a side AND the mirrored partner is present in
    ``known_short_names``. Otherwise returns ``"C"``. The twin-exists
    gate prevents false positives where a name accidentally matches a
    side rule but has no real partner in this particular DNA.
    """
    mirrored, side = mirror_raw_control_name(short_name)
    if side is None or mirrored is None or mirrored == short_name:
        return "C"
    if mirrored not in known_short_names:
        return "C"
    return side


def rbf_mirror_side_replacement(source_side: str) -> str:
    """
    Get the replacement string for the opposite side, preserving case.

    Supports both the ``_L_`` / ``_R_`` infix (case-insensitive) and the
    ``_L`` / ``_R`` suffix (case-insensitive) conventions. The case of
    each ``L`` / ``R`` character is preserved in the result so
    ``FACIAL_L_*`` mirrors to ``FACIAL_R_*`` and ``thigh_l`` mirrors to
    ``thigh_r``.

    Args:
        source_side: The matched side string (e.g., ``_L_``, ``_r_``, ``_L``, ``_r``).

    Returns:
        The opposite side string with matching format, or ``source_side``
        unchanged if no convention matches.
    """
    swap = {"l": "r", "L": "R", "r": "l", "R": "L"}
    # Infix: _L_ / _R_ (or lowercase) -- swap the middle character.
    if len(source_side) == 3 and source_side[0] == "_" and source_side[2] == "_" and source_side[1] in swap:
        return f"_{swap[source_side[1]]}_"
    # Suffix: _L / _R (or lowercase) -- swap the trailing character.
    if len(source_side) == 2 and source_side[0] == "_" and source_side[1] in swap:
        return f"_{swap[source_side[1]]}"
    return source_side


def rbf_mirrored_name(name: str, regex_pattern: str) -> str | None:
    try:
        match = re.match(regex_pattern, name)
        if not match:
            return None

        groups = match.groupdict()
        if "side" not in groups or groups["side"] is None:
            return None

        source_side = groups["side"]
        target_side = rbf_mirror_side_replacement(source_side)

        if source_side == target_side:
            return None  # No valid mirror side found

        # Swap only the matched span -- avoids double-substitution if the
        # side token (e.g. ``_l``) appears elsewhere in the name.
        start, end = match.span("side")
        return name[:start] + target_side + name[end:]
    except re.error:
        logger.warning(f"Invalid regex pattern: {regex_pattern}")
        return None


def can_mirror_rbf_name(name: str, regex_pattern: str) -> bool:
    return rbf_mirrored_name(name, regex_pattern) is not None


DEFAULT_SIDE_MIRROR_REGEX = r"(?P<prefix>.+?)(?P<side>_[LlRr]_|_[Ll]eft(?=_|$)|_[Rr]ight(?=_|$)|[LR](?=_|$))"


DEFAULT_SOLVER_MIRROR_REGEX = r"(?P<prefix>.+?)(?P<side>_[LlRr]_)(?P<suffix>.+)?"


DEFAULT_POSE_MIRROR_REGEX = DEFAULT_SOLVER_MIRROR_REGEX


DEFAULT_BONE_MIRROR_REGEX = r"(?P<prefix>.+?)(?P<side>_[LlRr]_|_[LlRr]$)"
