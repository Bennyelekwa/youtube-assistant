"""Streamlit front-end for the YouTube Assistant.

Layout:
- URL field at the top.
- Three quick-action buttons (Summarize, Key takeaways, Action items) for
  one-click prompts.
- Once a URL is loaded, the embedded video appears above a chat history of
  user/assistant turns. Each assistant turn carries a "Sources used" expander
  showing the transcript chunks that grounded the answer.
- A persistent `st.chat_input` at the bottom for follow-up questions.

Key resolution and rate limiting carry over from the previous version: server
key first, daily-cap and quota-exhaustion both fall through to BYOK without
hiding the chat history.
"""

from __future__ import annotations

import streamlit as st

import langchain_helper as lch
import rate_limit


QUICK_ACTIONS: dict[str, str] = {
    "Summarize": (
        "Provide a concise summary of this video with the main topic and "
        "three to five key points."
    ),
    "Key takeaways": (
        "List the most important takeaways from this video as a bullet list, "
        "ordered by importance."
    ),
    "Action items": (
        "List any concrete actions, recommendations, or next steps mentioned "
        "in this video. If none are mentioned, say so."
    ),
}


st.set_page_config(
    page_title="YouTube Video Querying Assistant",
    page_icon=":clapper:",
    layout="centered",
)
st.markdown("<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Fetching transcript and building search index...")
def build_db(url: str, api_key: str):
    """Cache the FAISS index per (url, api_key) so repeated questions about
    the same video skip the embedding cost."""
    return lch.create_db_from_youtube_video_url(url, api_key)


def _resolve_api_key() -> tuple[str | None, str, int, int]:
    """Pick the right key for this render. Returns (api_key, source, remaining, cap)."""
    server_key = st.secrets.get("OPENAI_API_KEY")
    user_key = st.session_state.get("user_api_key")
    cap = int(st.secrets.get("DAILY_REQUEST_CAP", 5))
    ip = rate_limit.get_client_ip()
    remaining = rate_limit.remaining(ip, cap)

    if user_key:
        return user_key, "user", remaining, cap
    if not server_key or server_key == "sk-replace-me":
        return None, "needs_key", remaining, cap
    if st.session_state.get("server_key_exhausted"):
        return None, "needs_key", remaining, cap
    if remaining <= 0:
        return None, "needs_key", remaining, cap
    return server_key, "server", remaining, cap


def _persist_byok_key() -> None:
    """Mirror the BYOK widget's value into a stable session_state slot.

    Streamlit garbage-collects session_state entries whose owning widget
    isn't rendered on the current run. The BYOK input disappears as soon as
    a usable key is supplied, so binding the widget directly to
    `user_api_key` would cause it to be wiped on the very next rerun
    (manifesting as the BYOK prompt re-appearing right after the user
    pasted their key). Copying the value into `user_api_key` — which has
    no widget owner — keeps it alive for the rest of the session.
    """
    val = st.session_state.get("_byok_input", "")
    if isinstance(val, str) and val.strip():
        st.session_state["user_api_key"] = val.strip()


def _byok_block(reason: str) -> None:
    headlines = {
        "cap": "You've used today's free demo requests.",
        "exhausted": "The demo is temporarily out of free credits.",
        "missing": "No demo key is configured on the server.",
    }
    st.warning(
        f"{headlines.get(reason, headlines['missing'])} Paste your own OpenAI API "
        "key below to continue. The key stays in your browser session and is never stored."
    )
    st.text_input(
        "OpenAI API Key",
        type="password",
        key="_byok_input",
        on_change=_persist_byok_key,
        help="Get a key at https://platform.openai.com/account/api-keys",
    )
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")


def _stream_with_error_capture(token_gen):
    try:
        for token in token_gen:
            yield token
    except lch.ServerKeyUnusable:
        raise
    except Exception as e:
        if lch.is_server_key_unusable(e):
            raise lch.ServerKeyUnusable() from e
        raise


def _handle_server_key_failure(ip: str) -> None:
    rate_limit.refund(ip)
    st.session_state["server_key_exhausted"] = True
    build_db.clear()
    st.rerun()


def _render_sources(sources: list[str]) -> None:
    with st.expander("Sources used (transcript excerpts)"):
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**Chunk {i}**")
            st.write(src)


def _render_history() -> None:
    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "loaded_url" not in st.session_state:
    st.session_state["loaded_url"] = None


st.markdown(
    """
    <div style='text-align: center;'>
      <h1 style='margin-bottom: 0.2rem;'>
        <span style='color: #ea580c;'>Tube</span>Talks
      </h1>
      <h3 style='margin-top: 0; font-weight: 400; color: #6b4a33;'>
        YouTube Video Querying Assistant
      </h3>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Paste a YouTube link, then use a quick action or ask anything. "
    "Answers are grounded in the actual transcript. "
)
st.markdown(
    """
    <div style='font-size: 0.85rem; color: #6b4a33; margin-top: -0.4rem;'>
      <strong>Tips</strong>
      <ul style='margin: 0.25rem 0 0; padding-left: 1.2rem;'>
        <li>
          Only YouTube videos that have transcripts available can be used in this tool.
          Try this <a href='https://www.youtube.com/watch?v=arj7oStGLkU' target='_blank'>sample video</a>
          to see it in action.
        </li>
        <li>
          To switch to dark mode, open the <strong>⋮</strong> menu (top-right) →
          <strong>Settings</strong> → <strong>Theme</strong>.
        </li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key, source, remaining, cap = _resolve_api_key()
ip = rate_limit.get_client_ip()

if source == "server":
    st.caption(f"Free demo requests left today: **{remaining}/{cap}**")
elif source == "user":
    st.caption("Using your personal OpenAI key for this session.")

url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=...",
    key="url_input",
)

if url and url != st.session_state["loaded_url"]:
    st.session_state["chat_history"] = []
    st.session_state["loaded_url"] = url
elif not url and st.session_state["loaded_url"]:
    st.session_state["chat_history"] = []
    st.session_state["loaded_url"] = None

inputs_disabled = source == "needs_key" or not url

clicked_prompt: str | None = None
qa_cols = st.columns(len(QUICK_ACTIONS))
for col, (label, prompt) in zip(qa_cols, QUICK_ACTIONS.items()):
    if col.button(
        label,
        disabled=inputs_disabled,
        use_container_width=True,
        key=f"qa_{label}",
    ):
        clicked_prompt = prompt

if not url:
    st.caption("Paste a URL above to get started.")

if st.session_state["loaded_url"]:
    st.video(st.session_state["loaded_url"])
    _render_history()

if source == "needs_key":
    if st.session_state.get("server_key_exhausted"):
        _byok_block("exhausted")
    elif remaining <= 0:
        _byok_block("cap")
    else:
        _byok_block("missing")

chat_text = st.chat_input(
    "Ask anything about this video...",
    disabled=inputs_disabled,
)

# Resolve the query for this run. A fresh button click or chat submission
# wins; otherwise pick up a query stashed during a previous run (e.g. the
# user typed a question, the server key died mid-flight, BYOK was shown,
# and now they've supplied a working key — auto-replay so they don't have
# to retype).
new_query = clicked_prompt or chat_text or st.session_state.pop("pending_query", None)

if new_query and source == "needs_key":
    # Can't process yet. Hold onto the query while BYOK collects a key;
    # the next rerun (after the key is supplied) will pop and run it.
    st.session_state["pending_query"] = new_query
elif new_query and not inputs_disabled:
    if not url:
        st.error("Paste a YouTube URL first.")
        st.stop()

    if source == "server":
        allowed, _new_remaining = rate_limit.check_and_increment(ip, cap)
        if not allowed:
            st.session_state["pending_query"] = new_query
            st.rerun()

    try:
        db = build_db(url, api_key)
    except Exception as e:
        if lch.is_server_key_unusable(e):
            st.session_state["pending_query"] = new_query
            if source == "server":
                _handle_server_key_failure(ip)
            st.session_state.pop("user_api_key", None)
            st.error(
                "The OpenAI key was rejected (invalid or out of credit). "
                "Please paste a working key."
            )
            st.stop()
        st.error(f"Could not load this video's transcript: {e}")
        st.stop()

    with st.chat_message("user"):
        st.markdown(new_query)

    full_text = ""
    source_docs = []
    with st.chat_message("assistant"):
        try:
            token_gen, source_docs = lch.stream_response_from_query(db, new_query, api_key)
            full_text = st.write_stream(_stream_with_error_capture(token_gen))
        except lch.ServerKeyUnusable:
            st.session_state["pending_query"] = new_query
            if source == "server":
                _handle_server_key_failure(ip)
            st.session_state.pop("user_api_key", None)
            st.error(
                "The OpenAI key was rejected (invalid or out of credit). "
                "Please paste a working key."
            )
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong while generating the answer: {e}")
            st.stop()

        _render_sources([d.page_content for d in source_docs])

    st.session_state["chat_history"].append(
        {"role": "user", "content": new_query, "sources": None}
    )
    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": full_text,
            "sources": [d.page_content for d in source_docs],
        }
    )
