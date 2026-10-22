import html
import os
import requests
import streamlit as st

# PAGE CONFIG
st.set_page_config(
    page_title="Barrister",
    layout="centered"
)


# API CONFIG
API_URL = os.getenv("BARRISTER_API_URL", "http://localhost:8000/chat").rstrip("/")

def get_barrister_response(question, history):
    """Send a question and conversation history to FastAPI."""

    response = requests.post(
        API_URL,
        json={
            "question": question,
            "history": history
        },
        timeout=120
    )

    response.raise_for_status()
    data = response.json()

    return data["answer"]


# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "active_request" not in st.session_state:
    st.session_state.active_request = None

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


# PILL CALLBACK
def handle_pill_selection():
    """Capture a pill selection once and immediately clear it."""

    selected = st.session_state.get("question_pills")

    if (
        selected
        and not st.session_state.is_processing
        and st.session_state.pending_prompt is None
    ):
        st.session_state.pending_prompt = selected

    # Prevent the selected pill from triggering again on rerun.
    st.session_state.question_pills = None


# STYLING
st.markdown("""
<style>
.header-divider {
    border-bottom: 1px solid rgba(128, 128, 128, 0.35);
    margin-top: 0;
    margin-bottom: 24px;
}

.chat-bubble {
    display: inline-block;
    max-width: 100%;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 15px;
    line-height: 1.5;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}

.user-bubble {
    background-color: #e5e7eb;
    color: #111827;
    border-bottom-left-radius: 4px;
}

.ai-bubble {
    background-color: #d9fdd3;
    color: #111827;
    border-bottom-right-radius: 4px;
}

.message-row {
    margin-bottom: 16px;
}

[data-testid="stChatInput"] {
    padding-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# WRAPPING THE UI IN A CONTAINER
with st.container(border=True):
    # HEADER
    col1, col2 = st.columns(
        [4, 1],
        vertical_alignment="center"
    )

    with col1:
        st.markdown("### Barrister")

    with col2:
        if st.button(
            "New chat",
            use_container_width=True,
            disabled=st.session_state.is_processing
        ):
            st.session_state.messages = []
            st.session_state.pending_prompt = None
            st.session_state.active_request = None
            st.session_state.is_processing = False

            st.session_state.pop("question_pills", None)
            st.session_state.pop("api_error", None)

            st.rerun()

    st.markdown(
        '<div class="header-divider"></div>',
        unsafe_allow_html=True
    )


    # WELCOME SCREEN
    if not st.session_state.messages:
        st.html("""
        <div style="
            font-size: 26px;
            font-weight: 500;
            line-height: 1.3;
        ">
            <span>Hello,</span>
            <br>
            <span>How can I help you today?</span>
        </div>
        """)

        st.html("""
        <p style="
            line-height: 1.1;
            font-size: 14px;
            font-weight: 500;
        ">
            Ask a question or choose from the options below
        </p>
        """)

        prelisted_questions = [
            "What are the common defences in criminal cases?",
            "What are the stages of criminal trial in Nigeria?",
            "What are the requirements for filing a fundamental right suit in Nigeria?",
        ]

        # Prelisted questions
        st.pills(
            "Prelisted Questions",
            prelisted_questions,
            key="question_pills",
            on_change=handle_pill_selection,
            disabled=st.session_state.is_processing
        )

        # Disclaimer
        st.markdown("""
        <div style="
            margin-top: 40px;
            margin-bottom: 10px;
            padding-top: 12px;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
            font-size: 12px;
            color: gray;
            line-height: 1.5;
        ">
            <strong>Disclaimer:</strong> Barrister provides AI-generated
            legal information for educational and research purposes only
            and should not be considered a substitute for professional
            legal advice.
        </div>
        """, unsafe_allow_html=True)


    # CHAT MESSAGES
    else:
        for message in st.session_state.messages:

            safe_content = html.escape(str(message["content"]))

            if message["role"] == "user":

                left, right = st.columns([3, 7])

                with right:
                    st.markdown(
                        f'<div class="message-row" '
                        f'style="text-align: right;">'
                        f'<div class="chat-bubble user-bubble" '
                        f'style="text-align: left;">'
                        f'{safe_content}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

            else:
                left, right = st.columns([7, 3])

                with left:
                    st.markdown(
                        f'<div class="message-row">'
                        f'<div class="chat-bubble ai-bubble">'
                        f'{safe_content}'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

        # Disclaimer
        st.markdown("""
        <div style="
            margin-top: 40px;
            margin-bottom: 10px;
            padding-top: 12px;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
            font-size: 12px;
            color: gray;
            line-height: 1.5;
        ">
            <strong>Disclaimer:</strong> Barrister provides AI-generated
            legal information for educational and research purposes only
            and should not be considered a substitute for professional
            legal advice.
        </div>
        """, unsafe_allow_html=True)


# CHAT INPUT (outside the container)
prompt = st.chat_input(
    "Ask a question about Nigerian law",
    disabled=st.session_state.is_processing,
    max_chars=2000,
)

# A pill selection takes priority over typed input.
if st.session_state.pending_prompt is not None:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


# START A REQUEST
if (
    prompt
    and prompt.strip()
    and not st.session_state.is_processing
    and st.session_state.active_request is None
):

    question = prompt.strip()

    # History excludes the current question.
    # Match the API's history limit and avoid sending the full session each turn.
    history = list(st.session_state.messages[-30:])

    # Store the request before rerunning.
    st.session_state.active_request = {
        "question": question,
        "history": history
    }

    # Add the user's message immediately.
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.is_processing = True

    # Render the updated conversation before calling the API.
    st.rerun()


# PROCESS THE REQUEST
if st.session_state.active_request is not None:

    request_data = st.session_state.active_request

    try:
        with st.spinner("Barrister is thinking..."):

            answer = get_barrister_response(
                request_data["question"],
                request_data["history"]
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.session_state.pop("api_error", None)

    except requests.exceptions.Timeout:
        st.session_state.api_error = (
            "The request took too long. Please try again."
        )

    except requests.exceptions.ConnectionError:
        st.session_state.api_error = (
            "Cannot connect to the backend. "
            "Check that your FastAPI server is running."
        )

    except requests.exceptions.HTTPError as error:
        st.session_state.api_error = (
            f"The backend returned an HTTP error: {error}"
        )

    except (ValueError, KeyError):
        st.session_state.api_error = (
            "The backend returned an invalid response. "
            "Expected JSON containing an 'answer' field."
        )

    except Exception as error:
        st.session_state.api_error = (
            f"An unexpected error occurred: {error}"
        )

    finally:
        st.session_state.active_request = None
        st.session_state.is_processing = False

    st.rerun()


# API ERRORS
if st.session_state.get("api_error"):
    st.error(
        st.session_state.api_error
        + " You can submit another question to try again."
    )
