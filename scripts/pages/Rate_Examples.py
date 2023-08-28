from datetime import datetime
from eyeball_pp import (
    set_config,
    get_default_recorder,
    Checkpoint,
    EvalRecorder,
    OutputFeedback,
    FeedbackResult,
    TASK_OUTPUT_KEY,
    time_to_str,
)
from eyeball_pp.classes import MultiOutputFeedback
import streamlit as st
from typing import Optional, Any
from fire import Fire


FEEDBACK_MAP = {
    "👍": FeedbackResult.POSITIVE,
    "😐": FeedbackResult.NEUTRAL,
    "👎": FeedbackResult.NEGATIVE,
}


@st.cache_data
def get_recorder() -> EvalRecorder:
    return get_default_recorder()


def render_divider() -> None:
    st.write('<hr style="margin: 0; padding: 0;">', unsafe_allow_html=True)


def render_variable(title: str, value: Any) -> None:
    st.markdown(f"**{title}**: {value}")


def scroll_to_top():
    st.markdown("""
        <script>
        window.scrollTo(0,0);
        </script>
        """, unsafe_allow_html=True)


def submit_feedback(
        task_name: str, checkpoint: Checkpoint, rating: str, feedback: str
) -> None:
    st.session_state.index += 1
    scroll_to_top()
    output_feedback = MultiOutputFeedback({
        TASK_OUTPUT_KEY: OutputFeedback(
            result=FEEDBACK_MAP[rating],
            message=feedback)
    })
    get_recorder().record_output_feedback(
        task_name=task_name,
        checkpoint_id=checkpoint.checkpoint_id,
        feedback=output_feedback,
    )


def render_feedback_form(
        task_name: str, checkpoint: Checkpoint, num_checkpoints: int
) -> None:
    with st.form(key="rating_form"):
        checkpoint_time = datetime.fromisoformat(checkpoint.checkpoint_id)
        st.markdown(f"### Example {st.session_state.index + 1} of {num_checkpoints} ({time_to_str(checkpoint_time)})")  # noqa
        for title, value in checkpoint.input_variables.items():
            render_variable(title, value)
        render_divider()
        for title, value in checkpoint.intermediary_state.items():
            render_variable(title, value)
        render_divider()
        render_variable("output", checkpoint.output)
        render_divider()

        rating = st.radio(
            "**Rating**", FEEDBACK_MAP.keys(), horizontal=True,
            key=f"rating-{checkpoint.checkpoint_id}")
        feedback = st.text_input(
            "**Feedback**", key=f"feedback-{checkpoint.checkpoint_id}")
        st.form_submit_button(
            "Submit",
            on_click=submit_feedback,
            args=(task_name, checkpoint, rating, feedback))


def render_rater(task_name: str, checkpoints: list[Checkpoint]) -> None:
    if checkpoints is None or len(checkpoints) == 0:
        st.write("No examples to rate")
    elif st.session_state.index >= len(checkpoints):
        st.write("You're done! All examples rated.")
    else:
        checkpoint = checkpoints[st.session_state.index]
        render_feedback_form(task_name, checkpoint, len(checkpoints))


@st.cache_data(show_spinner=False)
def get_checkpoints_to_rate(task_name: str) -> list[Checkpoint]:
    recorder = get_recorder()
    checkpoints_to_rate: list[Checkpoint] = []
    input_hashes = recorder.get_input_hashes(task_name=task_name)

    for input_hash in input_hashes:
        checkpoints = recorder.get_latest_checkpoints(
            task_name, input_hash, num_checkpoints=4
        )
        for checkpoint in checkpoints:
            if checkpoint.output and not checkpoint.feedback:
                checkpoints_to_rate.append(checkpoint)

    return checkpoints_to_rate


def render_page(
    task_name: Optional[str] = None, eyeball_config: Optional[dict] = None
) -> None:
    if eyeball_config is None:
        eyeball_config = {}

    set_config(**eyeball_config)

    if 'index' not in st.session_state:
        st.session_state.index = 0
    if 'last_checkpoint' not in st.session_state:
        st.session_state.last_checkpoint = None

    st.markdown("# Rate Recorded Examples")

    recorder = get_recorder()
    if task_name is None:
        task_name = st.sidebar.selectbox("Task", recorder.get_task_names())

    if st.sidebar.button("Refresh Data"):
        st.session_state.index = 0
        st.cache_data.clear()

    checkpoints = get_checkpoints_to_rate(task_name)
    render_rater(task_name, checkpoints)


if __name__ == "__main__":
    Fire(render_page)
