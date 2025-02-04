from r1.data.core import extract_task_response


def test_extract_task_response():
    response = extract_task_response(
        f"Assistant: <think>1=1</think> <answer>1</answer>"
    )
    assert response is not None
    assert response.answer == "1"
    assert response.rationale == "1=1"


def test_extract_task_response_no_assistant():
    response = extract_task_response(f"<think>1=1</think> <answer>1</answer>")
    assert response is None


def test_extract_task_no_think():
    response = extract_task_response(f"Assistant: <answer>1</answer>")
    assert response is not None
    assert response.answer == "1"
    assert response.rationale is None


def test_extract_task_no_answer():
    response = extract_task_response(f"Assistant: <answer>1")
    assert response is None
